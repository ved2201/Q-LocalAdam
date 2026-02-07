
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
import random
import os
import json
import time
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_memory / 1e9:.2f} GB")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(" TF32 enabled for faster training\n")

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

class BasicBlockCustom(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return self.relu(out)


class MidResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 48
        self.conv1 = nn.Conv2d(3, 48, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(48, 2, stride=1)
        self.layer2 = self._make_layer(96, 2, stride=2)
        self.layer3 = self._make_layer(192, 2, stride=2)
        self.layer4 = self._make_layer(384, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(384, 10)

    def _make_layer(self, planes, blocks, stride):
        layers = []
        layers.append(BasicBlockCustom(self.inplanes, planes, stride))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlockCustom(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def dirichlet_split(dataset, num_clients=5, alpha=0.1, seed=42):
    np.random.seed(seed)
    labels = np.array(dataset.targets)
    num_classes = 10
    idx = [np.where(labels == i)[0] for i in range(num_classes)]
    clients = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        np.random.shuffle(idx[c])
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions / proportions.sum()) * len(idx[c])
        proportions = proportions.astype(int)
        start = 0
        for i in range(num_clients):
            take = proportions[i]
            clients[i].extend(idx[c][start:start + take])
            start += take

    return [Subset(dataset, ids) for ids in clients]


def iid_split(dataset, num_clients=5, seed=42):
    np.random.seed(seed)
    n = len(dataset)
    idxs = np.random.permutation(n)
    client_size = n // num_clients
    clients = []
    for i in range(num_clients):
        start = i * client_size
        end = start + client_size if i < num_clients - 1 else n
        clients.append(Subset(dataset, idxs[start:end].tolist()))
    return clients


class BlockwiseQuantizer:
    def __init__(self, block_size=64):
        self.block_size = block_size

    def quantize(self, t):
        orig_shape = t.shape
        flat = t.reshape(-1)
        n = flat.numel()
        pad = (self.block_size - n % self.block_size) % self.block_size
        if pad:
            flat = torch.cat([flat, torch.zeros(pad, device=t.device, dtype=t.dtype)])
        blocks = flat.view(-1, self.block_size)
        mins = blocks.min(dim=1, keepdim=True)[0]
        maxs = blocks.max(dim=1, keepdim=True)[0]
        scales = torch.clamp(maxs - mins, min=1e-8)
        q = (((blocks - mins) / scales) * 255).round().clamp(0, 255).to(torch.uint8)
        return (q.cpu(), mins.cpu().float(), maxs.cpu().float(), 
                scales.cpu().float(), pad, orig_shape, n)

    def dequantize(self, q, mins, maxs, scales, pad, orig_shape, n):
        blocks = (q.float() / 255.0) * scales + mins
        flat = blocks.reshape(-1)
        if pad:
            flat = flat[:n]
        return flat.reshape(orig_shape)


class LogQuantizer:
    def __init__(self, block_size=64, eps=1e-8):
        self.block_size = block_size
        self.eps = eps

    def quantize(self, t):
        orig_shape = t.shape
        flat = t.reshape(-1)
        n = flat.numel()
        pad = (self.block_size - n % self.block_size) % self.block_size
        if pad:
            flat = torch.cat([flat, self.eps * torch.ones(pad, device=t.device, dtype=t.dtype)])
        logs = torch.log(flat + self.eps)
        logs = torch.nan_to_num(logs, nan=0.0, posinf=10.0, neginf=-10.0)
        blocks = logs.view(-1, self.block_size)
        mins = blocks.min(dim=1, keepdim=True)[0]
        maxs = blocks.max(dim=1, keepdim=True)[0]
        scales = torch.clamp(maxs - mins, min=1e-4)
        q = (((blocks - mins) / scales) * 255).round().clamp(0, 255).to(torch.uint8)
        return (q.cpu(), mins.cpu().float(), maxs.cpu().float(), 
                scales.cpu().float(), pad, orig_shape, n)

    def dequantize(self, q, mins, maxs, scales, pad, orig_shape, n):
        logs = (q.float() / 255.0) * scales + mins
        flat = torch.exp(logs) - self.eps
        flat = torch.clamp(flat, min=self.eps)
        flat = flat.reshape(-1)
        if pad:
            flat = flat[:n]
        return flat.reshape(orig_shape)


class VanillaClientAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                st = self.state.setdefault(p, {})
                if "m" not in st:
                    st["m"] = torch.zeros_like(p.data)
                    st["v"] = torch.zeros_like(p.data)
                    st["step"] = 0

                st["step"] += 1
                m, v = st["m"], st["v"]
                g = p.grad.data

                m.mul_(b1).add_(g, alpha=1-b1)
                v.mul_(b2).addcmul_(g, g, value=1-b2)

                m_hat = m / (1 - b1 ** st["step"])
                v_hat = v / (1 - b2 ** st["step"])

                p.data.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)


class QLocalAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, block_size=64):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.Qm = BlockwiseQuantizer(block_size=block_size)
        self.Qv = LogQuantizer(block_size=block_size, eps=eps)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                st = self.state.setdefault(p, {})

                if "step" not in st:
                    st["step"] = 0
                    st["m_quantized"] = None
                    st["v_quantized"] = None

                st["step"] += 1
                g = p.grad.data

                if st["m_quantized"] is not None:
                    m = self.Qm.dequantize(*st["m_quantized"]).to(p.device)
                else:
                    m = torch.zeros_like(p.data)

                if st["v_quantized"] is not None:
                    v = self.Qv.dequantize(*st["v_quantized"]).to(p.device)
                else:
                    v = torch.zeros_like(p.data)

                m = m.mul(b1).add_(g, alpha=1-b1)
                v = v.mul(b2).addcmul_(g, g, value=1-b2)

                st["m_quantized"] = self.Qm.quantize(m.cpu())
                st["v_quantized"] = self.Qv.quantize(v.cpu())

                m_hat = m / (1 - b1 ** st["step"])
                v_hat = v / (1 - b2 ** st["step"])

                p.data.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)


class NaiveINT8(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, block_size=64):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.Qm = BlockwiseQuantizer(block_size=block_size)
        self.Qv = BlockwiseQuantizer(block_size=block_size)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                st = self.state.setdefault(p, {})

                if "step" not in st:
                    st["step"] = 0
                    st["m_quantized"] = None
                    st["v_quantized"] = None

                st["step"] += 1
                g = p.grad.data

                if st["m_quantized"] is not None:
                    m = self.Qm.dequantize(*st["m_quantized"]).to(p.device)
                else:
                    m = torch.zeros_like(p.data)

                if st["v_quantized"] is not None:
                    v = self.Qv.dequantize(*st["v_quantized"]).to(p.device)
                else:
                    v = torch.zeros_like(p.data)

                m = m.mul(b1).add_(g, alpha=1-b1)
                v = v.mul(b2).addcmul_(g, g, value=1-b2)

                st["m_quantized"] = self.Qm.quantize(m.cpu())
                st["v_quantized"] = self.Qv.quantize(v.cpu())

                m_hat = m / (1 - b1 ** st["step"])
                v_hat = v / (1 - b2 ** st["step"])

                p.data.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)


class QLocalAdam_MomentumOnly(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, block_size=64):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.Qm = BlockwiseQuantizer(block_size=block_size)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                st = self.state.setdefault(p, {})

                if "step" not in st:
                    st["step"] = 0
                    st["m_quantized"] = None
                    st["v"] = torch.zeros_like(p.data)

                st["step"] += 1
                g = p.grad.data

                if st["m_quantized"] is not None:
                    m = self.Qm.dequantize(*st["m_quantized"]).to(p.device)
                else:
                    m = torch.zeros_like(p.data)

                m = m.mul(b1).add_(g, alpha=1-b1)
                st["v"].mul_(b2).addcmul_(g, g, value=1-b2)

                st["m_quantized"] = self.Qm.quantize(m.cpu())

                m_hat = m / (1 - b1 ** st["step"])
                v_hat = st["v"] / (1 - b2 ** st["step"])

                p.data.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)


class QLocalAdam_VarianceOnly(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, block_size=64):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.Qv = LogQuantizer(block_size=block_size, eps=eps)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                st = self.state.setdefault(p, {})

                if "step" not in st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p.data)
                    st["v_quantized"] = None

                st["step"] += 1
                g = p.grad.data

                if st["v_quantized"] is not None:
                    v = self.Qv.dequantize(*st["v_quantized"]).to(p.device)
                else:
                    v = torch.zeros_like(p.data)

                st["m"].mul_(b1).add_(g, alpha=1-b1)
                v = v.mul(b2).addcmul_(g, g, value=1-b2)

                st["v_quantized"] = self.Qv.quantize(v.cpu())

                m_hat = st["m"] / (1 - b1 ** st["step"])
                v_hat = v / (1 - b2 ** st["step"])

                p.data.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)


def measure_optimizer_memory(optimizer):
    total_m_bytes = 0
    total_v_bytes = 0

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p in optimizer.state:
                st = optimizer.state[p]

                if "m" in st:
                    total_m_bytes += st["m"].numel() * st["m"].element_size()
                if "v" in st:
                    total_v_bytes += st["v"].numel() * st["v"].element_size()

                if "m_quantized" in st and st["m_quantized"] is not None:
                    qm = st["m_quantized"]
                    total_m_bytes += qm[0].numel() * qm[0].element_size()
                    total_m_bytes += qm[1].numel() * qm[1].element_size()
                    total_m_bytes += qm[2].numel() * qm[2].element_size()
                    total_m_bytes += qm[3].numel() * qm[3].element_size()

                if "v_quantized" in st and st["v_quantized"] is not None:
                    qv = st["v_quantized"]
                    total_v_bytes += qv[0].numel() * qv[0].element_size()
                    total_v_bytes += qv[1].numel() * qv[1].element_size()
                    total_v_bytes += qv[2].numel() * qv[2].element_size()
                    total_v_bytes += qv[3].numel() * qv[3].element_size()

    return {
        "momentum_mb": total_m_bytes / 1e6,
        "variance_mb": total_v_bytes / 1e6,
        "total_mb": (total_m_bytes + total_v_bytes) / 1e6
    }


def measure_model_memory(model):
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / 1e6


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def federated_averaging(client_states, client_weights):
    global_state = OrderedDict()
    total_weight = sum(client_weights)

    for key in client_states[0].keys():
        global_state[key] = sum(
            client_states[i][key] * (client_weights[i] / total_weight)
            for i in range(len(client_states))
        )

    return global_state


def run_federated_experiment(
    method_name="VanillaClientAdam",
    num_clients=5,
    num_rounds=120,
    local_epochs=2,
    alpha=0.1,
    lr=1e-3,
    batch_size=64,
    block_size=64,
    seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"\n{'='*70}")
    if alpha == "iid":
        print(f"Running: {method_name} | IID | {num_rounds} rounds")
    else:
        print(f"Running: {method_name} | α={alpha} (Non-IID) | {num_rounds} rounds")
    print(f"{'='*70}\n")

    transform_train = T.Compose([
        T.Resize(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    if alpha == "iid":
        client_datasets = iid_split(trainset, num_clients=num_clients, seed=seed)
    else:
        client_datasets = dirichlet_split(trainset, num_clients=num_clients, alpha=alpha, seed=seed)

    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    global_model = MidResNet18().to(device)
    model_memory_mb = measure_model_memory(global_model)

    results = {
        "method": method_name,
        "alpha": alpha,
        "rounds": [],
        "test_accuracies": [],
        "model_memory_mb": model_memory_mb,
        "momentum_memory_mb": [],
        "variance_memory_mb": [],
        "total_optimizer_memory_mb": []
    }

    start_time = time.time()

    for round_num in range(1, num_rounds + 1):
        client_states = []
        client_weights = []

        for client_id in range(num_clients):
            local_model = MidResNet18().to(device)
            local_model.load_state_dict(global_model.state_dict())

            if method_name == "VanillaClientAdam" or method_name.startswith("VanillaClientAdam"):
                optimizer = VanillaClientAdam(local_model.parameters(), lr=lr)
            elif "NaiveINT8" in method_name:
                optimizer = NaiveINT8(local_model.parameters(), lr=lr, block_size=block_size)
            elif "MomentumOnly" in method_name:
                optimizer = QLocalAdam_MomentumOnly(local_model.parameters(), lr=lr, block_size=block_size)
            elif "VarianceOnly" in method_name:
                optimizer = QLocalAdam_VarianceOnly(local_model.parameters(), lr=lr, block_size=block_size)
            elif "Q" in method_name or "QLocalAdam" in method_name:
                optimizer = QLocalAdam(local_model.parameters(), lr=lr, block_size=block_size)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            client_loader = DataLoader(
                client_datasets[client_id], 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=2
            )

            for epoch in range(local_epochs):
                train_loss, train_acc = train_one_epoch(local_model, client_loader, optimizer, device)

            if client_id == 0:
                opt_mem = measure_optimizer_memory(optimizer)
                results["momentum_memory_mb"].append(opt_mem["momentum_mb"])
                results["variance_memory_mb"].append(opt_mem["variance_mb"])
                results["total_optimizer_memory_mb"].append(opt_mem["total_mb"])

            client_states.append(local_model.state_dict())
            client_weights.append(len(client_datasets[client_id]))

            del local_model, optimizer

        torch.cuda.empty_cache()

        global_state = federated_averaging(client_states, client_weights)
        global_model.load_state_dict(global_state)

        test_acc = evaluate(global_model, test_loader, device)
        results["rounds"].append(round_num)
        results["test_accuracies"].append(test_acc)

        elapsed = time.time() - start_time
        if round_num % 10 == 0 or round_num <= 5:
            print(f"Round {round_num:3d} | Acc={test_acc:5.2f}% | "
                  f"OptMem={results['total_optimizer_memory_mb'][-1]:5.2f}MB | "
                  f"{elapsed/60:5.1f}min")

    results["final_accuracy"] = results["test_accuracies"][-1]
    results["best_accuracy"] = max(results["test_accuracies"])
    results["avg_momentum_mb"] = float(np.mean(results["momentum_memory_mb"]))
    results["avg_variance_mb"] = float(np.mean(results["variance_memory_mb"]))
    results["avg_total_opt_mb"] = float(np.mean(results["total_optimizer_memory_mb"]))
    results["total_time_hours"] = (time.time() - start_time) / 3600

    print(f" Completed: Final={results['final_accuracy']:.2f}%, "
          f"Best={results['best_accuracy']:.2f}%, "
          f"Time={results['total_time_hours']:.2f}h\n")

    return results


def plot_all_results(all_results):
    plt.rcParams["font.size"] = 13
    plt.rcParams["axes.labelsize"] = 15

    fig1, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(all_results["vanilla_iid"]["rounds"], 
            all_results["vanilla_iid"]["test_accuracies"], 
            color='#1f77b4', linewidth=3, label="Vanilla-ClientAdam (IID)", marker='o', markevery=15, markersize=6)
    ax.plot(all_results["vanilla_noniid"]["rounds"], 
            all_results["vanilla_noniid"]["test_accuracies"], 
            color='#d62728', linewidth=3, label="Vanilla-ClientAdam (α=0.1, Extreme Non-IID)", 
            marker='s', markevery=15, markersize=6)
    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("IID vs Non-IID Comparison", fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig1.savefig("figures/1_iid_vs_noniid.png", dpi=300, bbox_inches="tight")
    fig1.savefig("figures/1_iid_vs_noniid.pdf", bbox_inches="tight")
    plt.close(fig1)

    fig2, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(all_results["vanilla_noniid"]["rounds"], 
            all_results["vanilla_noniid"]["test_accuracies"], 
            color='#1f77b4', linewidth=3, label="Vanilla-ClientAdam", marker='o', markevery=15, markersize=6)
    ax.plot(all_results["qlocaladam"]["rounds"], 
            all_results["qlocaladam"]["test_accuracies"], 
            color='#d62728', linewidth=3, linestyle='--', label="Q-LocalAdam (Ours)", 
            marker='s', markevery=15, markersize=6)
    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Main Comparison: Vanilla-ClientAdam vs Q-LocalAdam (α=0.1)", fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig2.savefig("figures/2_main_comparison.png", dpi=300, bbox_inches="tight")
    fig2.savefig("figures/2_main_comparison.pdf", bbox_inches="tight")
    plt.close(fig2)

    fig3, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(all_results["qlocaladam"]["rounds"], 
            all_results["qlocaladam"]["test_accuracies"], 
            color='#2ca02c', linewidth=3, label="Q-LocalAdam (Log-space)", marker='o', markevery=15, markersize=6)
    ax.plot(all_results["naive_int8"]["rounds"], 
            all_results["naive_int8"]["test_accuracies"], 
            color='#ff7f0e', linewidth=3, linestyle=':', label="Naive INT8 (Linear)", 
            marker='^', markevery=15, markersize=6)
    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Necessity of Log-Space Quantization", fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig3.savefig("figures/3_naive_comparison.png", dpi=300, bbox_inches="tight")
    fig3.savefig("figures/3_naive_comparison.pdf", bbox_inches="tight")
    plt.close(fig3)

    print(" Saved 3  figures to figures/\n")


def create_summary_table(all_results):
    data = []
    for key, res in all_results.items():
        data.append({
            "Experiment": res["method"],
            "Alpha": res["alpha"],
            "Final Acc (%)": f"{res['final_accuracy']:.2f}",
            "Best Acc (%)": f"{res['best_accuracy']:.2f}",
            "Opt Memory (MB)": f"{res['avg_total_opt_mb']:.2f}",
            "Time (h)": f"{res['total_time_hours']:.2f}"
        })

    df = pd.DataFrame(data)
    df.to_csv("results/summary_table.csv", index=False)
    print("\n Saved: results/summary_table.csv")
    print("\n" + df.to_string(index=False))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


if __name__ == "__main__":
    print("="*70)
    print("COMPLETE ABLATION SUITE - 11 EXPERIMENTS")
    print("="*70)
    print(" Includes IID baseline")
    print(" 120 rounds per experiment")
    print(" All ablation studies")
    print("="*70)

    all_results = {}

    print("\n Experiment 1/11: Vanilla-ClientAdam (IID)")
    all_results["vanilla_iid"] = run_federated_experiment(
        method_name="VanillaClientAdam_IID",
        num_rounds=120,
        alpha="iid",
        lr=1e-3,
        seed=42
    )

    print("\n Experiment 2/11: Vanilla-ClientAdam (α=0.1, Extreme Non-IID)")
    all_results["vanilla_noniid"] = run_federated_experiment(
        method_name="VanillaClientAdam_NonIID",
        num_rounds=120,
        alpha=0.1,
        lr=1e-3,
        seed=42
    )

    print("\n Experiment 3/11: Q-LocalAdam (α=0.1, B=64)")
    all_results["qlocaladam"] = run_federated_experiment(
        method_name="Q-LocalAdam",
        num_rounds=120,
        alpha=0.1,
        lr=1e-3,
        block_size=64,
        seed=42
    )

    print("\n Experiment 4/11: Naive INT8 (Linear Quantization)")
    all_results["naive_int8"] = run_federated_experiment(
        method_name="NaiveINT8",
        num_rounds=120,
        alpha=0.1,
        lr=1e-3,
        block_size=64,
        seed=42
    )

    print("\n Experiment 5/11: Component Ablation (Momentum Only)")
    all_results["momentum_only"] = run_federated_experiment(
        method_name="QLocalAdam_MomentumOnly",
        num_rounds=120,
        alpha=0.1,
        lr=1e-3,
        block_size=64,
        seed=42
    )

    print("\n Experiment 6/11: Component Ablation (Variance Only)")
    all_results["variance_only"] = run_federated_experiment(
        method_name="QLocalAdam_VarianceOnly",
        num_rounds=120,
        alpha=0.1,
        lr=1e-3,
        block_size=64,
        seed=42
    )

    print("\n Experiment 7/11: Non-IID Robustness (α=0.5, Moderate)")
    all_results["noniid_alpha05"] = run_federated_experiment(
        method_name="Q-LocalAdam",
        num_rounds=120,
        alpha=0.5,
        lr=1e-3,
        block_size=64,
        seed=42
    )

    print("\n Experiment 8/11: Non-IID Robustness (α=1.0, Mild)")
    all_results["noniid_alpha10"] = run_federated_experiment(
        method_name="Q-LocalAdam",
        num_rounds=120,
        alpha=1.0,
        lr=1e-3,
        block_size=64,
        seed=42
    )

    print("\n Experiment 9/11: Block Size Ablation (B=32)")
    all_results["block_32"] = run_federated_experiment(
        method_name="Q-LocalAdam",
        num_rounds=120,
        alpha=0.1,
        lr=1e-3,
        block_size=32,
        seed=42
    )

    print("\n Experiment 10/11: Block Size Ablation (B=128)")
    all_results["block_128"] = run_federated_experiment(
        method_name="Q-LocalAdam",
        num_rounds=120,
        alpha=0.1,
        lr=1e-3,
        block_size=128,
        seed=42
    )

    print("\n Experiment 11/11: Learning Rate Sensitivity (lr=5e-4)")
    all_results["lr_5e4"] = run_federated_experiment(
        method_name="Q-LocalAdam",
        num_rounds=120,
        alpha=0.1,
        lr=5e-4,
        block_size=64,
        seed=42
    )

    print("\n" + "="*70)
    print("SAVING RESULTS AND GENERATING FIGURES")
    print("="*70)

    json_path = "results/complete_ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, cls=NumpyEncoder, indent=4)
    print(f" Saved: {json_path}")

    try:
        plot_all_results(all_results)
    except Exception as e:
        print(f" Error generating plots: {e}")

    try:
        create_summary_table(all_results)
    except Exception as e:
        print(f" Error generating summary table: {e}")

    print("\n" + "="*70)
    print(" ALL 11 EXPERIMENTS COMPLETE!")
    print("="*70)

    total_time = sum(r['total_time_hours'] for r in all_results.values())
    print(f"\nTotal runtime: {total_time:.2f} hours")

    print("\n KEY RESULTS:")
    print("-" * 70)

    iid_acc = all_results['vanilla_iid']['final_accuracy']
    noniid_acc = all_results['vanilla_noniid']['final_accuracy']
    q_acc = all_results['qlocaladam']['final_accuracy']
    q_mem = all_results['qlocaladam']['avg_total_opt_mb']
    vanilla_mem = all_results['vanilla_noniid']['avg_total_opt_mb']

    print(f"Vanilla-ClientAdam (IID):              {iid_acc:.2f}% accuracy")
    print(f"Vanilla-ClientAdam (Non-IID, α=0.1):   {noniid_acc:.2f}% accuracy, {vanilla_mem:.1f} MB")
    print(f"Q-LocalAdam (Non-IID):                 {q_acc:.2f}% accuracy, {q_mem:.1f} MB")

    acc_drop = noniid_acc - q_acc
    mem_reduction = vanilla_mem / q_mem if q_mem > 0 else 0

    print(f"\nAccuracy Drop:           {acc_drop:+.2f}%")
    print(f"Memory Reduction:        {mem_reduction:.2f}×")
    print("="*70)
