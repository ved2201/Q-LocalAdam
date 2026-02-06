# Q-LocalAdam: Quantized Federated Learning Optimizer

Comprehensive experimental suite for evaluating Q-LocalAdam, a novel INT8-quantized federated learning optimizer that achieves 4× memory reduction with minimal accuracy loss.

## Overview

This repository contains complete ablation studies comparing Q-LocalAdam against Vanilla-ClientAdam (FP32 baseline) on CIFAR-10 and CIFAR-100 datasets under various Non-IID data distributions.

## Key Features

- **Memory Efficient**: 4× reduction in optimizer state memory via INT8 quantization
- **Novel Log-Space Quantization**: Specialized quantizer for variance terms
- **Robust to Non-IID**: Maintains performance under extreme data heterogeneity (α=0.1)
- **Comprehensive Evaluation**: 11 experiments per dataset covering all ablations
- **Resumable Training**: Automatic checkpointing for long-running experiments

## Requirements

```bash
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
```

Install dependencies:
```bash
pip install torch torchvision numpy matplotlib pandas
```

## Hardware Requirements

- **Recommended**: NVIDIA RTX 5080 or equivalent (16GB+ VRAM)
- **Minimum**: NVIDIA GPU with 8GB VRAM
- **CPU**: Multi-core processor (experiments use 2 workers per data loader)
- **RAM**: 16GB+ recommended
- **Storage**: ~5GB for datasets and results

## Quick Start

### CIFAR-10 Experiments

```bash
python qlocaladam_complete_experiments.py
```

Expected runtime: ~6-7 hours on RTX 5080

### CIFAR-100 Experiments

```bash
python run_cifar100_qlocaladam_experiments.py
```

Expected runtime: ~8-10 hours on RTX 5080

## Experiment Suite

Both scripts run 11 comprehensive experiments:

1. **Vanilla-ClientAdam (IID)** - FP32 baseline with uniform data distribution
2. **Vanilla-ClientAdam (Non-IID, α=0.1)** - FP32 under extreme heterogeneity
3. **Q-LocalAdam (Main)** - Proposed method with INT8 quantization (B=64)
4. **Naive INT8** - Linear quantization baseline (failure case)
5. **Momentum Only** - Ablation with only momentum quantized
6. **Variance Only** - Ablation with only variance quantized
7. **Non-IID Robustness (α=0.5)** - Moderate heterogeneity
8. **Non-IID Robustness (α=1.0)** - Mild heterogeneity
9. **Block Size B=32** - Smaller quantization blocks
10. **Block Size B=128** - Larger quantization blocks
11. **Learning Rate Sensitivity (lr=5e-4)** - Lower learning rate

## Algorithm Details

### Vanilla-ClientAdam (FP32 Baseline)

Standard Adam optimizer with full precision (FP32) momentum and variance states:

- Momentum: FP32 storage
- Variance: FP32 storage
- **Total memory**: ~2× model parameters

### Q-LocalAdam (Proposed Method)

INT8-quantized optimizer with specialized quantizers:

- **Momentum**: Linear block-wise INT8 quantization
- **Variance**: Log-space block-wise INT8 quantization
- **Total memory**: ~0.5× model parameters (4× reduction)

### Key Innovation: Log-Space Quantization

Variance terms in Adam have exponential distribution. Linear quantization fails to capture small values accurately. Our log-space quantization:

1. Transform to log-domain: `log(v + ε)`
2. Quantize linearly in log-space
3. Transform back: `exp(q) - ε`

This preserves precision for small variance values critical for convergence.

## Model Architectures

### CIFAR-10
- **MidResNet18**: 48 → 96 → 192 → 384 channels
- **Parameters**: ~6.3M
- **Input size**: 224×224 (resized from 32×32)
- **Classes**: 10

### CIFAR-100
- **MidResNet18_CIFAR100**: 64 → 128 → 256 → 512 channels
- **Parameters**: ~11M
- **Input size**: 224×224 (resized from 32×32)
- **Classes**: 100

## Non-IID Data Partitioning

Data heterogeneity simulated via Dirichlet distribution with concentration parameter α:

- **α = ∞ (IID)**: Uniform distribution across clients
- **α = 0.1**: Extreme Non-IID (each client has 1-2 dominant classes)
- **α = 0.5**: Moderate Non-IID
- **α = 1.0**: Mild Non-IID

## Federated Learning Setup

- **Clients**: 5
- **Local epochs**: 2 per round
- **Batch size**: 64
- **Communication rounds**: 120
- **Aggregation**: FedAvg (weighted averaging)
- **Learning rate**: 1e-3 (default)
- **Optimizer betas**: (0.9, 0.999)

## Output Structure

```
results/
├── complete_ablation_results.json      # CIFAR-10 results
├── summary_table.csv                   # CIFAR-10 summary
└── cifar100/
    ├── complete_ablation_results.json  # CIFAR-100 results
    ├── summary_table.csv               # CIFAR-100 summary
    └── checkpoint.json                 # Resume checkpoint

figures/
├── 1_iid_vs_noniid.png                 # CIFAR-10 IID comparison
├── 2_main_comparison.png               # CIFAR-10 main results
├── 3_naive_comparison.png              # CIFAR-10 quantization comparison
└── cifar100/
    ├── 1_iid_vs_noniid.png             # CIFAR-100 IID comparison
    ├── 2_main_comparison.png           # CIFAR-100 main results
    └── 3_naive_comparison.png          # CIFAR-100 quantization comparison
```

All figures also saved as PDF for publication quality.

## Resuming Interrupted Experiments

If experiments are interrupted (crash, power loss, etc.), simply rerun the script:

```bash
python run_cifar100_qlocaladam_experiments.py
```

The script automatically:
- Detects existing checkpoint
- Loads completed experiments
- Resumes from next experiment
- Preserves all previous results

Manual checkpoint management:
```bash
# Remove checkpoint to restart from scratch
rm results/cifar100/checkpoint.json
```

## Performance Optimization

### Enabled by Default:
- TensorFlow 32 (TF32) acceleration on Ampere+ GPUs
- cuDNN benchmarking for optimal convolution algorithms
- Automatic mixed precision ready (though not used)

### GPU Memory Management:
- Automatic `torch.cuda.empty_cache()` after each round
- Optimizer states stored on CPU (quantized)
- Only active model on GPU during training

### Multi-Processing:
- DataLoader workers: 2 per client
- Adjust `num_workers` if CPU-bound

## Expected Results (CIFAR-10)

| Method | Final Acc | Memory (MB) | Time (h) |
|--------|-----------|-------------|----------|
| Vanilla-ClientAdam (IID) | ~75% | 50 MB | 0.5 |
| Vanilla-ClientAdam (Non-IID) | ~68% | 50 MB | 0.5 |
| Q-LocalAdam | ~66% | 12 MB | 0.5 |

**Memory Reduction**: 4× (50 MB → 12 MB)  
**Accuracy Drop**: <2% under extreme Non-IID

## Expected Results (CIFAR-100)

| Method | Final Acc | Memory (MB) | Time (h) |
|--------|-----------|-------------|----------|
| Vanilla-ClientAdam (IID) | ~50% | 88 MB | 0.8 |
| Vanilla-ClientAdam (Non-IID) | ~42% | 88 MB | 0.8 |
| Q-LocalAdam | ~40% | 22 MB | 0.8 |

**Memory Reduction**: 4× (88 MB → 22 MB)  
**Accuracy Drop**: <2% under extreme Non-IID

## Customization

### Modify Hyperparameters

Edit experiment configurations in `__main__`:

```python
{
    "key": "custom_experiment",
    "desc": "Custom Configuration",
    "kwargs": dict(
        method_name="Q-LocalAdam",
        num_rounds=200,           # More rounds
        alpha=0.3,                # Different Non-IID level
        lr=5e-4,                  # Lower learning rate
        block_size=128,           # Larger blocks
        seed=42,
    ),
}
```

### Add New Experiments

Append to `experiments` list:

```python
experiments.append({
    "key": "my_experiment",
    "desc": "My Custom Experiment",
    "kwargs": dict(...)
})
```

### Change Model Architecture

Modify `MidResNet18` or `MidResNet18_CIFAR100` classes:

```python
class MyCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture
```

Update model instantiation in `run_federated_experiment()`.

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `batch_size=32`
- Reduce model size: Modify channel dimensions
- Reduce number of clients: `num_clients=3`

### Slow Training
- Check GPU utilization: `nvidia-smi`
- Increase `num_workers` if CPU-bound
- Verify TF32 is enabled (logged at startup)

### Numerical Instability
- Increase quantization block size: `block_size=128`
- Reduce learning rate: `lr=5e-4`
- Check gradient norms (add logging)

### Checkpoint Corruption
- Remove and restart: `rm results/cifar100/checkpoint.json`
- Verify disk space: Checkpoints can be large (>100MB)

## Code Structure

```
qlocaladam_complete_experiments.py          # CIFAR-10 main script
run_cifar100_qlocaladam_experiments.py      # CIFAR-100 main script

Key Components:
├── Model Classes
│   ├── BasicBlockCustom                    # ResNet building block
│   ├── MidResNet18                         # CIFAR-10 model
│   └── MidResNet18_CIFAR100                # CIFAR-100 model
├── Data Partitioning
│   ├── dirichlet_split()                   # Non-IID partitioning
│   └── iid_split()                         # IID partitioning
├── Quantizers
│   ├── BlockwiseQuantizer                  # Linear INT8 quantization
│   └── LogQuantizer                        # Log-space INT8 quantization
├── Optimizers
│   ├── VanillaClientAdam                   # FP32 baseline
│   ├── QLocalAdam                          # Main quantized optimizer
│   ├── NaiveINT8                           # Naive quantization baseline
│   ├── QLocalAdam_MomentumOnly             # Ablation: momentum only
│   └── QLocalAdam_VarianceOnly             # Ablation: variance only
├── Training
│   ├── train_one_epoch()                   # Local training loop
│   ├── evaluate()                          # Test evaluation
│   ├── federated_averaging()               # FedAvg aggregation
│   └── run_federated_experiment()          # Main experiment runner
├── Utilities
│   ├── measure_optimizer_memory()          # Memory profiling
│   ├── measure_model_memory()              # Model size
│   ├── plot_all_results()                  # Figure generation
│   ├── create_summary_table()              # CSV export
│   ├── save_checkpoint()                   # Checkpoint saving
│   └── load_checkpoint()                   # Checkpoint loading
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{qlocaladam2026,
  title={Q-LocalAdam: Memory-Efficient Federated Learning via INT8 Quantized Optimizer States},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests if applicable
4. Submit a pull request

## Contact

For questions or issues:
- Open a GitHub issue
- Email: your.email@example.com

## Acknowledgments

- CIFAR-10/100 datasets from Alex Krizhevsky
- PyTorch team for the deep learning framework
- Federated Learning community for inspiration

## Changelog

### v1.0.0 (February 2026)
- Initial release
- CIFAR-10 and CIFAR-100 experiments
- 11 ablation studies per dataset
- Checkpointing support
- Publication-quality figures
