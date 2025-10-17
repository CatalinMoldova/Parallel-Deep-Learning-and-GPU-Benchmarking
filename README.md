# Parallel-Deep-Learning-and-GPU-Benchmarking




***

# Homework 2 - Parallel Deep Learning and GPU Benchmarking

This project investigates **multi-GPU training performance, compute-communication tradeoffs, and bandwidth utilization** for deep learning using PyTorch. The main model used is **ResNet-18** trained on the **CIFAR-10 dataset**, and the analysis explores **scaling efficiency** under different batch sizes and GPU configurations.

***

## Overview

The notebook performs a comprehensive benchmarking study on **multi-GPU data parallelism** using PyTorch's `DataParallel` wrapper. It measures:
- The training performance on **single vs. multiple GPUs**.
- Speedup and scaling behavior across 1, 2, and 4 GPUs.
- The **compute vs. communication time breakdown**.
- **Bandwidth utilization** for gradient synchronization using the Ring AllReduce model.

***

## Project Structure

| Section | Description |
|----------|--------------|
| Data Preparation | Loads CIFAR-10 and applies standard training/testing transformations. |
| Model Setup | Builds a ResNet-18 model and wraps it for multi-GPU training. |
| Training Functions | Defines training and testing loops with loss, accuracy, and scheduler. |
| Benchmarking Functions | Implements timing experiments for varying batch sizes and GPU counts. |
| Results Analysis | Computes scaling speedups, communication overhead, and bandwidth utilization. |
| Visualization | Plots training/test loss and accuracy over epochs. |

***

## Requirements

To run this project, install dependencies using:

```bash
pip install torch torchvision matplotlib
```

Tested versions:
- PyTorch 2.8.0+cu128  
- CUDA available with 4× NVIDIA L4 GPUs  
- Python 3.13.5  

***

## Dataset

**CIFAR-10**, automatically downloaded via `torchvision.datasets.CIFAR10`.  
Transforms include:
- Random cropping and horizontal flipping for data augmentation.
- Normalization using dataset-specific means and standard deviations.

***

## Model Details

- **Architecture**: ResNet-18 implemented via `pytorch-cifar` repository.
- **Loss Function**: CrossEntropyLoss.
- **Optimizer**: SGD with momentum (0.9) and weight decay (5e−4).
- **Scheduler**: MultiStepLR with milestones at epochs 150 and 250.

Training runs for **50 epochs**, achieving a **best test accuracy of 87.55%**.

***

## Benchmarking Experiments

### 1. Single GPU Training
Measured training time for batch sizes 32, 128, and 512.

### 2. Multi-GPU Scaling
Evaluated strong scaling performance for 1, 2, and 4 GPUs:
- Strong scaling keeps per-GPU batch size fixed while increasing GPUs.
- Achieved near-doubling in speedup at `batch_size=512` on 4 GPUs (≈1.96×).

### 3. Compute vs Communication Breakdown
- Compute time corresponds to single-GPU time per batch.
- Communication time = total multi-GPU time − compute time.
- Analyzed how synchronization dominates as GPUs increase.

### 4. Bandwidth Utilization
Based on the **Ring AllReduce** model:

$$
T_{\text{comm}} = \frac{2 (N - 1)}{N} \frac{S}{B}
$$

where  
N = number of GPUs,  
S = model size in bytes,  
B = link bandwidth (e.g., 300 GB/s for NVLink).

Effective bandwidth and utilization were derived from measured times.

***

## Key Results

| Metric | Observation |
|--------|--------------|
| Best test accuracy | 87.55% |
| Batch size 32 | Scaling efficiency low due to synchronization overhead |
| Batch size 128 | Stable scaling, near-linear behavior on 2 GPUs |
| Batch size 512 | Best multi-GPU speedup (up to 1.96× with 4 GPUs) |
| Model size | 11,173,962 parameters ≈ 0.0416 GB |
| Bandwidth utilization | Low (<0.01%) due to small model size relative to link capacity |

***

## Visualization

The training process produces plots of:
- **Loss curves** (train/test)
- **Accuracy curves** (train/test)

Saved as `trainingprogress.png`.

***

## How to Run

1. Clone or download this repository.
2. Run the notebook on a system with multiple CUDA-enabled GPUs:
   ```bash
   jupyter notebook Catalin_Botezat_HW2_P3.ipynb
   ```
3. Follow cell execution order to:
   - Train the model.
   - Benchmark performance.
   - Generate and analyze results plots.

***

## Author

**Catalin Botezat**  
NYU Abu Dhabi – Mathematics and Computer Science  
Assignment: Homework 2, Part 3 – Distributed Deep Learning Performance  

***

## License

This repository is provided for academic purposes as part of coursework on distributed deep learning systems.

---

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/38276670/b3e74a7a-f74d-443f-a9cb-37c431d7932d/Catalin_Botezat_HW2_P3.ipynb)
