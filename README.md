# What Does a CNN See? From LeNet-5 to AlexNet: Building and Interpreting CNNs on CIFAR-10

A deep learning final project for CS 615 at Drexel University. Three CNN architectures are implemented from scratch in PyTorch and trained on CIFAR-10, with a focus on interpretability: learned kernel visualization, intermediate feature maps, and gradient-based saliency maps.

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Reproducing Results

This project uses [DVC](https://dvc.org/) to manage the full pipeline. All stages — data preparation, training, evaluation, and visualization — are defined in `dvc.yaml` and configured via `params.yaml`.

### 1. Clone the repository

```bash
git clone <repo-url>
cd <repo-name>
```

### 2. Configure the DVC remote

Model checkpoints and plot outputs are stored in a DVC remote. Configure it before pulling:

```bash
dvc remote add -d <remote-name> <remote-url>
```

Or if you want to run the full pipeline locally from scratch (no remote needed):

```bash
dvc repro
```

This will execute all stages in order: `prepare → train_expN → eval_expN → visualize_expN`.

### 3. Pull cached outputs (optional)

If the DVC remote is configured, you can pull pre-computed outputs instead of re-running:

```bash
dvc pull
```

### 4. Run individual stages

To run a specific stage only:

```bash
dvc repro train_exp1
dvc repro eval_exp3
dvc repro visualize_exp2
```

### 5. Changing hyperparameters

All hyperparameters are in `params.yaml`. Edit the relevant section and re-run `dvc repro`, DVC will detect which stages are stale and re-run only those.

---

## Device Configuration

The default device in `params.yaml` is `mps` (Apple Silicon). To run on CUDA or CPU, update the `device` field:

```yaml
train:
  device: cuda   # or cpu
```

---

## Experiments

### Exp 1 — LeNet-5 (`exp1_lenet`)

A faithful adaptation of LeCun et al. (1998) for CIFAR-10. Uses 5×5 average pooling, tanh activations, and a three-layer fully connected head.

| Metric | Value |
|---|---|
| Parameters | 83,104 |
| Epochs | 30 |
| Best Val Accuracy | 54.13% |
| Test Accuracy | **54.13%** |
| Test Loss | 1.300 |

Per-class test accuracy:

| Class | Accuracy |
|---|---|
| airplane | 53.1% |
| automobile | 64.7% |
| bird | 37.1% |
| cat | 42.9% |
| deer | 40.1% |
| dog | 41.2% |
| frog | 62.6% |
| horse | 63.6% |
| ship | 72.3% |
| truck | 63.7% |

---

### Exp 2 — HybridNet (`exp2_hybrid`)

A modernized hybrid that retains the LeNet topology but replaces average pooling with max pooling, adds ReLU activations, increases filter counts, and adds dropout.

| Metric | Value |
|---|---|
| Parameters | 341,214 |
| Epochs | 30 |
| Best Val Accuracy | 76.39% |
| Test Accuracy | **76.39%** |
| Test Loss | 0.704 |

Per-class test accuracy:

| Class | Accuracy |
|---|---|
| airplane | 82.5% |
| automobile | 85.1% |
| bird | 61.4% |
| cat | 62.2% |
| deer | 73.3% |
| dog | 60.1% |
| frog | 86.2% |
| horse | 80.0% |
| ship | 85.8% |
| truck | 87.3% |

---

### Exp 3 — AlexNet-style (`exp3_alexnet`)

A CIFAR-10-adapted AlexNet with five convolutional layers, batch normalization, and two fully connected hidden layers with dropout.

| Metric | Value |
|---|---|
| Parameters | 5,420,490 |
| Epochs | 50 |
| Best Val Accuracy | 85.87% |
| Test Accuracy | **85.87%** |
| Test Loss | 0.415 |

Per-class test accuracy:

| Class | Accuracy |
|---|---|
| airplane | 87.0% |
| automobile | 94.1% |
| bird | 73.2% |
| cat | 75.3% |
| deer | 85.4% |
| dog | 76.5% |
| frog | 91.5% |
| horse | 91.6% |
| ship | 92.7% |
| truck | 91.4% |

---

## Visualization Outputs

Running `dvc repro visualize_expN` (or `dvc repro` for all) produces the following plots under `plots/<exp_name>/`:

| File | Description |
|---|---|
| `kernels_conv1.png` | RGB composite visualization of all learned conv1 kernels |
| `feature_maps_conv1.png` | Conv1 activations for one sample image per class (viridis colormap) |
| `feature_maps_conv2.png` | Conv2 activations — same layout |
| `feature_maps_convN.png` | Deeper layer activations (exp3 only: conv3, conv4) |
| `saliency_maps.png` | Input image, SmoothGrad saliency, and jet overlay for one image per class |

Saliency maps use SmoothGrad (N=20, σ=0.1) by default. This is controlled in `params.yaml`:

```yaml
visualize:
  num_sample_images: 8
  saliency_smooth: true
  saliency_n_smooth: 20
  saliency_noise_std: 0.1
  output_dir: plots
```

Set `saliency_smooth: false` to use vanilla gradients instead.

---

## Summary of Results

| Experiment | Params | Test Acc | Test Loss |
|---|---|---|---|
| LeNet-5 (Exp 1) | 83K | 54.1% | 1.300 |
| HybridNet (Exp 2) | 341K | 76.4% | 0.704 |
| AlexNet-style (Exp 3) | 5.4M | 85.9% | 0.415 |

---

## References

- LeCun et al. (1998). *Gradient-Based Learning Applied to Document Recognition.*
- Krizhevsky, Sutskever & Hinton (2012). *ImageNet Classification with Deep Convolutional Neural Networks.*
- Zeiler & Fergus (2014). *Visualizing and Understanding Convolutional Networks.*
- Krizhevsky (2009). *Learning Multiple Layers of Features from Tiny Images.* (CIFAR-10 dataset)