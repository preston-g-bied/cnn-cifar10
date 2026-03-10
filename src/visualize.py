"""
src/visualize.py
"""

import logging
import math
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from typing import cast

from data import get_dataloaders, get_class_names
from layers import ConvolutionalLayer
from utils import get_device, load_model, parse_args, seed_everything


# --- CONSTANTS ---
# CIFAR-10 normalization constants (matches data.py)

_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
_STD  = torch.tensor([0.2470, 0.2435, 0.2616])

MAX_FEATURE_CHANNELS = 16


# --- HELPER FUNCTIONS ---

def _outdir(exp_name: str) -> str:
    path = os.path.join("plots", exp_name)
    os.makedirs(path, exist_ok=True)
    return path

def _denormalize(img: torch.Tensor) -> np.ndarray:
    """
    converts normalized (C, H, W) tensor to (H, W, C) float32 np array
    """
    img = img.cpu().float()
    img = img * _STD[:, None, None] + _MEAN[:, None, None]
    return img.permute(1, 2, 0).clamp(0.0, 1.0).numpy()

def _get_conv_layers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    """
    returns (name, module) for every conv layer in the model
    """
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, ConvolutionalLayer)
    ]

def get_one_per_class(params: dict) -> tuple[torch.Tensor, torch.Tensor]:
    _, test_loader = get_dataloaders(params)
    num_classes = len(get_class_names())

    found_images: list[torch.Tensor | None] = [None] * num_classes
    found_labels: list[torch.Tensor | None] = [None] * num_classes
    remaining = set(range(num_classes))

    for images, labels in test_loader:
        for img, lbl in zip(images, labels):
            c = lbl.item()
            if c in remaining:
                found_images[c] = img
                found_labels[c] = lbl
                remaining.discard(c)
        if not remaining:
            break

    if remaining:
        raise ValueError(f"Could not find any samples for classes: {remaining}")
    
    images_out: list[torch.Tensor] = [t for t in found_images if t is not None]
    labels_out: list[torch.Tensor] = [t for t in found_labels if t is not None]

    return torch.stack(images_out), torch.stack(labels_out)


# --- KERNEL VISUALIZATION ---

def visualize_kernels(model: torch.nn.Module, exp_name: str, logger: logging.Logger) -> None:
    """
    Render every conv1 kernel as an RGB composite
    Normalize (3, K, K) kernel to [0, 1], display as R/G/B
    Matches style from Zeiler & Fergus
    """
    logger.info(f"[{exp_name}] Visualizing conv1 kernels...")

    conv1 = cast(ConvolutionalLayer, model.conv1)
    kernels = conv1.getKernel().detach().cpu()
    P = kernels.shape[0]

    rgb_kernels = []
    for i in range(P):
        k = kernels[i].float()
        k_min = k.min()
        k_max = k.max()
        k = (k - k_min) / (k_max - k_min * 1e-8)
        rgb_kernels.append(k.permute(1, 2, 0).numpy())

    ncols = min(P, 8)
    nrows = math.ceil(P / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    model_label = model.__class__.__name__
    fig.suptitle(
        f"{model_label} - Conv1 Learned Kernels (RGB composite, {P} kernels)",
        fontsize=10,
        y=1.01
    )

    axes_flat = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes_flat):
        if i < P:
            ax.imshow(rgb_kernels[i], interpolation="nearest")
            ax.set_title(f"K{i}", fontsize=6, pad=2)
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(_outdir(exp_name), "kernels_conv1.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[{exp_name}] -> {path}")


# --- FEATURE MAPS ---

def visualize_feature_maps(
        model: torch.nn.Module,
        images: torch.Tensor,   # (10, 3, 32, 32)
        labels: torch.Tensor,   # (10,)
        exp_name: str,
        class_names: list[str],
        device: torch.device,
        logger: logging.Logger
) -> None:
    """
    Captures post-activation feature maps from every conv layer via forward hooks
    Saves one figure per layer

    Figure layout:
        rows: CIFAR-10 classes
        cols: feature-map channels
    """
    logger.info(f"[{exp_name}] Visualizing feature maps...")

    conv_layers = _get_conv_layers(model)
    activations: dict[str, torch.Tensor] = {}

    hooks = []
    for name, mod in conv_layers:
        def _make_hook(n: str):
            def _hook(_module, _input, output):
                activations[n] = output.detach().cpu()
            return _hook
        hooks.append(mod.register_forward_hook(_make_hook(name)))

    model.eval()
    with torch.no_grad():
        model(images.to(device))

    for h in hooks:
        h.remove()

    for layer_name, acts in activations.items():
        # acts: (10, C, H_out, W_out)
        _, C, H_out, W_out = acts.shape
        n_show = min(C, MAX_FEATURE_CHANNELS)

        fig, axes = plt.subplots(
            10, n_show,
            figsize=(n_show * 1.0, 10 * 1.0),
            squeeze=False
        )
        model_label = model.__class__.__name__
        fig.suptitle(
            f"{model_label} - Feature Maps: {layer_name}  "
            f"({H_out}x{W_out}, showing {n_show}/{C} channels)",
            fontsize=9,
            y=1.0005
        )

        for col in range(n_show):
            axes[0][col].set_title(f"ch{col}", fontsize=6, pad=2)

        for row in range(10):
            act = acts[row]
            class_name = class_names[int(labels[row].item())]

            axes[row][0].set_ylabel(
                class_name, fontsize=7, rotation=0, labelpad=32, va="center"
            )

            for col in range(n_show):
                ax = axes[row][col]
                fmap = act[col].numpy()
                ax.imshow(fmap, cmap="viridis", aspect="auto", interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        path = os.path.join(_outdir(exp_name), f"feature_maps_{layer_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"[{exp_name}] -> {path} ({C} channels, {H_out}x{W_out})")
    
    logger.info(f"[{exp_name}] Feature maps complete - {len(activations)} layer(s)")


# --- SALIENCY MAPS ---

def _compute_saliency(
        model: torch.nn.Module,
        image: torch.Tensor,    # (3, 32, 32)
        label_idx: int,
        device: torch.device,
        smooth: bool,
        n_smooth: int,
        noise_std: float
) -> np.ndarray:
    model_cpu = model.cpu()
    model_cpu.eval()

    def _single_pass(inp_cpu: torch.Tensor) -> torch.Tensor:
        inp = inp_cpu.unsqueeze(0).clone().requires_grad_(True)
        logits = model_cpu(inp)
        model_cpu.zero_grad()
        logits[0, label_idx].backward()
        grad = inp.grad
        assert grad is not None
        return grad[0].abs().mean(dim=0)
    
    if smooth:
        grads = [_single_pass(image.cpu() + torch.randn_like(image) * noise_std)
                 for _ in range(n_smooth)]
        saliency = torch.stack(grads).mean(0).numpy()
    else:
        saliency = _single_pass(image.cpu()).numpy()

    model.to(device)

    return saliency

def visualize_saliency(
        model: torch.nn.Module,
        images: torch.Tensor,   # (10, 3, 32, 32)
        labels: torch.Tensor,   # (10,)
        exp_name: str,
        class_names: list[str],
        params: dict,
        device: torch.device,
        logger: logging.Logger
) -> None:
    logger.info(f"[{exp_name}] Computing saliency maps...")

    vp = params.get("visualize", {})
    smooth = vp.get("saliency_smooth", True)
    n_smooth = vp.get("saliency_n_smooth", 20)
    noise_std = vp.get("saliency_noise_std", 0.1)

    smooth_tag = f" (SmoothGrad, N={n_smooth})" if smooth else ""
    model_label = model.__class__.__name__

    fig, axes = plt.subplots(10, 3, figsize=(6.5, 22))
    fig.suptitle(
        f"{model_label} - Gradient Saliency Maps{smooth_tag}",
        fontsize=11,
        y=1.002
    )

    col_titles = ["Input Image", "Saliency", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=9, pad=4)

    for row in range(10):
        img = images[row]   # (3, 32, 32)
        label_idx = int(labels[row].item())
        class_name = class_names[label_idx]

        logger.debug(f"[{exp_name}] saliency for class '{class_name}'...")

        sal = _compute_saliency(
            model, img, label_idx, device, smooth, n_smooth, noise_std
        )

        sal_norm = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

        orig = _denormalize(img)
        jet_rgb = plt.get_cmap("jet")(sal_norm)[:, :, :3]
        overlay = (0.5 * orig + 0.5 * jet_rgb).clip(0, 1)

        ax_orig, ax_sal, ax_ov = axes[row]

        ax_orig.imshow(orig, interpolation="nearest")
        ax_sal.imshow(sal_norm, cmap="hot", vmin=0, vmax=1, interpolation="nearest")
        ax_ov.imshow(overlay, interpolation="nearest")

        ax_orig.set_ylabel(class_name, fontsize=8, rotation=0, labelpad=36, va="center")

        for ax in (ax_orig, ax_sal, ax_ov):
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(_outdir(exp_name), "saliency_maps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[{exp_name}] -> {path}")


# --- ENTRY POINT ---

def main() -> None:
    args = parse_args(desc="Generate CNN interpretability visualizations")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    exp_name = args.experiment
    logger.info(f"=== Visualize: {exp_name} ===")

    with open("params.yaml", "r") as f:
        params: dict = yaml.safe_load(f)

    seed_everything(params["train"]["seed"])
    device = get_device(params["train"]["device"])
    logger.info(f"Using device: {device}")

    model = load_model(exp_name, device, logger)

    class_names = get_class_names()
    images, labels = get_one_per_class(params)
    logger.info(
        f"Loaded one image per class: "
        f"{[class_names[int(lab.item())] for lab in labels]}"
    )

    visualize_kernels(model, exp_name, logger)

    visualize_feature_maps(model, images, labels, exp_name, class_names, device, logger)

    visualize_saliency(model, images, labels, exp_name, class_names, params, device, logger)

    logger.info(f"=== Done - outputs in plots/{exp_name}/ ===")

if __name__ == "__main__":
    main()