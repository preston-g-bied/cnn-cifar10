"""
src/train.py
"""

import json
import yaml
import torch
import logging
from utils import seed_everything, get_device, parse_args, get_model
from data import get_dataloaders
import os

def train_one_epoch(model: torch.nn.Module,
                    loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: torch.nn.Module,
                    device: torch.device
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    args = parse_args(desc="Train a CNN on CIFAR-10")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    _logger = logging.getLogger(__name__)

    experiment = args.experiment
    _logger.info(f"Running Experiment: {experiment}")

    with open("params.yaml", 'r') as f:
        params: dict = yaml.safe_load(f)

    exp_params: dict = params[experiment]

    seed_everything(params["train"]["seed"])
    _logger.debug(f"Seeded RNGs with seed={params['train']['seed']}")

    device: torch.device = get_device(params["train"]["device"])
    _logger.info(f"Using device: {device}")

    model = get_model(experiment, _logger)

    model = model.to(device)
    _logger.debug(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    train_loader, test_loader = get_dataloaders(params)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    optimizer = optimizers[params["train"]["optimizer"]](
        model.parameters(),
        lr=exp_params["lr"],
        weight_decay=params["train"]["weight_decay"]
    )
    epochs: int = exp_params["epochs"]
    best_acc: float = 0.0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, test_loader, loss_fn, device)

        _logger.info(
            f"Epoch {epoch:>3}/{epochs} | "
            f"train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )
        _logger.debug(f"Epoch {epoch} raw stats: train_loss={train_loss}, val_acc={val_acc}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(f"models/{experiment}", exist_ok=True)
            torch.save(model.state_dict(), f"models/{experiment}/best.pt")
            _logger.debug(f"New best model saved (val_acc={best_acc:.4f})")

    _logger.info(f"Training complete. Best val accuracy: {best_acc:.4f}")

    os.makedirs(f"metrics/{experiment}", exist_ok=True)
    with open(f"metrics/{experiment}/results.json", "w") as f:
        json.dump({"best_val_acc": best_acc, "history": history}, f, indent=2)
    _logger.info(f"Metrics saved to metrics/{experiment}/results.json")

if __name__ == "__main__":
    main()