"""
src/evaluate.py
"""

import torch
import logging
import yaml
import json
import os
from utils import seed_everything, get_device, parse_args, load_model, count_parameters
from data import get_dataloaders, get_class_names

def evaluate_model(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device,
        logger: logging.Logger
) -> tuple[float, float, list[float], torch.Tensor]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    num_classes = 10
    per_class_correct = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    logger.info(f"Evaluating {model.__class__.__name__}")

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)
            preds = logits.argmax(dim=1)

            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for t, p in zip(labels, preds):
                confusion_matrix[t][p] += 1

            for c in range(num_classes):
                mask = (labels == c)
                per_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                per_class_total[c] += mask.sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    per_class_acc = (per_class_correct / per_class_total).tolist()

    return avg_loss, accuracy, per_class_acc, confusion_matrix
    

def main():
    args = parse_args(desc="Evaluate a CNN on CIFAR-10")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    _logger = logging.getLogger(__name__)

    experiment = args.experiment
    _logger.info(f"Evaluating Experiment: {experiment}")

    with open("params.yaml", 'r') as f:
        params: dict = yaml.safe_load(f)

    seed_everything(params["train"]["seed"])
    _logger.debug(f"Seeded RNGs with seed={params['train']['seed']}")

    device: torch.device = get_device(params["train"]["device"])
    _logger.info(f"Using device: {device}")

    model = load_model(experiment, device, _logger)

    _, test_loader = get_dataloaders(params)

    loss_fn = torch.nn.CrossEntropyLoss()

    avg_loss, accuracy, per_class_acc, confusion_matrix = evaluate_model(
        model, test_loader, loss_fn, device, _logger
    )

    class_names = get_class_names()

    for name, acc in zip(class_names, per_class_acc):
        _logger.info(f"{name}: {acc:.4f}")

    results = {
        "test_loss": avg_loss,
        "test_accuracy": accuracy,
        "per_class_accuracy": dict(zip(class_names, per_class_acc)),
        "num_parameters": count_parameters(model)
    }

    os.makedirs(f"metrics/{experiment}", exist_ok=True)
    with open(f"metrics/{experiment}/eval.json", "w") as f:
        json.dump(results, f, indent=2)

    torch.save(confusion_matrix, f"metrics/{experiment}/confusion_matrix.pt")

    _logger.info(f"Saved evaluation metrics to metrics/{experiment}")

if __name__ == "__main__":
    main()