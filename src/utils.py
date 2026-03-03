"""
src/utils.py
"""

import random
import numpy as np
import torch
import argparse
import logging

from models.lenet import LeNet
from models.hybrid import HybridNet
from models.alexnet import AlexNet

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def get_device(device_str: str) -> torch.device:
    if device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def parse_args(desc: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--experiment",
        type=str,
        choices=["exp1_lenet", "exp2_hybrid", "exp3_alexnet"],
        required=True,
        help="Which experiment to run"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging (default: INFO)"
    )
    
    return parser.parse_args()

def get_model(experiment: str, logger: logging.Logger):
    if experiment == "exp1_lenet":
        model = LeNet()
        logger.info('Using LeNet model')
    elif experiment == "exp2_hybrid":
        model = HybridNet()
        logger.info('Using HybridNet model')
    elif experiment == "exp3_alexnet":
        model = AlexNet()
        logger.info('Using AlexNet model')
    else:
        logger.error("No model found for this experiment, using LeNet")
        model = LeNet()
    return model

def load_model(exp_name: str, device: torch.device, logger: logging.Logger) -> torch.nn.Module:
    model = get_model(exp_name, logger)
    model_path = f"models/{exp_name}/best.pt"
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model = model.to(device).eval()
    logger.info(f"Loaded best checkpoint from {model_path}")
    return model

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)