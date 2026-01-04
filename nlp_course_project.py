#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NLP Course Project - Multi-Task Adapter Learning
Main entry point for training, testing, and evaluation
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.data.dataloader import MultiTaskDataLoader
from src.models.base_adapter import BaseAdapterModel
from src.models.improved_adapter import ImprovedAdapterModel
from src.training.trainer import MultiTaskTrainer
from src.evaluation.evaluator import MultiTaskEvaluator
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Multi-Task Adapter Learning Project"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Configuration file path"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="train", 
        choices=["train", "test", "reproduce", "improve"],
        help="Running mode"
    )
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="improved",
        choices=["base", "improved"],
        help="Model type"
    )
    parser.add_argument(
        "--tasks", 
        type=str, 
        nargs="+", 
        default=["sst2", "mrpc", "qnli"],
        help="List of tasks"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device (cuda:0, cpu)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/",
        help="Output directory"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Debug mode"
    )
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main function"""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)

    # Set random seed for reproducibility
    set_seed(config['experiment']['random_seed'])

    # Setup device
    device = args.device or config['experiment']['device']
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device = "cpu"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(output_dir / "run.log")
    logger.info(f"Starting run: mode={args.mode}, device={device}")
    logger.info(f"Configuration: {config}")

    # Load data
    logger.info("Loading datasets...")
    dataloader = MultiTaskDataLoader(
        tasks=args.tasks,
        batch_size=config['training']['batch_size'],
        max_length=128,
        debug=args.debug
    )
    
    train_loaders, val_loaders, test_loaders = dataloader.get_dataloaders()

    # Initialize model
    logger.info(f"Initializing {args.model_type} model...")
    if args.model_type == "base":
        model = BaseAdapterModel(
            model_name=config['model']['base_model'],
            tasks=args.tasks,
            adapter_size=config['model']['adapter_size']
        )
    else:
        model = ImprovedAdapterModel(
            model_name=config['model']['base_model'],
            tasks=args.tasks,
            adapter_size=config['model']['adapter_size'],
            use_adaptive_weights=config['improvements']['adaptive_weights'],
            use_sparse_gating=config['improvements']['sparse_gating'],
            weight_reg=config['improvements']['weight_regularization']
        )
    
    model = model.to(device)

    # Log model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} "
                f"({trainable_params/total_params*100:.2f}%)")

    # Training phase
    if args.mode in ["train", "reproduce", "improve"]:
        trainer = MultiTaskTrainer(
            model=model,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            tasks=args.tasks,
            device=device,
            config=config
        )

        logger.info("Starting training...")
        history = trainer.train()

        # Save model
        model_path = output_dir / f"{args.model_type}_model.pt"
        trainer.save_model(model_path)
        logger.info(f"Model saved to: {model_path}")

        # Plot training history
        trainer.plot_training_history(output_dir)

    # Testing/evaluation phase
    if args.mode in ["test", "reproduce", "improve"]:
        evaluator = MultiTaskEvaluator(
            model=model,
            test_loaders=test_loaders,
            tasks=args.tasks,
            device=device
        )
        
        logger.info("Starting evaluation...")
        test_results = evaluator.evaluate()

        # Save results
        results_path = output_dir / f"{args.model_type}_results.json"
        evaluator.save_results(test_results, results_path)
        logger.info(f"Results saved to: {results_path}")

        # Print results
        evaluator.print_results(test_results)

        # Visualize results
        evaluator.visualize_results(test_results, output_dir)
    
    logger.info("Run completed!")


if __name__ == "__main__":
    main()