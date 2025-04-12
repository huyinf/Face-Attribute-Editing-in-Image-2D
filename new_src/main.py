"""
Main script to run the ELEGANT model for attribute manipulation.
"""

import os
import argparse
import torch
from pathlib import Path

from elegant.config.config import Config
from elegant.models.elegant import ELEGANT

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ELEGANT: Entangled Latent Encoding for Attribute Manipulation')
    
    # Dataset arguments
    parser.add_argument('--data_dir', type=str, default='data/celeba',
                        help='Path to CelebA dataset directory')
    parser.add_argument('--attributes', type=str, nargs='+', default=['Smiling', 'Male'],
                        help='List of attributes to manipulate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--max_iter', type=int, default=200000,
                        help='Maximum number of training iterations')
    parser.add_argument('--restore', type=int, default=None,
                        help='Checkpoint iteration to restore from')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda/cpu)')
    
    return parser.parse_args()

def main():
    """Main function to run the ELEGANT model."""
    # Parse arguments
    args = parse_args()
    
    # Initialize configuration
    config = Config()
    
    # Update config with command line arguments
    config.data_dir = Path(args.data_dir)
    config.model_params['nchw'][0] = args.batch_size
    config.model_params['max_iter'] = args.max_iter
    config.training_params['device'] = args.device
    
    # Create model
    model = ELEGANT(args, config)
    
    # Start training
    print(f"Starting training with attributes: {args.attributes}")
    print(f"Using device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Maximum iterations: {args.max_iter}")
    
    model.train()

if __name__ == '__main__':
    main() 