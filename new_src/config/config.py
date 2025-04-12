"""
Configuration settings for the ELEGANT model.
"""

import os
from pathlib import Path

class Config:
    """Configuration class for ELEGANT model settings."""
    
    def __init__(self):
        # Base directories
        self.base_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.base_dir / 'data'
        self.exp_dir = self.base_dir / 'experiments'
        self.model_dir = self.exp_dir / 'models'
        self.log_dir = self.exp_dir / 'logs'
        self.img_dir = self.exp_dir / 'images'

        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.exp_dir, self.model_dir, self.log_dir, self.img_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @property
    def model_params(self):
        """Model hyperparameters."""
        return {
            'nchw': [16, 3, 256, 256],  # [batch_size, channels, height, width]
            'G_lr': 2e-4,  # Generator learning rate
            'D_lr': 2e-4,  # Discriminator learning rate
            'betas': [0.5, 0.999],  # Adam optimizer betas
            'weight_decay': 1e-5,  # Weight decay for regularization
            'step_size': 3000,  # Learning rate scheduler step size
            'gamma': 0.97,  # Learning rate scheduler gamma
            'shuffle': True,  # Shuffle dataset during training
            'num_workers': 4,  # Number of data loading workers
            'max_iter': 200000,  # Maximum training iterations
            'num_samples': None,  # Number of samples in subset (None for all)
        }

    @property
    def dataset_params(self):
        """Dataset parameters."""
        return {
            'image_size': (256, 256),
            'batch_size': 16,
            'num_workers': 4,
            'pin_memory': True,
        }

    @property
    def training_params(self):
        """Training parameters."""
        return {
            'save_interval': 2000,  # Save model every N iterations
            'log_interval': 100,  # Log metrics every N iterations
            'sample_interval': 2000,  # Save sample images every N iterations
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }

# Create a global config instance
config = Config() 