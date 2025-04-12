"""
CelebA dataset implementation for ELEGANT model.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

class SingleCelebADataset(Dataset):
    """Single CelebA dataset for attribute-specific training."""
    
    def __init__(self, im_names, labels, config):
        """
        Args:
            im_names (list): List of image paths
            labels (np.ndarray): Array of attribute labels
            config (Config): Configuration object
        """
        self.im_names = im_names
        self.labels = labels
        self.config = config

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        """Get a single image and its label."""
        image = Image.open(self.im_names[idx])
        image = self.transform(image) * 2 - 1  # Normalize to [-1, 1]
        label = (self.labels[idx] + 1) / 2  # Convert to [0, 1]
        return image, label

    @property
    def transform(self):
        """Image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize(self.config.model_params['nchw'][-2:]),
            transforms.ToTensor(),
        ])

    def get_dataloader(self):
        """Get a DataLoader for this dataset."""
        return DataLoader(
            self,
            batch_size=self.config.model_params['nchw'][0],
            shuffle=self.config.model_params['shuffle'],
            num_workers=self.config.model_params['num_workers'],
            drop_last=True
        )

class MultiCelebADataset:
    """Multi-attribute CelebA dataset for ELEGANT model."""
    
    def __init__(self, attributes, config):
        """
        Args:
            attributes (list): List of attribute names
            config (Config): Configuration object
        """
        self.attributes = attributes
        self.config = config
        self._load_data()
        self._setup_generators()

    def _load_data(self):
        """Load CelebA dataset metadata."""
        attr_file = self.config.data_dir / 'list_attr_celeba.txt'
        
        with open(attr_file, 'r') as f:
            lines = f.read().strip().split('\n')
            col_ids = [lines[1].split().index(attr) + 1 for attr in self.attributes]
            self.all_labels = np.array([
                [int(x.split()[col_id]) for col_id in col_ids]
                for x in lines[2:]
            ], dtype=np.float32)
            
            self.im_names = np.array([
                str(self.config.data_dir / '_align_5p' / f'{idx+1:06d}.jpg')
                for idx in range(len(self.all_labels))
            ])

        if self.config.model_params['num_samples'] is not None:
            self.all_labels = self.all_labels[:self.config.model_params['num_samples']]
            self.im_names = self.im_names[:self.config.model_params['num_samples']]
        
        print(f"Total images: {len(self.im_names)}")

    def _setup_generators(self):
        """Setup data generators for each attribute and label combination."""
        self.generators = {
            i: {True: None, False: None}
            for i in range(len(self.attributes))
        }
        
        for attribute_id in range(len(self.attributes)):
            for is_positive in [True, False]:
                idxs = np.where(
                    self.all_labels[:, attribute_id] == (int(is_positive)*2 - 1)
                )[0]
                im_names = self.im_names[idxs]
                labels = self.all_labels[idxs]
                dataset = SingleCelebADataset(im_names, labels, self.config)
                self.generators[attribute_id][is_positive] = dataset.get_dataloader()

    def get_generator(self, attribute_id, is_positive):
        """Get data generator for specific attribute and label."""
        return self.generators[attribute_id][is_positive]

    def verify_files(self):
        """Verify that all image files exist."""
        missing_files = []
        for im_name in self.im_names:
            if not os.path.exists(im_name):
                missing_files.append(im_name)
        
        if missing_files:
            print(f"Warning: {len(missing_files)} files not found")
            return False
        return True 