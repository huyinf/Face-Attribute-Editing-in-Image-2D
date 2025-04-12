"""
ELEGANT: Entangled Latent Encoding for Attribute Manipulation
Main model implementation.
"""

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from pathlib import Path

from ..datasets.celeba import MultiCelebADataset
from .components import Encoder, Decoder, Discriminator

class ELEGANT:
    """ELEGANT model for attribute manipulation."""
    
    def __init__(self, args, config):
        """
        Args:
            args: Command line arguments
            config: Configuration object
        """
        self.args = args
        self.config = config
        self.attributes = args.attributes
        self.n_attributes = len(self.attributes)
        self.device = torch.device(config.training_params['device'])

        # Initialize dataset and networks
        self.dataset = MultiCelebADataset(self.attributes, config)
        self.Enc = Encoder().to(self.device)
        self.Dec = Decoder().to(self.device)
        self.D1 = Discriminator(self.n_attributes, config.model_params['nchw'][-1]).to(self.device)
        self.D2 = Discriminator(self.n_attributes, config.model_params['nchw'][-1] // 2).to(self.device)

        # Define loss functions
        self.adv_criterion = nn.BCELoss()
        self.recon_criterion = nn.MSELoss()

        # Setup training
        self._setup_training()
        self._restore_from_checkpoint()

    def _setup_training(self):
        """Setup optimizers and learning rate schedulers."""
        # Generator (Encoder + Decoder) optimizer
        self.optimizer_G = Adam(
            list(self.Enc.parameters()) + list(self.Dec.parameters()),
            lr=self.config.model_params['G_lr'],
            betas=self.config.model_params['betas'],
            weight_decay=self.config.model_params['weight_decay']
        )

        # Discriminator optimizer
        self.optimizer_D = Adam(
            list(self.D1.parameters()) + list(self.D2.parameters()),
            lr=self.config.model_params['D_lr'],
            betas=self.config.model_params['betas'],
            weight_decay=self.config.model_params['weight_decay']
        )

        # Learning rate schedulers
        self.G_scheduler = StepLR(
            self.optimizer_G,
            step_size=self.config.model_params['step_size'],
            gamma=self.config.model_params['gamma']
        )
        self.D_scheduler = StepLR(
            self.optimizer_D,
            step_size=self.config.model_params['step_size'],
            gamma=self.config.model_params['gamma']
        )

        # TensorBoard writer
        self.writer = SummaryWriter(self.config.log_dir)

    def _restore_from_checkpoint(self):
        """Restore model from checkpoint if specified."""
        if self.args.restore is not None:
            checkpoint_path = self.config.model_dir / f'checkpoint_{self.args.restore:06d}.pt'
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path)
                self.Enc.load_state_dict(checkpoint['encoder'])
                self.Dec.load_state_dict(checkpoint['decoder'])
                self.D1.load_state_dict(checkpoint['discriminator1'])
                self.D2.load_state_dict(checkpoint['discriminator2'])
                self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
                self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
                self.G_scheduler.load_state_dict(checkpoint['G_scheduler'])
                self.D_scheduler.load_state_dict(checkpoint['D_scheduler'])
                self.start_step = checkpoint['step'] + 1
            else:
                raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
        else:
            self.start_step = 1

    def _save_checkpoint(self, step):
        """Save model checkpoint."""
        checkpoint = {
            'encoder': self.Enc.state_dict(),
            'decoder': self.Dec.state_dict(),
            'discriminator1': self.D1.state_dict(),
            'discriminator2': self.D2.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'G_scheduler': self.G_scheduler.state_dict(),
            'D_scheduler': self.D_scheduler.state_dict(),
            'step': step
        }
        torch.save(checkpoint, self.config.model_dir / f'checkpoint_{step:06d}.pt')

    def _get_attribute_channels(self, encodings, attribute_id):
        """Get channels corresponding to a specific attribute."""
        num_chs = encodings.size(1)
        per_chs = float(num_chs) / self.n_attributes
        start = int(per_chs * attribute_id)
        end = int(per_chs * (attribute_id + 1))
        return encodings.narrow(1, start, end - start)

    def forward_G(self, A, B, y_A, y_B, attribute_id):
        """Forward pass through generator."""
        # Encode images
        z_A, A_skip = self.Enc(A)
        z_B, B_skip = self.Enc(B)

        # Combine encodings for attribute transfer
        z_C = torch.cat([
            self._get_attribute_channels(z_A, i) if i != attribute_id
            else self._get_attribute_channels(z_B, i)
            for i in range(self.n_attributes)
        ], 1)

        z_D = torch.cat([
            self._get_attribute_channels(z_B, i) if i != attribute_id
            else self._get_attribute_channels(z_A, i)
            for i in range(self.n_attributes)
        ], 1)

        # Decode images
        R_A = self.Dec(z_A, z_A, skip=A_skip)
        R_B = self.Dec(z_B, z_B, skip=B_skip)
        R_C = self.Dec(z_C, z_A, skip=A_skip)
        R_D = self.Dec(z_D, z_B, skip=B_skip)

        # Reconstruct and transfer attributes
        A1 = torch.clamp(A + R_A, -1, 1)
        B1 = torch.clamp(B + R_B, -1, 1)
        C = torch.clamp(A + R_C, -1, 1)
        D = torch.clamp(B + R_D, -1, 1)

        return A1, B1, C, D

    def forward_D(self, images, labels, detach=False):
        """Forward pass through discriminator."""
        if detach:
            images = [img.detach() for img in images]
        
        d1_outputs = [self.D1(img, label) for img, label in zip(images, labels)]
        d2_outputs = [self.D2(img, label) for img, label in zip(images, labels)]
        
        return d1_outputs, d2_outputs

    def compute_losses(self, real_outputs, fake_outputs):
        """Compute generator and discriminator losses."""
        # Discriminator loss
        D_loss = {
            'D1': sum(self.adv_criterion(out, torch.ones_like(out)) for out in real_outputs[0]) +
                  sum(self.adv_criterion(out, torch.zeros_like(out)) for out in fake_outputs[0]),
            'D2': sum(self.adv_criterion(out, torch.ones_like(out)) for out in real_outputs[1]) +
                  sum(self.adv_criterion(out, torch.zeros_like(out)) for out in fake_outputs[1])
        }
        loss_D = (D_loss['D1'] + 0.5 * D_loss['D2']) / 4

        # Generator loss
        G_loss = {
            'reconstruction': self.recon_criterion(self.A1, self.A) + 
                            self.recon_criterion(self.B1, self.B),
            'adv1': sum(self.adv_criterion(out, torch.ones_like(out)) for out in fake_outputs[0]),
            'adv2': sum(self.adv_criterion(out, torch.ones_like(out)) for out in fake_outputs[1])
        }
        loss_G = 5 * G_loss['reconstruction'] + G_loss['adv1'] + 0.5 * G_loss['adv2']

        return loss_G, loss_D, G_loss, D_loss

    def train(self):
        """Train the model."""
        for step in range(self.start_step, self.config.model_params['max_iter'] + 1):
            for attribute_id in range(self.n_attributes):
                # Get batch of images
                A, y_A = next(self.dataset.get_generator(attribute_id, True))
                B, y_B = next(self.dataset.get_generator(attribute_id, False))
                
                A = A.to(self.device)
                B = B.to(self.device)
                y_A = y_A.to(self.device)
                y_B = y_B.to(self.device)

                # Forward pass through generator
                self.A1, self.B1, self.C, self.D = self.forward_G(A, B, y_A, y_B, attribute_id)

                # Forward pass through discriminator
                real_outputs = self.forward_D([A, B], [y_A, y_B])
                fake_outputs = self.forward_D([self.C, self.D], [y_A, y_B], detach=True)

                # Compute losses
                loss_G, loss_D, G_loss, D_loss = self.compute_losses(real_outputs, fake_outputs)

                # Update discriminator
                self.optimizer_D.zero_grad()
                loss_D.backward()
                self.optimizer_D.step()

                # Update generator
                self.optimizer_G.zero_grad()
                loss_G.backward()
                self.optimizer_G.step()

                # Log metrics
                if step % self.config.training_params['log_interval'] == 0:
                    self._log_metrics(step, loss_G, loss_D, G_loss, D_loss)

                # Save samples
                if step % self.config.training_params['sample_interval'] == 0:
                    self._save_samples(step, attribute_id)

            # Update learning rates
            self.G_scheduler.step()
            self.D_scheduler.step()

            # Save checkpoint
            if step % self.config.training_params['save_interval'] == 0:
                self._save_checkpoint(step)

        self.writer.close()

    def _log_metrics(self, step, loss_G, loss_D, G_loss, D_loss):
        """Log metrics to TensorBoard."""
        self.writer.add_scalar('loss/G', loss_G.item(), step)
        self.writer.add_scalar('loss/D', loss_D.item(), step)
        
        for key, value in G_loss.items():
            self.writer.add_scalar(f'G_loss/{key}', value.item(), step)
        
        for key, value in D_loss.items():
            self.writer.add_scalar(f'D_loss/{key}', value.item(), step)

    def _save_samples(self, step, attribute_id):
        """Save sample images."""
        # Save original and generated images
        samples = torch.cat([self.A, self.B, self.C, self.D, self.A1, self.B1], -1)
        samples = (samples + 1) / 2  # Convert to [0, 1]
        
        # Save to TensorBoard
        self.writer.add_images(
            f'samples/attr_{attribute_id}',
            samples,
            step
        )

    def swap_attributes(self, input_image, target_image, attribute_id):
        """Swap attributes between two images."""
        # Preprocess images
        A = self._preprocess_image(input_image)
        B = self._preprocess_image(target_image)
        
        # Forward pass
        A1, B1, C, D = self.forward_G(A, B, None, None, attribute_id)
        
        # Return swapped images
        return C, D

    def _preprocess_image(self, image):
        """Preprocess image for model input."""
        transform = transforms.Compose([
            transforms.Resize(self.config.model_params['nchw'][-2:]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform(image).unsqueeze(0).to(self.device) 