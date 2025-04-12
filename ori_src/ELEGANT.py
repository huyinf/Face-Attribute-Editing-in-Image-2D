# -*- coding:utf-8 -*-
# Created Time: 2018/03/12 10:48:38
# Author: Taihong Xiao <xiaotaihong@126.com>

from dataset import config, MultiCelebADataset
from nets import Encoder, Decoder, Discriminator

import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from tensorboardX import SummaryWriter
from itertools import chain

# from distutils.version import LooseVersion
from packaging import version
from time import time

class ELEGANT(object):
    def __init__(self, args,
                 config=config, dataset=MultiCelebADataset, \
                 encoder=Encoder, decoder=Decoder, discriminator=Discriminator):

        self.args = args
        self.attributes = args.attributes
        self.n_attributes = len(self.attributes)
        self.gpu = args.gpu
        self.mode = args.mode
        self.restore = args.restore

        # init dataset and networks
        self.config = config
        self.dataset = dataset(self.attributes)
        self.Enc = encoder()
        self.Dec = decoder()
        self.D1 = discriminator(self.n_attributes, self.config.nchw[-1])
        self.D2 = discriminator(self.n_attributes, self.config.nchw[-1] // 2)

        # define loss functions
        self.adv_criterion = torch.nn.BCELoss()
        self.recon_criterion = torch.nn.MSELoss()

        self.restore_from_file()
        self.set_mode_and_gpu()

    def restore_from_file(self):
        """
        Restore model parameters from checkpoint files.

        This method checks if a checkpoint ID is provided for restoration.
        If a checkpoint ID is provided, it loads the encoder and decoder
        parameters from the corresponding checkpoint files. If the mode is
        'train', it also loads discriminator parameters. The method updates
        the start step for training.

        Returns:
            None
        """
        if self.restore is not None:  # Check if a checkpoint ID is provided for restoration
            # Construct paths to encoder and decoder checkpoint files
            ckpt_file_enc = os.path.join(self.config.model_dir, 'Enc_iter_{:06d}.pth'.format(self.restore))
            assert os.path.exists(ckpt_file_enc)  # Ensure the encoder checkpoint file exists
            ckpt_file_dec = os.path.join(self.config.model_dir, 'Dec_iter_{:06d}.pth'.format(self.restore))
            assert os.path.exists(ckpt_file_dec)  # Ensure the decoder checkpoint file exists

            # Load encoder and decoder parameters from checkpoint files
            if self.gpu:
                self.Enc.load_state_dict(torch.load(ckpt_file_enc), strict=False)
                self.Dec.load_state_dict(torch.load(ckpt_file_dec), strict=False)
            else:
                self.Enc.load_state_dict(torch.load(ckpt_file_enc, map_location='cpu'), strict=False)
                self.Dec.load_state_dict(torch.load(ckpt_file_dec, map_location='cpu'), strict=False)

            # If in training mode, load discriminator parameters
            if self.mode == 'train':
                # Construct paths to discriminator checkpoint files
                ckpt_file_d1 = os.path.join(self.config.model_dir, 'D1_iter_{:06d}.pth'.format(self.restore))
                assert os.path.exists(ckpt_file_d1)  # Ensure the D1 discriminator checkpoint file exists
                ckpt_file_d2 = os.path.join(self.config.model_dir, 'D2_iter_{:06d}.pth'.format(self.restore))
                assert os.path.exists(ckpt_file_d2)  # Ensure the D2 discriminator checkpoint file exists

                # Load discriminator parameters from checkpoint files
                if self.gpu:
                    self.D1.load_state_dict(torch.load(ckpt_file_d1), strict=False)
                    self.D2.load_state_dict(torch.load(ckpt_file_d2), strict=False)
                else:
                    self.D1.load_state_dict(torch.load(ckpt_file_d1, map_location='cpu'), strict=False)
                    self.D2.load_state_dict(torch.load(ckpt_file_d2, map_location='cpu'), strict=False)

            self.start_step = self.restore + 1  # Update the start step for training
        else:
            self.start_step = 1  # If no checkpoint ID is provided, start from step 1

    def set_mode_and_gpu(self):
        """
        Set the mode of the model (train or test) and configure GPU settings accordingly.

        If the mode is 'train', sets the model components (encoder, decoder, discriminators) to training mode,
        initializes optimizers and learning rate schedulers, moves components to GPU if available, and sets up
        data parallelism if multiple GPUs are used.

        If the mode is 'test', sets the model components (encoder, decoder) to evaluation mode, and moves
        components to GPU if available.

        Raises:
            NotImplementedError: If the mode is neither 'train' nor 'test'.

        Returns:
            None
        """
        if self.mode == 'train':
            # Set model components to training mode
            self.Enc.train()
            self.Dec.train()
            self.D1.train()
            self.D2.train()

            # Initialize SummaryWriter for logging
            self.writer = SummaryWriter(self.config.log_dir)

            # Initialize optimizers for generator (encoder and decoder) and discriminators
            self.optimizer_G = torch.optim.Adam(chain(self.Enc.parameters(), self.Dec.parameters()),
                                                lr=self.config.G_lr, betas=(0.5, 0.999),
                                                weight_decay=self.config.weight_decay)
            self.optimizer_D = torch.optim.Adam(chain(self.D1.parameters(), self.D2.parameters()),
                                                lr=self.config.D_lr, betas=(0.5, 0.999),
                                                weight_decay=self.config.weight_decay)

            # Initialize learning rate schedulers
            self.G_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=self.config.step_size,
                                                                  gamma=self.config.gamma)
            self.D_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=self.config.step_size,
                                                                  gamma=self.config.gamma)

            # If restoring from a checkpoint, adjust learning rates according to the restored step
            if self.restore is not None:
                for _ in range(self.restore):
                    self.G_lr_scheduler.step()
                    self.D_lr_scheduler.step()

            # Move model components to GPU if available
            if self.gpu:
                with torch.cuda.device(0):
                    self.Enc.cuda()
                    self.Dec.cuda()
                    self.D1.cuda()
                    self.D2.cuda()
                    self.adv_criterion.cuda()
                    self.recon_criterion.cuda()

            # Set up data parallelism if multiple GPUs are used
            if len(self.gpu) > 1:
                self.Enc = torch.nn.DataParallel(self.Enc, device_ids=list(range(len(self.gpu))))
                self.Dec = torch.nn.DataParallel(self.Dec, device_ids=list(range(len(self.gpu))))
                self.D1 = torch.nn.DataParallel(self.D1, device_ids=list(range(len(self.gpu))))
                self.D2 = torch.nn.DataParallel(self.D2, device_ids=list(range(len(self.gpu))))

        elif self.mode == 'test':
            # Set model components to evaluation mode
            self.Enc.eval()
            self.Dec.eval()

            # Move model components to GPU if available
            if self.gpu:
                with torch.cuda.device(0):
                    self.Enc.cuda()
                    self.Dec.cuda()

            # Set up data parallelism if multiple GPUs are used
            if len(self.gpu) > 1:
                self.Enc = torch.nn.DataParallel(self.Enc, device_ids=list(range(len(self.gpu))))
                self.Dec = torch.nn.DataParallel(self.Dec, device_ids=list(range(len(self.gpu))))

        else:
            # If mode is neither 'train' nor 'test', raise NotImplementedError
            raise NotImplementedError()

    def tensor2var(self, tensors, volatile=False):
        """
        Convert tensor or list of tensors to torch Variables.

        In PyTorch, Variable is a wrapper around a tensor that allows automatic differentiation,
        which is essential for backpropagation during training.
        By converting tensors to variables,
        you enable PyTorch to track the operations applied to those tensors and compute gradients with respect to them.

        starting from PyTorch version 0.4.0,
        Variable is deprecated, and tensors themselves can directly support autograd operations.
        So, in newer versions of PyTorch, you may not need to explicitly convert tensors to variables.

        Args:
            tensors (Tensor or list of Tensors): Input tensor(s).
            volatile (bool, optional): If True, the Variable will not be used in further gradient computations.

        Returns:
            Variable or list of Variables: Converted Variable(s).
        """

        # If input is not iterable, convert it to a list
        if not hasattr(tensors, '__iter__'):
            tensors = [tensors]

        out = []
        for tensor in tensors:
            # Move tensor to GPU if available
            if len(self.gpu):
                tensor = tensor.cuda(0)

            # Convert tensor to Variable
            if version.parse(torch.__version__) >= version.parse("0.4.0"):
                var = tensor
            else:
                var = torch.autograd.Variable(tensor, volatile=volatile)
            out.append(var)

        # If only one Variable is created, return it; otherwise, return the list of Variables
        if len(out) == 1:
            return out[0]
        else:
            return out

    def get_attr_chs(self, encodings, attribute_id):
        """
        Extracts the channels corresponding to a specific attribute from the encoding tensor.

        Args:
            encodings (torch.Tensor): The encoding tensor containing features.
            attribute_id (int): The index of the attribute for which channels are to be extracted.

        Returns:
            torch.Tensor: A subset of channels corresponding to the specified attribute.

          The narrow method takes three parameters: h (which is calculated as the difference between the ending and
          starting indices).
        """
        # Determines the total number of channels in the encoding tensor
        num_chs = encodings.size(1)
        # Calculates the number of channels per attribute
        per_chs = float(num_chs) / self.n_attributes
        # Calculates the starting index of the channels corresponding to the given attribute_id
        # It multiplies the number of channels per attribute by the attribute_id to determine the starting index.
        start = int(np.rint(per_chs * attribute_id))
        # Calculates the ending index of the channels corresponding to the given attribute_id.
        # It multiplies the number of channels per attribute by the next attribute_id to determine the ending index.
        end = int(np.rint(per_chs * (attribute_id + 1)))

        # Returns a narrowed view of the encoding tensor along the channel dimension,
        # selecting only the channels corresponding to the specified attribute.

        # parameters:
        #   the dimension along which to narrow (1 for channels)
        #   the starting index
        #   the length

        return encodings.narrow(1, start, end - start)

    def forward_G(self):
        """
        Performs the forward pass of the generator network (G).
        """
        # Encodes image A and returns the encoded features along with skip connections.
        self.z_A, self.A_skip = self.Enc(self.A, return_skip=True)
        # Encodes image B and returns the encoded features along with skip connections.
        self.z_B, self.B_skip = self.Enc(self.B, return_skip=True)

        # torch.cat([...], 1): Concatenates the encoded features corresponding to different attributes into a single tensor.
        # The encoded features from image A for attributes other than the specified attribute_id are concatenated with
        # the encoded features from image B for the specified attribute_id, and vice versa.

        self.z_C = torch.cat([self.get_attr_chs(self.z_A, i) if i != self.attribute_id \
                                  else self.get_attr_chs(self.z_B, i) for i in range(self.n_attributes)], 1)
        self.z_D = torch.cat([self.get_attr_chs(self.z_B, i) if i != self.attribute_id \
                                  else self.get_attr_chs(self.z_A, i) for i in range(self.n_attributes)], 1)

        # self.Dec(...): Decodes the concatenated encoded features to generate reconstructed images.
        self.R_A = self.Dec(self.z_A, self.z_A, skip=self.A_skip)
        self.R_B = self.Dec(self.z_B, self.z_B, skip=self.B_skip)
        self.R_C = self.Dec(self.z_C, self.z_A, skip=self.A_skip)
        self.R_D = self.Dec(self.z_D, self.z_B, skip=self.B_skip)

        # torch.clamp(...): Applies element-wise clamping to ensure the pixel values of the reconstructed images
        # are within the valid range [-1, 1].
        self.A1 = torch.clamp(self.A + self.R_A, -1, 1)
        self.B1 = torch.clamp(self.B + self.R_B, -1, 1)
        self.C = torch.clamp(self.A + self.R_C, -1, 1)
        self.D = torch.clamp(self.B + self.R_D, -1, 1)

    def forward_D_real_sample(self):
        """
        Performs the forward pass of the discriminator network (D) using real samples.
        """
        # print("forward_D_real_sample: in")
        # Pass real samples A and B through discriminator D1 and D2
        # print("self.d1_A"+"-"*20)
        self.d1_A = self.D1(self.A, self.y_A)
        # print("self.d1_B"+"-"*20)
        self.d1_B = self.D1(self.B, self.y_B)
        # print("self.d2_A"+"-"*20)
        self.d2_A = self.D2(self.A, self.y_A)
        # print("self.d2_B"+"-"*20)
        self.d2_B = self.D2(self.B, self.y_B)

    def forward_D_fake_sample(self, detach):
        """
        Performs the forward pass of the discriminator network (D) using fake samples.

        Args:
        - detach (bool): If True, detach the computation graph to prevent gradient flow during backward pass.

        Detaching gradients refers to a process in PyTorch where you stop the gradient from flowing backward
        through a specific computation graph. When you call .detach() on a tensor or a variable,
        it creates a new tensor that shares the same data, but does not have the gradient history attached to it.
        This means that any operation on this new tensor will not have its gradients computed during the backward pass.

        Detaching gradients is useful in scenarios such as training generative models like GANs,
        where you want to update only the discriminator's parameters and not the generator's parameters
        during certain steps of training. This separation helps stabilize training and prevents the generator from
        collapsing to a single solution.
        """

        # Clone attribute labels y_A and y_B and modify the attribute corresponding to attribute_id
        self.y_C, self.y_D = self.y_A.clone(), self.y_B.clone()
        self.y_C.data[:, self.attribute_id] = self.y_B.data[:, self.attribute_id]
        self.y_D.data[:, self.attribute_id] = self.y_A.data[:, self.attribute_id]

        # Pass manipulated fake samples through discriminator D1 and D2

        # Passes the manipulated fake samples C and D (with detached gradients) through discriminators D1 and D2.
        # This means that the gradients will not flow back through the generator during the subsequent backward pass.
        if detach:
            self.d1_C = self.D1(self.C.detach(), self.y_C)
            self.d1_D = self.D1(self.D.detach(), self.y_D)
            self.d2_C = self.D2(self.C.detach(), self.y_C)
            self.d2_D = self.D2(self.D.detach(), self.y_D)

        # Passes the manipulated fake samples C and D (with gradients) through discriminators D1 and D2.
        # This allows the gradients to flow back through the generator during the subsequent backward pass.
        else:
            self.d1_C = self.D1(self.C, self.y_C)
            self.d1_D = self.D1(self.D, self.y_D)
            self.d2_C = self.D2(self.C, self.y_C)
            self.d2_D = self.D2(self.D, self.y_D)


    # computed based on the adversarial criterion (usually a binary cross-entropy loss)
    def compute_loss_D(self):
        self.D_loss = {
            'D1': self.adv_criterion(self.d1_A, torch.ones_like(self.d1_A)) + \
                  self.adv_criterion(self.d1_B, torch.ones_like(self.d1_B)) + \
                  self.adv_criterion(self.d1_C, torch.zeros_like(self.d1_C)) + \
                  self.adv_criterion(self.d1_D, torch.zeros_like(self.d1_D)),

            'D2': self.adv_criterion(self.d2_A, torch.ones_like(self.d2_A)) + \
                  self.adv_criterion(self.d2_B, torch.ones_like(self.d2_B)) + \
                  self.adv_criterion(self.d2_C, torch.zeros_like(self.d2_C)) + \
                  self.adv_criterion(self.d2_D, torch.zeros_like(self.d2_D)),
        }
        self.loss_D = (self.D_loss['D1'] + 0.5 * self.D_loss['D2']) / 4

    def compute_loss_G(self):
        self.G_loss = {
            'reconstruction': self.recon_criterion(self.A1, self.A) + self.recon_criterion(self.B1, self.B),
            'adv1': self.adv_criterion(self.d1_C, torch.ones_like(self.d1_C)) + \
                    self.adv_criterion(self.d1_D, torch.ones_like(self.d1_D)),
            'adv2': self.adv_criterion(self.d2_C, torch.ones_like(self.d2_C)) + \
                    self.adv_criterion(self.d2_D, torch.ones_like(self.d2_D)),
        }
        self.loss_G = 5 * self.G_loss['reconstruction'] + self.G_loss['adv1'] + 0.5 * self.G_loss['adv2']

    def backward_D(self):
        self.loss_D.backward()
        self.optimizer_D.step()

    def backward_G(self):
        self.loss_G.backward()
        self.optimizer_G.step()

    def img_denorm(self, img, scale=255):
        return (img + 1) * scale / 2.

    def save_image_log(self, save_num=20):
        """
        saving images to the log directory for visualization during training

        Specifies the maximum number of images to save for each variable

        The cpu() method moves the data from GPU to CPU for compatibility with operations like numpy() and Image.fromarray().
        :return:
        """
        image_info = {
            'A/img': self.img_denorm(self.A.data.cpu(), 1)[:save_num],
            'B/img': self.img_denorm(self.B.data.cpu(), 1)[:save_num],
            'C/img': self.img_denorm(self.C.data.cpu(), 1)[:save_num],
            'D/img': self.img_denorm(self.D.data.cpu(), 1)[:save_num],
            'A1/img': self.img_denorm(self.A1.data.cpu(), 1)[:save_num],
            'B1/img': self.img_denorm(self.B1.data.cpu(), 1)[:save_num],
            'R_A/img': self.img_denorm(self.R_A.data.cpu(), 1)[:save_num],
            'R_B/img': self.img_denorm(self.R_B.data.cpu(), 1)[:save_num],
            'R_C/img': self.img_denorm(self.R_C.data.cpu(), 1)[:save_num],
            'R_D/img': self.img_denorm(self.R_D.data.cpu(), 1)[:save_num],
        }
        for tag, images in image_info.items():
            for idx, image in enumerate(images):
                self.writer.add_image(tag + '/{}_{:02d}'.format(self.attribute_id, idx), image, self.step)

    def save_sample_images(self, save_num=5):
        canvas = torch.cat((self.A, self.B, self.C, self.D, self.A1, self.B1), -1)
        img_array = np.transpose(self.img_denorm(canvas.data.cpu().numpy()), (0, 2, 3, 1)).astype(np.uint8)
        for i in range(save_num):
            Image.fromarray(img_array[i]).save(os.path.join(
                self.config.img_dir,'step_{:06d}_attr_{}_{:02d}.jpg'.format(self.step, self.attribute_id,i)))

    def save_scalar_log(self):
        scalar_info = {
            'loss_D': self.loss_D.data.cpu().numpy()[0],
            'loss_G': self.loss_G.data.cpu().numpy()[0],
            'G_lr': self.G_lr_scheduler.get_lr()[0],
            'D_lr': self.D_lr_scheduler.get_lr()[0],
        }

        for key, value in self.G_loss.items():
            scalar_info['G_loss/' + key] = value.data[0]

        for key, value in self.D_loss.items():
            scalar_info['D_loss/' + key] = value.data[0]

        for tag, value in scalar_info.items():
            self.writer.add_scalar(tag, value, self.step)

    def save_model(self):
        """
        The state_dict() method in PyTorch returns a dictionary object that maps each layer to its parameter tensor.

        Before saving, the .cpu() method is called on each parameter value to ensure that
        they are in CPU memory and can be serialized properly.
        """
        reduced = lambda key: key[7:] if key.startswith('module.') else key
        torch.save({reduced(key): val.cpu() for key, val in self.Enc.state_dict().items()},
                   os.path.join(self.config.model_dir, 'Enc_iter_{:06d}.pth'.format(self.step)))
        torch.save({reduced(key): val.cpu() for key, val in self.Dec.state_dict().items()},
                   os.path.join(self.config.model_dir, 'Dec_iter_{:06d}.pth'.format(self.step)))
        torch.save({reduced(key): val.cpu() for key, val in self.D1.state_dict().items()},
                   os.path.join(self.config.model_dir, 'D1_iter_{:06d}.pth'.format(self.step)))
        torch.save({reduced(key): val.cpu() for key, val in self.D2.state_dict().items()},
                   os.path.join(self.config.model_dir, 'D2_iter_{:06d}.pth'.format(self.step)))

    def train(self):
        for self.step in range(self.start_step, 1 + self.config.max_iter):

            for self.attribute_id in range(self.n_attributes):
                A, y_A = next(self.dataset.gen(self.attribute_id, True))
                B, y_B = next(self.dataset.gen(self.attribute_id, False))
                self.A, self.y_A, self.B, self.y_B = self.tensor2var([A, y_A, B, y_B])

                self.forward_G()

                self.forward_D_real_sample()
                self.forward_D_fake_sample(detach=True)
                self.compute_loss_D()
                self.optimizer_D.zero_grad()

                self.backward_D()

                self.forward_D_fake_sample(detach=False)
                self.compute_loss_G()
                self.optimizer_G.zero_grad()

                self.backward_G()

                #  this code has bug and the team cannot solve, and we don't need this log too much
                # if self.step % 100 == 0:
                #     self.save_image_log()

                if self.step % 2000 == 0:
                    self.save_sample_images()

                # Free up GPU memory by deleting unused tensors
                del A, y_A, B, y_B

            self.G_lr_scheduler.step()
            self.D_lr_scheduler.step()

            print('step: %06d, loss D: %.6f, loss G: %.6f' % (
                self.step, self.loss_D.data.cpu().numpy(), self.loss_G.data.cpu().numpy()))

            # same issue with image_log
            # if self.step % 100 == 0:
            #     self.save_scalar_log()

            if self.step % 2000 == 0:
                self.save_model()

        print('Finished Training!')
        self.writer.close()

    def transform(self, *images):
        transform1 = transforms.Compose([
            transforms.Resize(self.config.nchw[-2:]),
            transforms.ToTensor(),
        ])
        """
        This reshapes the tensor x into a new shape. The view method is used to change the shape of the tensor.
        The first dimension is set to 1, which effectively adds a new dimension to the tensor, making it a 4D tensor.
        The *x.size() syntax is a way of unpacking the dimensions of the original tensor x.
        It allows us to specify the remaining dimensions based on the size of x.
        
        Subtracting 1 shifts the scaled values so that they are centered around zero, with the resulting range being [-1, 1].
        """
        transform2 = lambda x: x.view(1, *x.size()) * 2 - 1
        out = [transform2(transform1(image)) for image in images]
        return out

    def swap(self):
        '''
        Swap attributes of two images.
        '''
        # Select the attribute to swap based on the first element of the swap_list
        self.attribute_id = self.args.swap_list[0]

        # Load and transform the two images
        images = self.transform(Image.open(self.args.input), Image.open(self.args.target[0]))
        self.B, self.A = self.tensor2var(images, volatile=True)

        # Perform a forward pass through the generator network
        self.forward_G()

        # Concatenate the original images B and A along with their swapped versions D and C horizontally
        img = torch.cat((self.B, self.A, self.D, self.C), -1)

        # Denormalize the image tensor and convert it to a numpy array
        img = np.transpose(self.img_denorm(img.data.cpu().numpy()), (0, 2, 3, 1)).astype(np.uint8)[0]

        # Save the resulting swapped image to a file named 'swap.jpg'
        Image.fromarray(img).save('swap.jpg')

    def linear(self):
        '''
        Perform linear interpolation between two images.
        '''
        # Select the attribute to interpolate based on the first element of the swap_list
        self.attribute_id = self.args.swap_list[0]

        # Load and transform the two images
        images = self.transform(Image.open(self.args.input), Image.open(self.args.target[0]))
        self.B, self.A = self.tensor2var(images, volatile=True)

        # Encode the images A and B
        self.z_A = self.Enc(self.A, return_skip=False)
        self.z_B, self.B_skip = self.Enc(self.B, return_skip=True)

        # Compute the encoded representation for the interpolated image
        self.z_D = torch.cat([self.get_attr_chs(self.z_B, i) if i != self.attribute_id \
                                  else self.get_attr_chs(self.z_A, i) for i in range(self.n_attributes)], 1)

        # Determine the number of interpolated images to generate
        m = self.args.size[0]
        out = [self.B]

        # Perform linear interpolation
        for i in range(1, 1 + m):
            z_i = float(i) / m * (self.z_D - self.z_B) + self.z_B
            R_i = self.Dec(z_i, self.z_B, skip=self.B_skip)
            D_i = torch.clamp(self.B + R_i, -1, 1)
            out.append(D_i)

        # Append the second original image to the end of the list of interpolated images
        out.append(self.A)

        # Concatenate the interpolated images and the original images horizontally
        out = torch.cat(out, -1)

        # Denormalize the image tensor and convert it to a numpy array
        img = np.transpose(self.img_denorm(out.data.cpu().numpy()), (0, 2, 3, 1)).astype(np.uint8)[0]

        # Save the resulting linearly interpolated image to a file named 'linear_interpolation.jpg'
        Image.fromarray(img).save('linear_interpolation.jpg')

    def matrix1(self):
        '''
        Perform matrix interpolation with respect to one attribute.
        '''
        # Select the attribute to interpolate based on the first element of the swap_list
        self.attribute_id = self.args.swap_list[0]

        # Load and transform the base image (B)
        self.B = self.tensor2var(self.transform(Image.open(self.args.input)), volatile=True)

        # Load and transform the target images (As)
        self.As = [self.tensor2var(self.transform(Image.open(self.args.target[i])), volatile=True) for i in range(3)]

        # Encode the base image (B) and the target images (As)
        self.z_B, self.B_skip = self.Enc(self.B, return_skip=True)
        self.z_As = [self.Enc(self.As[i], return_skip=False) for i in range(3)]

        # Compute the encoded representations for the interpolated images (Ds)
        self.z_Ds = [torch.cat([self.get_attr_chs(self.z_B, i) if i != self.attribute_id \
                                    else self.get_attr_chs(self.z_As[j], i) for i in range(self.n_attributes)], 1)
                     for j in range(3)]

        # Extract size information
        m, n = self.args.size
        h, w = self.config.nchw[-2:]

        # Initialize the output tensor
        out = torch.ones(1, 3, m * h, n * w)

        # Perform matrix interpolation
        for i in range(m):
            for j in range(n):
                # Compute interpolation coefficients
                a = i / float(m - 1)
                b = j / float(n - 1)
                four = [(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b]

                # Compute interpolated encoded representation
                z_ij = four[0] * self.z_B + four[1] * self.z_Ds[0] + four[2] * self.z_Ds[1] + four[3] * self.z_Ds[2]

                # Decode the interpolated representation to obtain the interpolated image
                R_ij = self.Dec(z_ij, self.z_B, skip=self.B_skip)
                D_ij = torch.clamp(self.B + R_ij, -1, 1)

                # Update the output tensor with the interpolated image
                out[:, :, i * h:(i + 1) * h, j * w:(j + 1) * w] = D_ij.data.cpu()

        # Create a canvas for visualization
        first_col = torch.cat((self.B.data.cpu(), torch.ones(1, 3, (m - 2) * h, w), self.As[1].data.cpu()), -2)
        last_col = torch.cat((self.As[0].data.cpu(), torch.ones(1, 3, (m - 2) * h, w), self.As[2].data.cpu()), -2)
        canvas = torch.cat((first_col, out, last_col), -1)

        # Denormalize the image tensor and convert it to a numpy array
        img = np.transpose(self.img_denorm(canvas.numpy()), (0, 2, 3, 1)).astype(np.uint8)[0]

        # Save the resulting matrix interpolation image to a file named 'matrix_interpolation1.jpg'
        Image.fromarray(img).save('matrix_interpolation1.jpg')

    def matrix2(self):
        '''
        Perform matrix interpolation with respect to two attributes simultaneously.
        '''
        # Select the attributes to interpolate based on the elements of the swap_list
        self.attribute_ids = self.args.swap_list

        # Load and transform the base image (B) and the target images (A1, A2)
        self.B, self.A1, self.A2 = self.tensor2var(
            self.transform(Image.open(self.args.input), Image.open(self.args.target[0]),
                           Image.open(self.args.target[1])), volatile=True)

        # Encode the base image (B) and the target images (A1, A2)
        self.z_B, self.B_skip = self.Enc(self.B, return_skip=True)
        self.z_A1, self.z_A2 = self.Enc(self.A1, return_skip=False), self.Enc(self.A2, return_skip=False)

        # Compute the encoded representations for the interpolated images (D1, D2)
        self.z_D1 = torch.cat([self.get_attr_chs(self.z_B, i) if i != self.attribute_ids[0]
                               else self.get_attr_chs(self.z_A1, i) for i in range(self.n_attributes)], 1)

        self.z_D2 = torch.cat([self.get_attr_chs(self.z_B, i) if i != self.attribute_ids[1]
                               else self.get_attr_chs(self.z_A2, i) for i in range(self.n_attributes)], 1)

        # Extract size information
        m, n = self.args.size
        h, w = self.config.nchw[-2:]

        # Initialize the output tensor
        out = torch.ones(1, 3, m * h, n * w)

        # Perform matrix interpolation
        for i in range(m):
            for j in range(n):
                # Compute interpolation coefficients
                a = i / float(m - 1)
                b = j / float(n - 1)

                # Compute interpolated encoded representation
                z_ij = a * self.z_D1 + b * self.z_D2 + (1 - a - b) * self.z_B
                R_ij = self.Dec(z_ij, self.z_B, skip=self.B_skip)
                D_ij = torch.clamp(self.B + R_ij, -1, 1)

                # Update the output tensor with the interpolated image
                out[:, :, i * h:(i + 1) * h, j * w:(j + 1) * w] = D_ij.data.cpu()

        # Create a canvas for visualization
        first_col = torch.cat((self.B.data.cpu(), torch.ones(1, 3, (m - 2) * h, w), self.A1.data.cpu()), -2)
        last_col = torch.cat((self.A2.data.cpu(), torch.ones(1, 3, (m - 1) * h, w)), -2)
        canvas = torch.cat((first_col, out, last_col), -1)

        # Denormalize the image tensor and convert it to a numpy array
        img = np.transpose(self.img_denorm(canvas.numpy()), (0, 2, 3, 1)).astype(np.uint8)[0]

        # Save the resulting matrix interpolation image to a file named 'matrix_interpolation2.jpg'
        Image.fromarray(img).save('matrix_interpolation2.jpg')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--attributes', nargs='+', type=str, help='Specify attribute names.')
    parser.add_argument('-g', '--gpu', default=[], nargs='+', type=str, help='Specify GPU ids.')
    parser.add_argument('-m', '--mode', default='train', type=str, choices=['train', 'test'])
    parser.add_argument('-r', '--restore', default=None, action='store', type=int,
                        help='Specify checkpoint id to restore')

    # test parameters
    parser.add_argument('--swap', action='store_true', help='Swap attributes.')
    parser.add_argument('--linear', action='store_true', help='Linear interpolation.')
    parser.add_argument('--matrix', action='store_true', help='Matraix interpolation with respect to one attribute.')
    parser.add_argument('--swap_list', default=[], nargs='+', type=int, help='Specify the attributes ids for swapping.')
    parser.add_argument('-i', '--input', type=str, help='Specify the input image.')
    parser.add_argument('-t', '--target', nargs='+', type=str, help='Specify target images.')
    parser.add_argument('-s', '--size', nargs='+', type=int, help='Specify the interpolation size.')

    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
    if args.mode == 'test':
        assert args.swap + args.linear + args.matrix == 1
        assert args.restore is not None

    print("init model")
    model = ELEGANT(args)
    if args.mode == 'train':
        model.train()
    elif args.mode == 'test' and args.swap:
        assert len(args.swap_list) == 1 and args.input and len(args.target) == 1
        model.swap()
    elif args.mode == 'test' and args.linear:
        assert len(args.swap_list) == 1 and len(args.size) == 1
        model.linear()
    elif args.mode == 'test' and args.matrix:
        assert len(args.swap_list) in [1, 2]
        if len(args.swap_list) == 1:
            assert len(args.target) == 3 and len(args.size) == 2
            model.matrix1()
        elif len(args.swap_list) == 2:
            assert len(args.target) == 2 and len(args.size) == 2
            model.matrix2()
    else:
        raise NotImplementedError()


if __name__ == "__main__":

    main()
