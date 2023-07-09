'''
bunch of helper functions to work with the ATLAS dataset
'''

import torchvision.transforms as transforms
import nibabel as nib
from torch.utils.data import Dataset
import torch
import os
import numpy as np


def slice_with_mask(img=None, mask=None, xslice=None, yslice=None, zslice=None):
    '''
    img -> can be either string(path name) or numpy array
    mask -> can be either string(path name) or numpy array
    '''

    import nibabel as nb
    from matplotlib import pyplot as plt
    import numpy as np
    import torch

    if isinstance(img, str):
        image = nb.load(img)
        mask = nb.load(mask)
        img_narr = image.get_fdata()
        mask_narr = mask.get_fdata()
    elif type(img) is torch.Tensor:
        # it's a torch tensor
        img_narr = img.numpy()
        mask_narr = mask.numpy()
    else:
        print("invalid input")

    import matplotlib.colors as colors
    # Define a custom colormap with transparency
    cmap = colors.ListedColormap(['none', 'red'])
    bounds = [-0.5, 0.5, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    if (xslice != None):
        img_narr = img_narr[xslice, :, :]
        mask_narr = mask_narr[xslice, :, :]
    elif (yslice != None):
        img_narr = img_narr[:, yslice, :]
        mask_narr = mask_narr[:, yslice, :]
    elif (zslice != None):
        img_narr = img_narr[:, :, zslice]
        mask_narr = mask_narr[:, :, zslice]

    fig, ax = plt.subplots()
    ax.imshow(img_narr, cmap='gray')
    ax.imshow(mask_narr, cmap=cmap, norm=norm, alpha=0.4)
    ax.axis('off')

    plt.show()


class ATLAS_Dataset(Dataset):
    '''
    returns a custom dataset object made from the root directory of the atlas dataset
    '''

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.input_files, self.mask_files = self._get_file_lists()

    def _get_file_lists(self):
        input_files = []
        mask_files = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith(".nii.gz"):
                    file_path = os.path.join(dirpath, filename)
                    if 'mask' in filename:
                        mask_files.append(file_path)
                    else:
                        input_files.append(file_path)
        return input_files, mask_files

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_file = self.input_files[index]
        mask_file = self.mask_files[index]

        # Load the input and output volumes using nibabel
        input_vol = nib.load(input_file)
        output_vol = nib.load(mask_file)

        # Get the numpy arrays from the loaded volumes
        input_arr = input_vol.get_fdata().astype(np.float32)
        output_arr = output_vol.get_fdata().astype(np.float32)

        # Convert the arrays to PyTorch tensors
        input_tensor = torch.from_numpy(input_arr).permute(2, 1, 0)
        output_tensor = torch.from_numpy(output_arr).permute(2, 1, 0)

        return torch.flip(input_tensor, [0]), torch.flip(output_tensor, [0])
        # output is of shape (189, 233, 193)


class ATLAS_Sliced_Dataset(ATLAS_Dataset):
    def __len__(self):
        return len(self.input_files)*189

    def __getitem__(self, index):
        from math import floor, ceil
        import torchvision.transforms as transforms

        file_index = floor(index/189)
        slice_index = index - 189*file_index

        input_file = self.input_files[file_index]
        mask_file = self.mask_files[file_index]

        # Load the input and output volumes using nibabel
        input_vol = nib.load(input_file)
        output_vol = nib.load(mask_file)

        # Get the numpy arrays from the loaded volumes
        input_arr = input_vol.get_fdata().astype(np.float32)
        output_arr = output_vol.get_fdata().astype(np.float32)

        # Convert the arrays to PyTorch tensors
        input_tensor = torch.from_numpy(input_arr).permute(2, 1, 0)
        output_tensor = torch.from_numpy(output_arr).permute(2, 1, 0)

        input_tensor = input_tensor[slice_index, :, :]
        output_tensor = output_tensor[slice_index, :, :]

        # zero padding to convert to shape 256*256
        current_size = (233, 197)
        desired_size = (256, 256)

        padding_height = desired_size[0] - current_size[0]
        padding_width = desired_size[1] - current_size[1]

        pad_transform = transforms.Pad(
            (ceil(padding_width / 2), ceil(padding_height / 2), floor(padding_width / 2), floor(padding_height / 2)))
        # left, top, right, bottom

        input_tensor = pad_transform(input_tensor).unsqueeze(0)
        output_tensor = pad_transform(output_tensor).unsqueeze(0)

        return torch.flip(input_tensor, [0]), torch.flip(output_tensor, [0])
        # output is of shape (256, 256) and is 2D and is flipped so that the neck is on the bottom side of the image


def center_crop(tensor, output_size):
    _, h, w = tensor.size()
    th, tw = output_size

    y1 = int((h - th) / 2)
    x1 = int((w - tw) / 2)

    return tensor[:, y1:y1+th, x1:x1+tw]
