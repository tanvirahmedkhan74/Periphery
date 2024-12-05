import os
import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    """
    Custom Dataset class for loading image-mask pairs for image segmentation tasks.

    Args:
        img_ids (list): List of image IDs (filenames without extensions).
        img_dir (str): Directory containing images.
        mask_dir (str): Directory containing masks.
        img_ext (str): File extension for images (e.g., '.jpg', '.png').
        mask_ext (str): File extension for masks (e.g., '.png').
        transform (callable, optional): Transform function (e.g., albumentations) to apply to image and mask.

    Dataset structure (expected):
    ├── images/
    │   ├── img001.jpg
    │   ├── img002.jpg
    │   ├── img003.jpg
    │   ├── ...
    │
    └── masks/
        ├── img001.png
        ├── img002.png
        ├── img003.png
        ├── ...

    Each image should have a corresponding mask with the same name (e.g., `img001.jpg` -> `img001.png`).

    The mask is assumed to be a single-channel (grayscale) binary mask.
    If no object is present in the mask (i.e., the mask is all zeros), the mask is set to 1 for binary consistency.
    """

    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, transform=None):
        """
        Initializes the dataset.

        Parameters:
            img_ids (list): List of image IDs (e.g., ['img001', 'img002', ...]).
            img_dir (str): Directory containing the images.
            mask_dir (str): Directory containing the corresponding masks.
            img_ext (str): Extension for image files (e.g., '.jpg').
            mask_ext (str): Extension for mask files (e.g., '.png').
            transform (callable, optional): Transform function to apply to the image and mask. Defaults to None.
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples (images) in the dataset.
        """
        return len(self.img_ids)

    def __getitem__(self, idx):
        """
        Fetches the image and corresponding mask for the given index.

        Args:
            idx (int): Index for fetching the image-mask pair.

        Returns:
            Tuple: A tuple containing:
                - img (Tensor): The image tensor of shape (C, H, W).
                - mask (Tensor): The corresponding mask tensor of shape (1, H, W).
                - metadata (dict): Dictionary containing metadata, in this case, the image ID.
        """
        # Get the image ID for this index
        img_id = self.img_ids[idx]

        # Load the image from the images directory
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image {img_path} not found!")

        # Load the corresponding mask from the masks directory
        mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask {mask_path} not found!")

        # Apply any transformations (e.g., resizing, normalization, etc.)
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # Normalize the image and mask
        img = img.astype('float32') / 255.0  # Normalize image to range [0, 1]
        img = img.transpose(2, 0, 1)  # Convert from HWC to CHW format for PyTorch

        mask = mask.astype('float32') / 255.0  # Normalize mask to range [0, 1]
        mask = mask[np.newaxis, ...]  # Add a channel dimension: (H, W) -> (1, H, W)

        # Convert mask values to binary (0 or 1) if necessary
        if mask.max() < 1:
            mask[mask > 0] = 1.0  # If the mask is all zeros, set the mask value to 1 for binary consistency

        # Return the image, mask, and metadata (image ID)
        return img, mask, {'img_id': img_id}