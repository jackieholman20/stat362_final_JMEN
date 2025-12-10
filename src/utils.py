import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut

class RSNADataset(Dataset):
    """
    Custom Dataset class for RSNA Pneumonia Detection Challenge.
    Handles reading DICOM files, applying windowing, and converting to Tensor.
    """
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, "path"]
        label = int(self.dataframe.loc[idx, "target"])

        # Read the DICOM file
        ds = pydicom.dcmread(img_path)
        img = ds.pixel_array.astype(np.float32)

        # Apply VOI LUT (better windowing when available)
        try:
            img = apply_voi_lut(img, ds).astype(np.float32)
        except Exception:
            pass

        # Handle inverted grayscale (Monochrome1)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            img = img.max() - img

        # Robust normalize with percentile clipping (1st to 99th percentile)
        lo, hi = np.percentile(img, (1, 99))
        img = np.clip(img, lo, hi)
        img = (img - lo) / (hi - lo + 1e-6)  # Normalize to [0, 1]

        # Convert to 8-bit and 3-channel PIL for torchvision transforms
        img = (img * 255.0).astype(np.uint8)
        image = Image.fromarray(img).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)