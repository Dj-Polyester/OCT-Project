from pathlib import Path

import numpy as np
from PIL import Image
import cv2 as cv

import torch
from torch.utils.data import Dataset

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2

TRANSFORMS = v2.Compose([
	v2.ToDtype(torch.float32),
	v2.Normalize(mean=[35.3662, 35.3662, 35.3662], std=[45.0505, 45.0505, 45.0505]), #OCT5k
	v2.ToDtype(torch.float32, scale=True),
])

class OCT5kDataset(Dataset):
	def __init__(
			self, 
			root:str = "OCTData", 
			images_root:str = "Images/Images_Original", 
			labels_root:str = "Masks/Masks_Manual/Grading_1",
		):
		OCT5kDataset.root = Path(root)
		OCT5kDataset.images_root = OCT5kDataset.root / images_root
		OCT5kDataset.labels_root = OCT5kDataset.root / labels_root

		self.labels_paths = [
			p for p in OCT5kDataset.labels_root.rglob("*") 
			if p.name.endswith(".png") and p.parent.parent.name.endswith(".E2E")
		]

	def __len__(self):
		return len(self.labels_paths)
	
	def __getitem__(self, index: int):
		label_path: Path = self.labels_paths[index]

		rellabel_path = label_path.relative_to(OCT5kDataset.labels_root)
		relimg_path = rellabel_path.with_suffix(".TIFF")

		image_path = OCT5kDataset.images_root / relimg_path

		# Load as BGR (Standard)
		img_bgr = cv.imread(image_path)
		
		# Convert to Grayscale
		#img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
		#image = torch.from_numpy(img_rgb).unsqueeze(0)

		# Convert to RGB (Crucial for PyTorch/ML) and permute dimensions
		img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
		image = torch.from_numpy(img_rgb).permute(2, 0, 1)

		label = read_image(str(label_path), mode=ImageReadMode.GRAY)
		return image, label
	
	@classmethod
	def classes(cls):
		if hasattr(cls, "CLASSES"):
			return cls.CLASSES

		# 2. Get all subdirectory names
		# We use .is_dir() to ignore any loose files
		folder_names = [folder.name for folder in OCT5kDataset.labels_root.iterdir() if folder.is_dir()]

		# 3. Clean the names (remove " Part1", " Part2", etc.)
		# We split by " Part" and take the first part of the string
		cls.CLASSES = sorted(list(set(name.split(" Part")[0] for name in folder_names)))

		return cls.CLASSES
	
class IterableOCT5kDataset(OCT5kDataset):
	def __init__(
			self, 
			root:str = "OCTData", 
			images_root:str = "Images/Images_Original", 
			labels_root:str = "Masks/Masks_Manual/Grading_1",
			transforms = v2.ToDtype(torch.float32),
		):
		super().__init__(root, images_root, labels_root)
		self.current_index = 0
		self.transforms = transforms

	def __iter__(self):
		return self
	def __next__(self):
		if self.current_index >= len(self):
			raise StopIteration
		
		image, _ = self[self.current_index]
		self.current_index += 1
		return self.transforms(image)