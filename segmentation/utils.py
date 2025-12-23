from typing import Tuple, Dict, Mapping, Callable, Optional, Iterable
from pathlib import Path

import numpy as np
from PIL import Image
import cv2 as cv

import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
from torch.utils.data import Dataset

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2

import lightning as L
import torchmetrics.functional as MF

import segmentation_models_pytorch as smp

class OCT5kDataset(Dataset):
	def __init__(
			self, 
			root:str = "OCTData_pruned", 
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
	
class IterableOCT5kDataset(Iterable):
	def __init__(
			self, 
			oct5k: OCT5kDataset = OCT5kDataset(),
			transforms = v2.ToDtype(torch.float32),
		):
		self.oct5k = oct5k
		self.current_index = 0
		self.transforms = transforms

	def __len__(self):
		return len(self.oct5k)

	def __iter__(self):
		return self
	
	def __next__(self):
		if self.current_index >= len(self.oct5k):
			raise StopIteration
		
		image, _ = self.oct5k[self.current_index]
		self.current_index += 1
		return self.transforms(image)

def soft_dice_score(
	output: torch.Tensor,
	target: torch.Tensor,
	smooth: float = 0.0,
	eps: float = 1e-7,
	dims=None,
) -> torch.Tensor:
	assert output.size() == target.size()
	dice_score = soft_tversky_score(output, target, 0.5, 0.5, smooth, eps, dims)
	return dice_score


def soft_tversky_score(
	output: torch.Tensor,
	target: torch.Tensor,
	alpha: float,
	beta: float,
	smooth: float = 0.0,
	eps: float = 1e-7,
	dims=None,
) -> torch.Tensor:
	"""Tversky loss

	References:
		https://arxiv.org/pdf/2302.05666
		https://arxiv.org/pdf/2303.16296

	"""
	assert output.size() == target.size()

	output_sum = output.abs().sum(dim=dims)
	target_sum = target.abs().sum(dim=dims)
	difference = torch.linalg.vector_norm(output - target, ord=1, dim=dims)
	
	intersection = (output_sum + target_sum - difference) / 2  # TP
	fp = output_sum - intersection
	fn = target_sum - intersection

	tversky_score = (intersection + smooth) / (
		intersection + alpha * fp + beta * fn + smooth
	).clamp_min(eps)
	return tversky_score

# LightningModule for Training
class LitSegmentation(L.LightningModule):
	TRAIN_LOSS = "Train Loss"
	VALIDATION_LOSS = "Validation Loss"
	TRAIN_IOU = "Train IOU"
	VALIDATION_IOU = "Validation IOU"

	TRANSFORMS = v2.Compose([
		v2.ToDtype(torch.float32),
		#v2.Normalize(mean=[35.3662, 35.3662, 35.3662], std=[45.0505, 45.0505, 45.0505]), #OCT5k
		v2.Normalize(mean=[34.2609, 34.2609, 34.2609], std=[44.7659, 44.7659, 44.7659]), #OCT5k_padded
		#v2.Normalize(mean=[35.9994, 35.9994, 35.9994], std=[43.8581, 43.8581, 43.8581]), #OCT5k_resized
		v2.ToDtype(torch.float32, scale=True),
	])
	def __init__(self, config: Mapping):
		super().__init__()
		self.config = config
		self.loss_type = config.get("loss_type", "cross_entropy")

		self.model = smp.Unet(
			encoder_name=config.get("encoder"),        # Choose your backbone
			encoder_weights="imagenet",     # Use pretrained weights
			in_channels=3,                  # 1 for grayscale OCT images
			classes=6,                      # Number of output classes (e.g., AMD, DME, Normal)
		)

	def reset_parameters(self, verbose=False):
		for name, layer in self.model.named_modules():
			if verbose:
				print(name, layer)
			if hasattr(layer, 'reset_parameters'):
				layer.reset_parameters()

	def cross_entropy(self, preds: Tensor, target: Tensor) -> Tensor:
		return F.cross_entropy(preds, target)

	def dice_loss(self, preds: Tensor, target: Tensor) -> Tensor:
		return 1 - soft_dice_score(preds, target, dims=(0, 2, 3)).mean()

	def get_metrics(self, batch, transforms, labels: Mapping[str, str]):
		instance, target = batch
		instance = transforms(instance)
		preds: Tensor = self.model(instance)

		# Logging to TensorBoard (if installed) by default
		values = {}
		if "loss" in labels:
			values[labels["loss"]] = getattr(self, self.loss_type)(preds, target)
		if "iou" in labels:
			tp, fp, fn, tn = smp.metrics.get_stats(preds, target, mode='multilabel', threshold=0.5)
			values[labels["iou"]] = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
			
		self.log_dict(
			values, 
			prog_bar=True,
			on_step=False,
			on_epoch=True,
			reduce_fx="mean",
		)
		return {k: values[v] for k,v in labels.items()}

	def training_step(self, batch, batch_idx):
		return self.get_metrics(
			batch, 
			LitSegmentation.TRANSFORMS,
			{"loss": LitSegmentation.TRAIN_LOSS, "iou": LitSegmentation.TRAIN_IOU},
		)
	
	def validation_step(self, batch, batch_idx):
		return self.get_metrics(
			batch, 
			LitSegmentation.TRANSFORMS,
			{"loss": LitSegmentation.VALIDATION_LOSS, "iou": LitSegmentation.VALIDATION_IOU},
		)

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.config["lr"])
		return optimizer