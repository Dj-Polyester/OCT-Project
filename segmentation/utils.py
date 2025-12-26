import random
from typing import Tuple, Dict, Mapping, Callable, Optional, Iterable
from pathlib import Path
from collections import defaultdict

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

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
from core_utils import IMG_EXTS

# Config from paper
GLOBAL_CONFIG = {
	# train data loader
	"batch_size": 32,
	# optimizer
	"optimizer": ["Adam"],
}

class OCT5kDataset(Dataset):
	@classmethod
	def from_path(
		cls, 
		root:str = "OCTData_pruned_padded", 
		images_root:str = "Images/Images_Original", 
		labels_root:str = "Masks/Masks_Manual/Grading_1",
	):
		OCT5kDataset.root = Path(root)
		OCT5kDataset.images_root = OCT5kDataset.root / images_root
		OCT5kDataset.masks_root = OCT5kDataset.root / labels_root
		mask_paths = [
			p for p in OCT5kDataset.masks_root.rglob("*") 
			if p.name.endswith(".png") and p.parent.parent.name.endswith(".E2E")
		]
		img_paths = []
		for p in mask_paths:
			rellabel_path = p.relative_to(OCT5kDataset.masks_root)
			relimg_path = rellabel_path.with_suffix(".TIFF")
			image_path = OCT5kDataset.images_root / relimg_path
			img_paths.append(image_path)
		return cls(mask_paths, img_paths)
	
	@classmethod
	def cls_name_from_path(cls, p: Path):
		return p.name.split(" Part")[0]

	@classmethod
	def classes(cls):
		if hasattr(cls, "CLASSES"):
			return cls.CLASSES

		# 2. Get all subdirectory names
		# We use .is_dir() to ignore any loose files
		cls.CLASSES = list(set([
			cls.cls_name_from_path(folder) for folder in OCT5kDataset.masks_root.iterdir() if folder.is_dir()
		]))

		return cls.CLASSES
	
	def __init__(
			self, 
			img_paths: list[Path],
			mask_paths: list[Path],
		):
		self.mask_paths = mask_paths
		self.img_paths = img_paths
		self.indices_by_cls = defaultdict(list)
		self.targets = [
			self.classes().index(self.cls_name_from_path(p.parent.parent.parent)) 
			for p in self.mask_paths
		]
		for i, p in enumerate(mask_paths):
			class_name = self.cls_name_from_path(p.parent.parent.parent)
			self.indices_by_cls[class_name].append(i)

	def __len__(self):
		return len(self.labels_paths)
	
	def img_mask_pair(self, index: int):
		mask_path: Path = self.mask_paths[index]
		img_path: Path = self.img_paths[index]

		# Load as BGR (Standard)
		img_bgr = cv.imread(img_path)
		
		# Convert to Grayscale
		#img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
		#image = torch.from_numpy(img_rgb).unsqueeze(0)

		# Convert to RGB (Crucial for PyTorch/ML) and permute dimensions
		img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
		image = torch.from_numpy(img_rgb).permute(2, 0, 1)

		mask = read_image(str(mask_path), mode=ImageReadMode.GRAY).squeeze(0).long()
		return image, mask

	def __getitem__(self, index: int):
		return self.img_mask_pair(index)
	
	def sample_from_class(self, cls_name: str, n: int):
		return random.sample(self.indices_by_cls[cls_name], n)
	
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

def dice_loss(preds: Tensor, target: Tensor) -> Tensor:
	preds = F.log_softmax(preds, dim=1).exp()
	target = F.one_hot(target, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
	return 1 - soft_dice_score(preds, target, dims=(0, 2, 3)).mean()

class TransferLearning(nn.Module):
	@classmethod
	def set_grads(cls, model, fill = None, verbose = False):
		for name, param in model.named_parameters():	
			param.requires_grad = (
				fill 
				if isinstance(fill, bool) 
				else name.startswith("decoder")) or name.startswith("segmentation_head") 
			if verbose:
				print(name, param.grad, param.requires_grad)

# LightningModule for Training
class LitSegmentation(L.LightningModule):
	TRAIN_LOSS = "Train Loss"
	VALIDATION_LOSS = "Validation Loss"
	TRAIN_MICRO_IOU = "Train Micro IOU"
	VALIDATION_MICRO_IOU = "Validation Micro IOU"
	TRAIN_MACRO_IOU = "Train Macro IOU"
	VALIDATION_MACRO_IOU = "Validation Macro IOU"

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
		self.num_classes = config.get("num_classses", 6) 
		runname = config["run"]

		self.model = smp.Unet(
			encoder_name=config.get("encoder"),        # Choose your backbone
			encoder_weights="imagenet",     # Use pretrained weights
			in_channels=3,                  # 1 for grayscale OCT images
			classes=self.num_classes,                      # Number of output classes (e.g., AMD, DME, Normal)
		)

		if runname == "step1":
			print("Freezing weights")
			# Step 1: Freeze weights
			TransferLearning.set_grads(self.model)
		elif runname == "step2":
			print("Unfreezing weights")
			# Step 2: Unfreeze weights
			TransferLearning.set_grads(self.model, True)
	def reset_parameters(self, verbose=False):
		for name, layer in self.model.named_modules():
			if verbose:
				print(name, layer)
			if hasattr(layer, 'reset_parameters'):
				layer.reset_parameters()

	def forward(self, *args, **kwargs):
		return self.model(*args, **kwargs)

	def cross_entropy(self, preds: Tensor, target: Tensor) -> Tensor:
		return F.cross_entropy(preds, target)

	def dice_loss(self, preds: Tensor, target: Tensor) -> Tensor:
		return smp.losses.DiceLoss(mode='multiclass')(preds, target)

	def get_metrics(self, batch, transforms, labels: Mapping[str, str]):
		instance, target = batch
		instance = transforms(instance)
		logits: Tensor = self.model(instance)
		preds: Tensor = torch.argmax(logits, dim=1)
		# Logging to TensorBoard (if installed) by default
		values = {}
		for k in labels:
			if k == "loss":
				values[labels["loss"]] = getattr(self, self.loss_type)(logits, target)
			elif k.endswith("-iou"):
				tp, fp, fn, tn = smp.metrics.get_stats(preds, target, mode='multiclass', num_classes=self.num_classes)
				reduction = None
				if k == "micro-iou":
					reduction = "micro"
				elif k == "macro-iou":
					reduction = "macro"	
				else:
					raise ValueError(f"Unknown IOU type: {k}")
				values[labels[k]] = smp.metrics.iou_score(tp, fp, fn, tn, reduction=reduction)
			
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
			{
				"loss": LitSegmentation.TRAIN_LOSS, 
				"micro-iou": LitSegmentation.TRAIN_MICRO_IOU,
				"macro-iou": LitSegmentation.TRAIN_MACRO_IOU,
			},
		)
	
	def validation_step(self, batch, batch_idx):
		return self.get_metrics(
			batch, 
			LitSegmentation.TRANSFORMS,
			{
				"loss": LitSegmentation.VALIDATION_LOSS, 
				"micro-iou": LitSegmentation.VALIDATION_MICRO_IOU,
				"macro-iou": LitSegmentation.VALIDATION_MACRO_IOU,
			},
		)

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.config["lr"])
		return optimizer