from typing import Iterable
import time
from collections import defaultdict
import shutil
from pathlib import Path

import numpy as np
import cv2 as cv
from tqdm import tqdm

import torch
from torch import Tensor
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2
import kagglehub

def download_dataset(dataset_name: str):
	print(f"Downloading dataset {dataset_name} from Kaggle...")
	path = kagglehub.dataset_download(dataset_name)
	print("Path to dataset files:", path)
	return path

def copy_folder(src, dst, desc):
	src_path = Path(src)
	dst_path = Path(dst)
	# Copy folder structure and process images
	
	with tqdm(
		total=sum(1 for _ in src_path.rglob('*') if not _.is_dir()),
		desc=desc,
	) as pbar:
		for item in src_path.rglob('*'):
			rel_path = item.relative_to(src_path)
			dest_item = dst_path / rel_path
			if item.is_dir():
				dest_item.mkdir(parents=True, exist_ok=True)
			else:
				dest_item.parent.mkdir(parents=True, exist_ok=True)
				shutil.copy2(item, dest_item)
				pbar.update()

def cvreadimg(path: Path):
	img = cv.imread(str(path))
	assert img is not None, "file could not be read, check with os.path.exists()"
	return img

def resize_image(path: Path, width=128, height=128):
	img = cvreadimg(path)
	resized_img = cv.resize(img, (width, height), interpolation=cv.INTER_LANCZOS4)
	return resized_img

def copy_folder_process(src, dst, desc, callback, **kwargs):
	src_path = Path(src)
	dst_path = Path(dst)
	img_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
	# Copy folder structure and process images
	with tqdm(
		total=sum(1 for _ in src_path.rglob('*') if _.suffix.lower() in img_exts),
		desc=desc,
	) as pbar:	
		for item in src_path.rglob('*'):
			rel_path = item.relative_to(src_path)
			dest_item = dst_path / rel_path

			if item.is_dir():
				dest_item.mkdir(parents=True, exist_ok=True)
			elif item.suffix.lower() in img_exts:
				dest_item.parent.mkdir(parents=True, exist_ok=True)
				try:
					processed_img = callback(item, **kwargs)
					cv.imwrite(str(dest_item), processed_img, [int(cv.IMWRITE_JPEG_QUALITY), 95])
				except Exception as e:
					print(f"\nError processing {item}: {e} (falling back to plain copy)")
					shutil.copy2(item, dest_item)
				pbar.update()
			else:
				dest_item.parent.mkdir(parents=True, exist_ok=True)
				shutil.copy2(item, dest_item)

def check_equal(src, dst, width=128, height=128):
	src_path = Path(src)
	dst_path = Path(dst)
	# Get total image count
	img_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
	total = sum(1 for _ in src_path.rglob('*') if _.suffix.lower() in img_exts)
	# Copy folder structure and process images
	equal_count = 0
	with tqdm(
		total=total,
		desc="Checking for equality...",
	) as pbar:	
		for src_item, dst_item in zip(src_path.rglob('*'), dst_path.rglob('*')):
			rel_src_path = src_item.relative_to(src_path)
			rel_dst_path = dst_item.relative_to(dst_path)

			if src_item.suffix.lower() in img_exts:
				if rel_src_path == rel_dst_path:
					equal_count += 1
				pbar.update()
		print(f"({equal_count}/{total} equal)")

# NORMAL STATISTICS FUNCTIONS

def combined_mean(x1,n1,x2,n2):
	return (n1*x1 + n2*x2) / (n1+n2)

def combined_var(v1,x1,n1,v2,x2,n2,x, c=1):
	"""c is the correction term in the variance"""
	return ((n1-c)*v1 + (n2-c)*v2 + n1*(x1-x)**2 + n2*(x2-x)**2) / (n1+n2-c)

class CombinedNormalStats:
	def __init__(self):
		self.mean = 0
		self.var = 0
		self.numelems = 0
	def update(self, instance: Tensor, dim=(1,2)):
		instance_numelems = instance.numel()
		# Mean
		instance_mean = instance.mean(dim=dim)
		_combined_mean = combined_mean(
			self.mean, self.numelems, 
			instance_mean, instance_numelems
		)
		# var
		instance_var = instance.var(dim=dim)
		_combined_var = combined_var(
			self.var, self.mean, self.numelems,
			instance_var, instance_mean, instance_numelems,
			_combined_mean,
		)
		self.numelems += instance_numelems
		self.mean = _combined_mean
		self.var = _combined_var
	def get(self):
		return self.mean, self.var.sqrt()
	
def get_normal_statistics_whole(
		iterable: Iterable,
		dim = (0,2,3),
	):
	all_instances = None
	for instance in tqdm(iterable, desc="Calculating normal statistics..."):
		all_instances = instance.unsqueeze(0) if all_instances == None else torch.cat((all_instances, instance.unsqueeze(0)))
	return all_instances.mean(dim=dim), all_instances.std(dim=dim)

def get_normal_statistics(
		iterable: Iterable,
		dim = (1,2),
	):
	combinedNormalStats = CombinedNormalStats()
	for instance in tqdm(iterable, desc="Calculating normal statistics..."):
		combinedNormalStats.update(instance, dim=dim)
	return combinedNormalStats.get()