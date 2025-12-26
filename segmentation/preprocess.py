from typing import Iterable, Union
from pathlib import Path
import shutil
from collections import defaultdict
import random

import cv2 as cv
from tqdm import tqdm

from core_utils import (
	cvreadimg,
	resize_image, 
	copy_folder,
	copy_folder_process, 
	check_equal, 
	get_normal_statistics,
	ClassPopulation,
)

from segmentation.utils import OCT5kDataset, IterableOCT5kDataset

def all_images(dir_path: Path):
	img_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
	instances = [p for p in dir_path.rglob('*') if p.suffix.lower() in img_exts]
	total = len(instances)
	return instances, total

def random_split(directory, names: Iterable[str], ps: Iterable[Union[int, float]]):
	dir_path = Path(directory)
	# Copy folder structure and process images
	instances, total = all_images(dir_path)
	img_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
	with tqdm(
		total=total,
		desc="Splitting into subsets...",
	) as pbar:
		for name, p in zip(names, ps):
			count = None
			if isinstance(p, int):
				count = int(p)
			elif isinstance(p, float) and p < 1:
				count = int(p*total)
			else:
				raise ValueError(f"p has invalid value {p}")
			files2mv = random.sample(instances, k=count)
			for src_mask in files2mv:
				if (
					"train" in src_mask.parts or 
					"test" in src_mask.parts or 
					"val" in src_mask.parts 
				):
					continue
				rel_path_mask = src_mask.relative_to(dir_path)

				rel_path = Path(*rel_path_mask.parts[2:])
				if rel_path.parts[0].startswith("Grading"):
					rel_path = Path(*rel_path.parts[1:])

				rel_path_img = "Images/Images_Original" / rel_path
				src_img = dir_path / rel_path_img
				dst_mask = dir_path / name / rel_path_mask
				dst_img = dir_path / name / rel_path_img
				dst_mask.parent.mkdir(parents=True, exist_ok=True)
				src_mask.rename(dst_mask)
				if not dst_img.exists():
					dst_img.parent.mkdir(parents=True, exist_ok=True)
					for img_ex in img_exts:
						src_img_suffix = src_img.with_suffix(img_ex.upper())
						if src_img_suffix.exists():
							src_img_suffix.rename(dst_img.with_suffix(img_ex.upper()))
				instances.remove(src_mask)
				pbar.update()

def get_class_dist(dir_path: Path):
	img_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
	class_dist = defaultdict(int)
	for item in dir_path.rglob('*'):
		desc = item.parent.parent.name
		if item.suffix.lower() in img_exts and desc.endswith('.E2E'):
			class_name = desc.split()[0]
			class_dist[class_name] += 1
	print(f"Number of files in {dir_path.name} set per class: {dict(class_dist)}")
	return class_dist


def pad_image(path: Path, target_size=(512, 512)):
	img = cvreadimg(path)
	h, w = img.shape[:2]
	tw, th = target_size
	delta_w = max(0, tw - w)
	delta_h = max(0, th - h)
	top, bottom = delta_h // 2, delta_h - (delta_h // 2)
	left, right = delta_w // 2, delta_w - (delta_w // 2)
	color = [0, 0, 0]
	new_img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
	return new_img

def preprocess(
	src_dir, 
	target_dir=None, 
	pad_flag=False, 
	resize_flag=False,
	preprocess_flag=False,
	split_flag=False,
	calc_normal = False, 
):
	src_path = Path(src_dir).resolve()

	if target_dir is None:
		target_dir = src_dir
	
	target_dir_padded = f'{target_dir}_padded' 
	target_path_padded = Path(target_dir_padded).resolve() 

	if pad_flag:
		copy_folder_process(src_path, target_path_padded, "Copy-padding...", pad_image)
		check_equal(src_path, target_path_padded)
	
	target_dir_resized = f'{target_dir}_resized' 
	target_path_resized = Path(target_dir_resized).resolve() 
	if resize_flag:
		copy_folder_process(src_path, target_path_resized, "Copy-resizing...", resize_image)
		check_equal(src_path, target_path_resized)

	
	target_dir_preprocessed = f'{target_dir}_preprocessed' 
	target_path_preprocessed = Path(target_dir_preprocessed).resolve() 
	if preprocess_flag:
		copy_folder(src_path, target_path_preprocessed, "Backing up files...")
	
	if split_flag:
		_, total = all_images(target_path_preprocessed)
		test_size = 1000
		remainder = total - test_size
		val_size = int(remainder*0.2)
		train_size = remainder - val_size
		random_split(
			target_path_preprocessed, 
			names = ["train", "val", "test"], 
			ps = [train_size, val_size, test_size],
		)
	if calc_normal:
		mean, std = get_normal_statistics(
			IterableOCT5kDataset(OCT5kDataset(target_path_padded))
		)
		print("After padding:")
		print(mean, std)
		mean, std = get_normal_statistics(
			IterableOCT5kDataset(OCT5kDataset(target_path_resized))
		)
		print("After resizing:")
		print(mean, std)
	
if __name__ == '__main__':
	get_class_dist(Path("segmentation/OCTData_pruned/Masks/Masks_Manual/Grading_1"))
	preprocess("segmentation/OCTData_pruned", preprocess_flag=True, split_flag=True)