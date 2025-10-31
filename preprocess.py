#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
from PIL import Image
import random

SOURCE_DIR = "CellData/OCT"
TARGET_DIR_RESIZED = "OCT_resized"
TARGET_DIR_UNDERSAMPLED = "OCT_undersampled"

def print_progress(count, total, end = ""):
	_end = f" {end}" if isinstance(end, str) else ""
	pct = (count / total * 100) if total else 0
	bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
	print(f'\r[{bar}] {pct:.1f}% ({count}/{total})', end=_end, flush=True)

def copy_folder(src, dst):
	src_path = Path(src)
	dst_path = Path(dst)
	# Get total image count
	total = sum(1 for _ in src_path.rglob('*') if not _.is_dir())
	# Copy folder structure and process images
	count = 0
	for item in src_path.rglob('*'):
		rel_path = item.relative_to(src_path)
		dest_item = dst_path / rel_path
		if item.is_dir():
			dest_item.mkdir(parents=True, exist_ok=True)
		else:
			count += 1
			dest_item.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(item, dest_item)
			print_progress(count, total)
	print('\n✓ Done!')

def copy_folder_resize(src, dst, width=128, height=128):
	src_path = Path(src)
	dst_path = Path(dst)
	# Get total image count
	img_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
	total = sum(1 for _ in src_path.rglob('*') if _.suffix.lower() in img_exts)
	# Copy folder structure and process images
	img_count = 0
	for item in src_path.rglob('*'):
		rel_path = item.relative_to(src_path)
		dest_item = dst_path / rel_path

		if item.is_dir():
			dest_item.mkdir(parents=True, exist_ok=True)
		elif item.suffix.lower() in img_exts:
			img_count += 1
			dest_item.parent.mkdir(parents=True, exist_ok=True)
			try:
				img = Image.open(item)
				resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
				resized_img.save(dest_item, quality=95)
			except Exception as e:
				print(f"\nError processing {item}: {e} (falling back to plain copy)")
				shutil.copy2(item, dest_item)
			print_progress(img_count, total)
		else:
			dest_item.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(item, dest_item)

	print('\n✓ Done!')

def check_equal(src, dst, width=128, height=128):
	src_path = Path(src)
	dst_path = Path(dst)
	# Get total image count
	img_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
	total = sum(1 for _ in src_path.rglob('*') if _.suffix.lower() in img_exts)
	# Copy folder structure and process images
	equal_count = 0
	img_count = 0
	for src_item, dst_item in zip(src_path.rglob('*'), dst_path.rglob('*')):
		rel_src_path = src_item.relative_to(src_path)
		rel_dst_path = dst_item.relative_to(dst_path)

		if src_item.suffix.lower() in img_exts:
			img_count += 1
			if rel_src_path == rel_dst_path:
				equal_count += 1
			print_progress(img_count, total, end=f"({equal_count}/{total} equal)")
	print('\n✓ Done!')

def get_undersample_statistics(train_path):
	train_path = Path(train_path)
	class_paths = list(train_path.iterdir())
	class_dist = {class_path.name: len(list(class_path.iterdir())) for class_path in class_paths}
	print("Number of files per class:")
	print(class_dist)
	num_files2del_per_class = {k: v - min(class_dist.values()) for k, v in class_dist.items()}
	print("Number of files to delete per class:")
	print(num_files2del_per_class)
	return class_paths, num_files2del_per_class, sum(num_files2del_per_class.values())

def undersample(train_path):
	train_path = Path(train_path)
	# Get statistics
	class_paths, num_files2del_per_class, total = get_undersample_statistics(train_path)
	# Copy folder structure and process images
	img_count = 0
	for class_path, num_files2del in zip(class_paths, num_files2del_per_class.values()):
		files2del = random.sample(list(class_path.iterdir()), k=num_files2del)
		#print(num_files2del, files2del)
		for file in files2del:
			img_count += 1
			file.unlink()
			print_progress(img_count, total)

	print('\n✓ Done!')
		
	
if __name__ == '__main__':
	# print("Copying resized files")
	# copy_folder_resize(SOURCE_DIR, TARGET_DIR_RESIZED)
	# print("Checking for equality")
	# check_equal(SOURCE_DIR, TARGET_DIR_RESIZED)
	
	# print("Backing up resized files")
	# copy_folder(TARGET_DIR_RESIZED, TARGET_DIR_UNDERSAMPLED)
	# print("Checking for equality")
	# check_equal(TARGET_DIR_RESIZED, TARGET_DIR_UNDERSAMPLED)
	print("Undersampling")
	undersample(TARGET_DIR_UNDERSAMPLED / Path("train"))
