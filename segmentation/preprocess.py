from pathlib import Path
from utils import get_normal_statistics
import shutil

import cv2 as cv
from tqdm import tqdm

from utils import (
	cvreadimg,
	resize_image, 
	copy_folder_process, 
	check_equal, 
	get_normal_statistics,
)

from segmentation.utils import OCT5kDataset, IterableOCT5kDataset

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
	preprocess("segmentation/OCTData_pruned")