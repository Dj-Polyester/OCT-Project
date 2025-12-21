from utils import get_normal_statistics


from segmentation.utils import IterableOCT5kDataset

if __name__ == '__main__':
    mean, std = get_normal_statistics(IterableOCT5kDataset("segmentation/OCTData"))
    print(mean, std)