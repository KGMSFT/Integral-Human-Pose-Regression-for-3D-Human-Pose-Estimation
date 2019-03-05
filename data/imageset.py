from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import cv2
import numpy as np
class ImageSet(Dataset):
    def __init__(self, abs_path, cfg, transforms):
        self.root_dir = abs_path
        self.transforms = transforms
        self.input_shape = cfg.input_shape
    def __len__(self):
        count = 0
        for fn in os.listdir(self.root_dir):
            count = count + 1
        return count

    def __getitem__(self, idx):
        img_names = os.listdir(self.root_dir)
        img_name = img_names[idx]
        img = cv2.imread(os.path.join(self.root_dir, img_names[idx]), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # img = cv2.imread(os.path.join(self.root_dir, img_names[idx]))
        img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_CUBIC)
        #cv2.imwrite(img_names[idx], img)
        img = img[:,:,::-1].copy()
        img = img.astype(np.float32)
        if self.transforms:
            img = self.transforms(img)
        
        return img, img_name
