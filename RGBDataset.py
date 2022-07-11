import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import image


class RGBDataset(Dataset):
    def __init__(self, dataset_dir, has_gt):
        """
        In:
            img_dir: string, path of train, val or test folder.
            has_gt: bool, indicating if the dataset has ground truth masks.
        Out:
            None.
        Purpose:
            Initialize instance variables.
        Hint:
            Check __getitem__() and add more instance variables to initialize what you need in this method.
        """
        # Input normalization info to be used in transforms.Normalize()
        mean_rgb = [0.722, 0.751, 0.807]
        std_rgb = [0.171, 0.179, 0.197]
        
        self.dataset_dir = dataset_dir
        self.has_gt = has_gt
        self.img_dir = dataset_dir + '/rgb/'
        if has_gt:
            self.gt_dir = dataset_dir + '/gt/'
        
        #compose, transforms.ToTensor() and transforms.Normalize() for RGB image.
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean_rgb, std_rgb)])
        #sources: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        
        #get sample files
        self.imgFiles = [item for item in os.listdir(self.img_dir) if item.endswith('.png')]
        
        #sort list on first digits
        self.imgFiles.sort(key=lambda item:int(item.split("_")[0]))
        
        self.gtFiles = None
        
        if has_gt:
             #get gt masks
            self.gtFiles = [item for item in os.listdir(self.gt_dir) if item.endswith('.png')]
            
            #sort list on first digits
            self.gtFiles.sort(key=lambda item:int(item.split("_")[0]))
        
        # number of samples in the dataset, not hard coded number
        img_dir_len = len([item for item in os.listdir(self.img_dir) if item.endswith('.png')])
        self.dataset_length = img_dir_len

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        """
        In:
            idx: int, index of each sample, in range(0, dataset_length).
        Out:
            sample: a dictionary that stores paired rgb image and corresponding ground truth mask (if available).
                    rgb_img: Tensor [3, height, width]
                    target: Tensor [height, width], use torch.LongTensor() to convert.
        Purpose:
            Given an index, return paired rgb image and ground truth mask as a sample.
        Hint:
            Use image.read_rgb() and image.read_mask() to read the images.
            Think about how to associate idx with the file name of images.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #get file names for opening
        gtMaskName = None
        gt_mask = None
        
        imgFileName = self.imgFiles[idx]
        
        # TODO: read RGB image and ground truth mask, apply the transformation, and pair them as a sample.
        rgb_img = image.read_rgb(self.img_dir + '/' + imgFileName)
        rgb_img = self.transform(rgb_img.copy())
               
        if self.has_gt:
            gtMaskName = self.gtFiles[idx]
            gt_mask = image.read_mask(self.gt_dir + '/' + gtMaskName)
            gt_mask = torch.LongTensor(gt_mask)
        
        if self.has_gt is False:
            sample = {'input': rgb_img}
        else:
            sample = {'input': rgb_img, 'target': gt_mask}
        return sample
