import os
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

import image
import math
import cv2


class RotationDataset(Dataset):
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
        self.img_dir = dataset_dir + 'rgb/'
        self.depth_dir = dataset_dir + 'depth/'
        
        #WORKAROUND
        self.mask_dir = dataset_dir + 'gt/'
        
        #GT in this case is CSV file with rotation strings as [ 9 float elements ]
        if has_gt:
            self.gt_dir = dataset_dir + 'rotations/'  #has a CSV file of all the rotations in order
        

        self.imgFilesBefore = [item for item in os.listdir(self.img_dir) if (item.endswith('.png') and (item.find('before') is not -1))]
        self.imgFilesAfter = [item for item in os.listdir(self.img_dir) if (item.endswith('.png') and (item.find('after') is not -1))]
        self.imgFilesBefore.sort(key=lambda item:int(item.split("_")[0]))
        self.imgFilesAfter.sort(key=lambda item:int(item.split("_")[0]))
        
        self.depthFilesBefore = [item for item in os.listdir(self.depth_dir) if (item.endswith('.png') and (item.find('before') is not -1))]
        self.depthFilesAfter = [item for item in os.listdir(self.depth_dir) if (item.endswith('.png') and (item.find('after') is not -1))]
        self.depthFilesBefore.sort(key=lambda item:int(item.split("_")[0]))
        self.depthFilesAfter.sort(key=lambda item:int(item.split("_")[0]))
        
        self.maskFilesBefore = [item for item in os.listdir(self.mask_dir) if (item.endswith('.png') and (item.find('before') is not -1))]
        self.maskFilesAfter = [item for item in os.listdir(self.mask_dir) if (item.endswith('.png') and (item.find('after') is not -1))]
        self.maskFilesBefore.sort(key=lambda item:int(item.split("_")[0]))
        self.maskFilesAfter.sort(key=lambda item:int(item.split("_")[0]))
        
        #compose, transforms.ToTensor() and transforms.Normalize() for RGB image.
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean_rgb, std_rgb)])
        #sources: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        
        self.gtRotations = []
        self.rot_file = dataset_dir+"/rotations/rotations.csv"
       
        if has_gt:
             #get gt 
             with open(self.rot_file) as f:
                 rotStrings = f.readlines()
            
             for rot in rotStrings:
                 npLine = np.fromstring(rot, dtype=np.float32, sep=' ')
                 self.gtRotations.append(torch.as_tensor(npLine))
            
             
        
        # number of samples in the dataset, not hard coded number
        img_dir_len = len(self.imgFilesAfter)
        self.dataset_length = img_dir_len
        

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        """
        In:
            idx: int, index of each sample, in range(0, dataset_length).
        Out:
            sample: a dictionary that stores paired rgb/depth images and corresponding ground truth rotation matrix (if available).
                    rgb_img: Tensor [8, height, width]
                    target: Tensor [9], use torch.LongTensor() to convert.
        Purpose:
            Given an index, return paired rgbs/depths and ground truth matrix as a sample.
       
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #load mask, image, depth, tensor
        self.img_dir = Path(self.img_dir)
        self.depth_dir = Path(self.depth_dir)
        self.mask_dir = Path(self.mask_dir)
        
        rgbBeforeFileName = self.imgFilesBefore[idx]
        rgb_before = cv2.imread(str(self.img_dir/rgbBeforeFileName))
        rgb_before =  rgb_before.astype(np.float32)/255.0
        rgb_before =  torch.from_numpy(rgb_before.transpose((2,0,1)))
        #rgb_before = self.transform(rgb_before.copy())
        
        rgbAfterFileName = self.imgFilesAfter[idx]
        rgb_after = cv2.imread(str(self.img_dir/rgbAfterFileName))
        rgb_after =  rgb_after.astype(np.float32)/255.0
        rgb_after =  torch.from_numpy(rgb_after.transpose((2,0,1)))
        # rgb_after = self.transform(rgb_after.copy())
        
        depthBeforeName = self.depthFilesBefore[idx]
        depth_before = image.read_depth(str(self.depth_dir/depthBeforeName))
        depth_before = torch.from_numpy(depth_before).unsqueeze(0)
        
        depthAfterName = self.depthFilesAfter[idx]
        depth_after = image.read_depth(str(self.depth_dir/depthAfterName))
        depth_after = torch.from_numpy(depth_after).unsqueeze(0)
        
        #maskBeforeName = self.maskFilesBefore[idx]
        #mask_before = image.read_mask(self.mask_dir/maskBeforeName)
        
        #maskAfterName = self.maskFilesAfter[idx]
        #mask_after = image.read_mask(self.mask_dir/maskAfterName)
        
        #make library
        img_block = torch.tensor(np.concatenate((rgb_before,depth_before,rgb_after,depth_after)))
        img_block = img_block.float()
        
        gt_matrix = None
        
        if self.has_gt:
            gt_matrix = self.gtRotations[idx]
            #gt_matrix = torch.LongTensor(gt_matrix)
        
        if self.has_gt is False:
            sample = {'input': img_block}
        else:
            sample = {'input': img_block, 'target': gt_matrix}
        
        return sample

def main():
    root_dir = './dataset/'
    train_dir = root_dir + 'train1/'
    train_dataset = RotationDataset(train_dir, True )
    
    a = train_dataset.__getitem__(12)
    print(a['input'].shape)
    for idx,batch in enumerate(train_dataset):
        if idx >  2:
            break
        inps = batch['input']
        rgb_before = (inps[:3,:,:].numpy().transpose((1,2,0))*255.0).astype(np.uint8)
        cv2.imwrite(f"rgb_before {idx}.png",cv2.cvtColor(rgb_before, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    main()