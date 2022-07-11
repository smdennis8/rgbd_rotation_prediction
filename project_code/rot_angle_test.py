# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 21:38:46 2021

@author: sethd
"""
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
def compute_angle_from_r_matrices(m):
    
    batch=m.shape[0]
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.ones(batch,dtype=torch.float,device=cos.device))
    cos = torch.max(cos, torch.ones(batch,dtype=torch.float,device=cos.device)*-1 )
    
    theta = torch.acos(cos)
    
    return theta


def main():
    
    m1 = R.from_euler('x', 5, degrees=True).as_matrix()
    rot = R.from_euler('x', 20, degrees=True).as_matrix()
    m2 = m1 @ rot
    print(m1)
    print(m2)
    
    m = np.stack((m1,m2),axis=0)
    
    print(m.shape)
    
    m = torch.from_numpy(m.astype(np.float32))
    degRet = compute_angle_from_r_matrices(m)
    degRet = degRet*180
    degRet = degRet/np.pi
    print(degRet)
    
    
if __name__ == "__main__":    
    main()