import torch
import numpy as np
from data import Load_Blender, Load_LLFF, Load_mip360
from utils import SE3_exp, apply_pca

class Data_set():
    def __init__(self, sys_param):
        self.sys_param = sys_param
        self.pca_mode = self.sys_param['pca_mode']
        # 初始化数据集
        self.data_type = self.sys_param['data_type']
        if self.data_type == 'blender':
            self.data = Load_Blender(self.sys_param)
        elif self.data_type == 'llff':
            self.data = Load_LLFF(self.sys_param)
        elif self.data_type == 'mip360':
            self.data = Load_mip360(self.sys_param)
        # 图像
        self.imgs_train, self.imgs_test, self.imgs_val = self.data.imgs_train, self.data.imgs_test, self.data.imgs_val
        # 数量 索引
        self.train_num, self.test_num, self.val_num = self.data.train_num, self.data.test_num, self.data.val_num
        self.idx_train, self.idx_test, self.idx_val = self.data.idx_train, self.data.idx_test, self.data.idx_val
        self.img_h, self.img_w = self.data.img_h, self.data.img_w
        # 外参
        self.c2ws_pca_train, self.c2ws_pca_test, self.c2ws_pca_val = self.data.c2ws_pca_train, self.data.c2ws_pca_test, self.data.c2ws_pca_val
        self.c2ws_train, self.c2ws_test, self.c2ws_val = self.data.c2ws_train, self.data.c2ws_test, self.data.c2ws_val
        # 外参加初始噪声
        self.transform_mat, self.scale_factor = self.data.transform_mat, self.data.scale_factor
        self.c2ws_train_noise = self.init_c2ws_noise(self.c2ws_train, self.train_num)
        self.c2ws_pca_train_noise = apply_pca(self.c2ws_train_noise, self.transform_mat, self.scale_factor)
        # 内参
        self.intrs_train, self.intrs_test, self.intrs_val = self.data.intrs_train, self.data.intrs_test, self.data.intrs_val
        self.intrs_inv_train, self.intrs_inv_test, self.intrs_inv_val = self.data.intrs_inv_train, self.data.intrs_inv_test, self.data.intrs_inv_val

    def getTrainData(self):
        if self.pca_mode:
            return self.imgs_train, self.c2ws_pca_train, self.c2ws_pca_train_noise, self.intrs_train, self.intrs_inv_train, self.idx_train, self.train_num
        else:
            return self.imgs_train, self.c2ws_train, self.c2ws_train_noise, self.intrs_train, self.intrs_inv_train, self.idx_train, self.train_num
    
    def getTestData(self):
        if self.pca_mode:
            return self.imgs_test, self.c2ws_pca_test, self.intrs_test, self.intrs_inv_test, self.idx_test, self.test_num
        else:
            return self.imgs_test, self.c2ws_test, self.intrs_test, self.intrs_inv_test, self.idx_test, self.test_num
    
    def getValData(self):
        if self.pca_mode:
            return self.imgs_val, self.c2ws_pca_val, self.intrs_val, self.intrs_inv_val, self.idx_val, self.val_num
        else:
            return self.imgs_val, self.c2ws_val, self.intrs_val, self.intrs_inv_val, self.idx_val, self.val_num
        
    def init_c2ws_noise(self, c2ws, num):
        # 初始化se3的噪声
        self.cam_rot_noise = torch.rand(num, 3) * 2 * self.sys_param['rot_noise'] - self.sys_param['rot_noise']
        self.cam_trans_noise = torch.rand(num, 3) * 2 * self.sys_param['trans_noise'] - self.sys_param['trans_noise']
        new_c2ws = []
        for i in range(num):
            tau = torch.cat([self.cam_trans_noise[i], self.cam_rot_noise[i]], axis=0)
            new_c2w = SE3_exp(tau) @ c2ws[i]
            new_c2ws.append(new_c2w)
        new_c2ws = torch.stack(new_c2ws, dim=0)
        return new_c2ws
