import torch
import os
import re
import setproctitle
import argparse
from tqdm import tqdm
from config import Load_Config
from data import Data_set
from model import NeRF_Model
from utils import save_img_cv2, ssim_score, psnr_score, lpips_score

class Model_Tester():
    def __init__(self, sys_param):
        self.sys_param = sys_param
        self.device = self.sys_param['device']
        self.data_name = self.sys_param['data_name']
        self.batch_size = self.sys_param['batch_size']
        self.output_path = self.sys_param['output_path']
        self.test_root = os.path.join(self.output_path, "test")
        os.makedirs(self.test_root, exist_ok=True)
        # 加载数据集
        self.dataset = Data_set(self.sys_param)
        # 初始化模型
        self.nerf = NeRF_Model(self.sys_param, load_weights=True).to(self.device)
        self.nerf.eval()
    
    def forward(self):
        # 获取测试数据集
        imgs_test, c2ws_pca_test, intrs_test, intrs_inv_test, idx_test, num_test = self.dataset.getTestData()
        # 读取模型权重
        _, _, H, W = imgs_test.shape
        rays_o_test, rays_d_test = self.nerf.get_rays(H, W, intrs_inv_test, c2ws_pca_test)              # [B, H*W, 3]
        with torch.no_grad():
            psnr_scores, ssim_scores, lpips_scores = 0.0, 0.0, 0.0
            with tqdm(total=num_test, desc="Testing:", ncols=150) as pbar:
                for i_test in range(num_test):
                    rays_o, rays_d = rays_o_test[i_test], rays_d_test[i_test]                               # [H*W, 3]
                    rgbs = []
                    for i in range(int(len(rays_d)/self.batch_size)+1):
                        if (i+1)*self.batch_size > len(rays_d):
                            select_rays_o = rays_o[i*self.batch_size:len(rays_d)].to(self.device)           # [batch_size, 3]
                            select_rays_d = rays_d[i*self.batch_size:len(rays_d)].to(self.device)           # [batch_size, 3]
                        else:
                            select_rays_o = rays_o[i*self.batch_size:(i+1)*self.batch_size].to(self.device) # [batch_size, 3]
                            select_rays_d = rays_d[i*self.batch_size:(i+1)*self.batch_size].to(self.device) # [batch_size, 3]
                        # 进入nerf模型
                        rgb, _ = self.nerf(select_rays_o, select_rays_d, mode="test")
                        rgbs.append(rgb)
                    rgbs = torch.cat(rgbs, dim=0)                                           # [H*W, 3]
                    test_path = os.path.join(self.test_root, "{:03d}.png".format(i_test))
                    pd_rgb = rgbs.reshape(H, W, -1).permute(2, 0, 1)                        # [3, H, W]
                    gt_rgb = imgs_test[i_test].to(self.device)                              # [3, H, W]
                    save_img_cv2(pd_rgb, test_path)
                    psnr_i = psnr_score(pd_rgb, gt_rgb)
                    ssim_i = ssim_score(pd_rgb, gt_rgb)
                    lpips_i = lpips_score(pd_rgb, gt_rgb)
                    psnr_scores += psnr_i
                    ssim_scores += ssim_i
                    lpips_scores += lpips_i
                    pbar.update(1)
        print("Parameters: ")
        print("PSNR: {:.6f}".format(psnr_scores/num_test))
        print("SSIM: {:.6f}".format(ssim_scores/num_test))
        print("LPIPS: {:.6f}".format(lpips_scores/num_test))
    
    

if __name__ == "__main__":
    setproctitle.setproctitle("nerf-test")
    ##### 加载配置信息 #####
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument("--config", type=str, default="config/config.yaml", help="the path of config")
    # 从命令行中获取参数
    args = parser.parse_args()
    
    # 从配置文件中获取参数并整合命令行参数
    config = Load_Config(args)
    sys_param = config.sys_param    
    model = Model_Tester(sys_param)
    model.forward()