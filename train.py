import torch
import setproctitle
import argparse
import os
import re
from tqdm import tqdm
from random import randint
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from config import Load_Config
from data import Data_set
from model import NeRF_Model
from model import RAdam
from utils import save_img_cv2, save_depth_cv2, psnr_score, ssim_score, lpips_score
from utils import show_camera_position, init_show_figure, show_rays

class Model_Trainer():
    def __init__(self, sys_param):
        self.sys_param = sys_param
        self.device = self.sys_param['device']
        self.tb_mode = self.sys_param['tb_mode']
        self.data_name = self.sys_param['data_name']
        self.batch_size = self.sys_param['batch_size']
        self.num_steps = self.sys_param['num_steps']
        self.save_steps = self.sys_param['save_steps']
        self.val_steps = self.sys_param['val_steps']
        self.rays_visul = self.sys_param['rays_visul']
        self.rays_step = self.sys_param['rays_step']
        self.output_path = self.sys_param['output_path']
        self.lrate = self.sys_param['learning_rate']
        self.lrate_decay = self.sys_param['learning_rate_decay']
        self.weight_decay = self.sys_param['weight_decay']
        # 加载数据集
        self.dataset = Data_set(self.sys_param)
        # 初始化模型
        self.nerf = NeRF_Model(self.sys_param).to(self.device)
        self.only_coarse = self.sys_param['only_coarse']
        self.loss_l2 = torch.nn.MSELoss(reduction='mean')
        # 优化器
        # self.optimizer = torch.optim.Adam(params=self.nerf.parameters(), lr=self.lrate, betas=(0.9, 0.999))
        self.optimizer = RAdam(filter(lambda p: p.requires_grad, self.nerf.parameters()), lr=self.lrate, eps=1e-8, weight_decay=self.weight_decay)
        # 初始化Tensorboard
        if self.tb_mode:
            self.tb_path = os.path.join(self.output_path, "tensorboard")
            os.makedirs(self.tb_path, exist_ok=True)
            self.tblogger = SummaryWriter(log_dir=self.tb_path)
        
    def forward(self):
        # 获取训练数据
        imgs_train, c2ws_train, intrs_train, intrs_inv_train, idx_train, num_train = self.dataset.getTrainData()
        self.rays_root = os.path.join(self.output_path, "rays")
        os.makedirs(self.rays_root, exist_ok=True)
        # 随机选择视角
        indices_rand = list(idx_train)
        with tqdm(total=self.num_steps, desc="Training:", ncols=150) as pbar:
            for step in range(self.num_steps):
                ### 随机选择一个训练视角（相机位置）
                if not indices_rand:
                    indices_rand = list(idx_train)
                rand_idx = randint(0, len(indices_rand) - 1)
                idx = indices_rand.pop(rand_idx)
                ### 获取当前视角的图像、相机位姿和逆内参矩阵
                img, c2w, intr_inv, intr = imgs_train[idx].to(self.device), c2ws_train[idx].to(self.device), intrs_inv_train[idx].to(self.device), intrs_train[idx].to(self.device) # [3, H, W], [4, 4], [3, 3]
                _, H, W = img.shape
                ### 生成光线原点和方向
                rays_o, rays_d = self.nerf.get_rays(H, W, intr_inv.unsqueeze(0), c2w.unsqueeze(0))      # [1, H*W, 3], [1, H*W, 3]
                rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)                           # [H*W, 3]
                ### 随机选择一批光线进行训练
                indices_train = torch.randperm(rays_o.shape[0])[:self.batch_size]
                select_rays_o = rays_o[indices_train]                                                   # [batch_size, 3]
                select_rays_d = rays_d[indices_train]                                                   # [batch_size, 3]
                ### 进入NeRF模型进行体渲染
                data = self.nerf(select_rays_o, select_rays_d)
                ### 计算渲染损失
                img = img.permute(1, 2, 0).reshape(-1, 3)                                               # [H*W, 3]
                gt_rgbs = img[indices_train]                                                            # [batch_size, 3]
                if self.only_coarse:
                    rgbs_c = data
                    loss = self.loss_l2(rgbs_c, gt_rgbs)
                else:
                    rgbs_c, rgbs_f = data
                    loss_c = self.loss_l2(rgbs_c, gt_rgbs)
                    loss_f = self.loss_l2(rgbs_f, gt_rgbs)
                    loss = loss_c + loss_f
                ### 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                ### 更新学习率
                decay_rate = 0.1
                decay_steps = self.lrate_decay * 1000
                new_lrate = self.lrate * (decay_rate ** (step / decay_steps))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lrate
                ### 更新进度条
                pbar.set_postfix({"loss": f"{loss.item():.8f}"})
                pbar.set_postfix({"lr": f"{self.optimizer.param_groups[0]['lr']:.8f}"})  
                ### 可视化光线
                if self.rays_visul and step % self.rays_step == 0:
                    indices_rays = torch.randperm(rays_o.shape[0])[:100]
                    visul_rays_o = rays_o[indices_rays]                                                   # [batch_size, 3]
                    visul_rays_d = rays_d[indices_rays]                                                   # [batch_size, 3]
                    rays_path = os.path.join(self.rays_root, "step_{:06d}.png".format(step))
                    self.show_camera_rays(rays_path, intr.unsqueeze(0), c2w.unsqueeze(0), visul_rays_o.cpu(), visul_rays_d.cpu(), W)
                ### 保存tensorboard
                if self.tb_mode:
                    self.tblogger.add_scalar("Loss/train", loss, step)
                ### 保存模型检查点
                if step % self.save_steps == 0 or step == self.num_steps-1:
                    self.nerf.save_model(step)
                ### 验证模型性能
                if step % self.val_steps == 0:
                    self.valid(step)
                    self.nerf.train()
                pbar.update()   

    # 可视化验证集
    def valid(self, step):
        print("Valid: ")
        self.nerf.eval()
        with torch.no_grad():
            # 获取验证集
            imgs_val, c2ws_pca_val, intrs_val, intrs_inv_val, idx_val, num_val = self.dataset.getValData()
            # 随机选取索引
            # rand_idx_val = randint(0, num_val - 1)
            rand_idx_val = 10
            img_val, c2w_val, intr_inv_val = imgs_val[rand_idx_val].to(self.device), c2ws_pca_val[rand_idx_val].to(self.device), intrs_inv_val[rand_idx_val].to(self.device)
            _, H, W = img_val.shape                                                                             # img_val: [3, H ,W]  c2w_val: [4, 4]  intr_inv_val: [3, 3]
            rays_o_val, rays_d_val = self.nerf.get_rays(H, W, intr_inv_val.unsqueeze(0), c2w_val.unsqueeze(0))  # [1, H*W, 3] [1, H*W, 3]
            rays_o_val, rays_d_val = rays_o_val.reshape(-1, 3), rays_d_val.reshape(-1, 3)                       # [H*W, 3]
            num_rays = len(rays_d_val)
            rgbs = []
            depths = []
            for i in range(int(num_rays/self.batch_size)+1):
                if (i+1)*self.batch_size > num_rays:
                    rays_o, rays_d = rays_o_val[i*self.batch_size:num_rays], rays_d_val[i*self.batch_size:num_rays]
                else:
                    rays_o, rays_d = rays_o_val[i*self.batch_size:(i+1)*self.batch_size], rays_d_val[i*self.batch_size:(i+1)*self.batch_size]
                data = self.nerf(rays_o, rays_d, mode="test")
                if self.only_coarse:
                    rgb, depth = data                                                                           # [batch, 3] [batch]
                else:
                    _, _, rgb, depth = data                                                                     # [batch, 3] [batch]
                rgbs.append(rgb)
                depths.append(depth)
            rgbs = torch.cat(rgbs, dim=0)                                                                       # [H*W, 3] 
            depths = torch.cat(depths, dim=0)                                                                   # [H*W]        
            # 可视化
            train_root = os.path.join(self.output_path, "train")
            os.makedirs(train_root, exist_ok=True)
            pd_path = os.path.join(train_root, "step_{}_pd.png".format(step))
            depth_path = os.path.join(train_root, "step_{}_depth.png".format(step))
            gt_path = os.path.join(train_root, "step_{}_gt.png".format(step))
            pd_rgb_tensor = rgbs.reshape(H, W, -1).permute(2, 0, 1)                                             # [3, H, W]
            gt_rgb_tensor = img_val                                                                             # [3, H, W]
            depth_tensor = depths.reshape(H, W)                                                                 # [H, W]
            save_img_cv2(pd_rgb_tensor, pd_path)
            save_img_cv2(gt_rgb_tensor, gt_path)
            save_depth_cv2(depth_tensor, depth_path)
            # 计算评价参数
            ssim_scores = ssim_score(pd_rgb_tensor, gt_rgb_tensor)
            psnr_scores = psnr_score(pd_rgb_tensor, gt_rgb_tensor)
            lpips_scores = lpips_score(pd_rgb_tensor, gt_rgb_tensor)
            print("PSNR: {:.6f}".format(psnr_scores))
            print("SSIM: {:.6f}".format(ssim_scores))
            print("LPIPS: {:.6f}".format(lpips_scores))
    
    # 可视化相机函数
    def show_camera_rays(self, save_path, intrs_train, c2ws_train, rays_o, rays_d, img_w):
        draw_ax = init_show_figure(show_info=False)
        show_camera_position(intrs_train,
                            c2ws_train,
                            img_w, 
                            draw_ax, 
                            color = (0.0,0.6,0.7), 
                            cam_size=0.1, 
                            line_w=0.5)
        show_rays(rays_o, rays_d, draw_ax, line_w=0.5)
        plt.savefig(save_path, dpi=300)                 # 设置dpi来控制图片清晰度，可以根据需要调整
        plt.close()                                     # 关闭绘图，避免影响下一次绘图
        # plt.show()
        # plt.pause(0)
      
if __name__ == "__main__":
    setproctitle.setproctitle("nerf-slt")
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
    model = Model_Trainer(sys_param)
    model.forward()
        