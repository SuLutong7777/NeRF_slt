import torch
import os
import re
from model import CoarseFine_Net, SinCosEmbedding

class NeRF_Model(torch.nn.Module):
    def __init__(self, sys_param, load_weights=False):
        super(NeRF_Model, self).__init__()
        self.sys_param = sys_param
        self.device = self.sys_param['device']
        self.data_name = self.sys_param['data_name']
        self.use_view_dirs = sys_param['use_view_dirs']
        self.freqs_xyz = self.sys_param['freqs_xyz']
        self.freqs_view = self.sys_param['freqs_view']
        self.near = self.sys_param['near']
        self.far = self.sys_param['far']
        self.num_coarse_samples = self.sys_param['num_coarse_samples']
        self.fine_samples_scale = self.sys_param['fine_samples_scale']
        self.num_fine_samples = self.num_coarse_samples * self.fine_samples_scale
        self.white_background = self.sys_param['white_background']
        self.only_coarse = self.sys_param['only_coarse']
        self.fine_weight_threshold = self.sys_param['fine_weight_threshold']
        self.sigmas_default = self.sys_param['sigmas_default']
        self.output_path = self.sys_param['output_path']
        self.weights_path = os.path.join(self.output_path, "weights")
        os.makedirs(self.weights_path, exist_ok=True)
        self.embedding_xyz = SinCosEmbedding(self.sys_param, self.freqs_xyz)
        self.embedding_view = SinCosEmbedding(self.sys_param, self.freqs_view)
        self.coarse_net = CoarseFine_Net(self.sys_param, MLP_type='Coarse')
        self.fine_net = CoarseFine_Net(self.sys_param, MLP_type='Fine')
        # 初始化采样点参数
        self.z_samples_c = torch.linspace(self.near, self.far, self.num_coarse_samples, device=self.device)  # [num_coarse_samples]
        self.z_samples_f = torch.linspace(self.near, self.far, self.num_fine_samples, device=self.device)    # [num_fine_samples]
        if load_weights:
            self.load_weights_path = self.sys_param['test_model_path']
            self.nerf_ckpt_file = torch.load(self.load_weights_path, map_location = self.device)
            self.load_ckpt(self.nerf_ckpt_file, self.coarse_net, coarse=True)
            self.load_ckpt(self.nerf_ckpt_file, self.fine_net, coarse=False)
            print("Loading weights: {}".format(self.load_weights_path))

    def forward(self, rays_o, rays_d, mode="train"):
        if mode == "train":
            data = self.train_model(rays_o, rays_d)
        elif mode == "test":
            data = self.test_model(rays_o, rays_d)
        return data
    
    def train_model(self, rays_o, rays_d):
        batch_size = rays_d.shape[0]
        ##### 粗渲染 #####
        # 添加噪声
        z_vals_c = self.z_samples_c.clone().expand(batch_size, -1)                                          # [batch_size, num_c]
        z_vals_c_noise = torch.empty(batch_size, 1, device=self.device).uniform_(0.0, (self.far - self.near)/self.num_coarse_samples)
        z_vals_c = z_vals_c + z_vals_c_noise                                                                # [batch_size, num_c]
        # 计算采样点位置
        xyz_c = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_c.unsqueeze(2)                           # [batch_size, num_c, 3]
        # 进入mlp网络并进行体积渲染 rgbs_c: [B, 3] depth_c: [B]  opacity_c: [B]
        rgbs_c, depths_c, opacitys_c, weights_c, sigmas_c, deltas_c = self.render_pts(xyz_c, z_vals_c, rays_d, self.coarse_net, mode="train")
        if self.only_coarse:
            return rgbs_c
        ##### 细渲染 #####
        # 依据粗渲染结果进行重要性采样
        weights_c = weights_c.detach()                                                                      # [batch_size, num_c]
        idx_fine = torch.nonzero(weights_c >= min(self.fine_weight_threshold, weights_c.max().item()))      # [idx_num, 2]
        idx_fine = idx_fine.unsqueeze(1).expand(-1, self.fine_samples_scale, -1)                            # [idx_num, fine_sacle, 2]
        # 扩展索引
        idx_fine2 = idx_fine.clone()
        idx_fine2[..., 1] = idx_fine[..., 1] * self.fine_samples_scale + torch.arange(self.fine_samples_scale, dtype=torch.float32, device=self.device).reshape(1, self.fine_samples_scale)
        idx_fine2 = idx_fine2.reshape(-1, 2)                                                                # [idx_num*fine_scale, 2]
        # 采样点过多则随机选择
        if idx_fine2.shape[0] > batch_size * self.num_coarse_samples:
            perm = torch.randperm(idx_fine.shape[0])[:batch_size * self.num_coarse_samples]
            idx_fine2 = idx_fine2[perm]
        # 获取细采样点z_vals
        z_vals_f = self.z_samples_f.clone().expand(batch_size, -1)                                          # [batch_size, num_f]
        z_vals_f_noise = torch.empty(batch_size, 1, device=self.device).uniform_(0.0, (self.far - self.near)/self.num_fine_samples)
        z_vals_f = z_vals_f + z_vals_f_noise                                                                # [batch_size, num_f]
        xyz_f = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_f.unsqueeze(2)                           # [batch_size, num_f, 3]
        # 进入mlp网络并进行体积渲染 rgbs_f: [B, 3] depth_f: [B]  opacity_f: [B]
        rgbs_f, depths_f, opacitys_f, weights_f, sigmas_f, deltas_f = self.render_pts(xyz_f, z_vals_f, rays_d, self.fine_net, idx_fine2)
        return rgbs_c, rgbs_f
    
    def test_model(self, rays_o, rays_d):
        batch_size = rays_d.shape[0]
        ##### 粗渲染 #####
        z_vals_c = self.z_samples_c.clone().expand(batch_size, -1)                                          # [batch_size, num_c]
        xyz_c = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_c.unsqueeze(2)                           # [batch_size, num_c, 3]
        # 进入mlp网络并进行体积渲染 rgbs_c: [B, 3] depth_c: [B]  opacity_c: [B]
        rgbs_c, depths_c, opacitys_c, weights_c, sigmas_c, deltas_c = self.render_pts(xyz_c, z_vals_c, rays_d, self.coarse_net, mode="test")
        # 仅粗渲染
        if self.only_coarse:
            return rgbs_c, depths_c
        ##### 细渲染 #####
        # 依据粗渲染结果进行重要性采样
        weights_c = weights_c.detach()                                                                      # [batch_size, num_c]
        idx_fine = torch.nonzero(weights_c >= min(self.fine_weight_threshold, weights_c.max().item()))      # [idx_num, 2]
        idx_fine = idx_fine.unsqueeze(1).expand(-1, self.fine_samples_scale, -1)                            # [idx_num, fine_sacle, 2]
        # 扩展索引
        idx_fine2 = idx_fine.clone()
        idx_fine2[..., 1] = idx_fine[..., 1] * self.fine_samples_scale + torch.arange(self.fine_samples_scale, dtype=torch.float32, device=self.device).reshape(1, self.fine_samples_scale)
        idx_fine2 = idx_fine2.reshape(-1, 2)                                                                # [idx_num*fine_scale, 2]
        # 获取细采样点z_vals
        z_vals_f = self.z_samples_f.clone().expand(batch_size, -1)                                          # [batch_size, num_f]
        xyz_f = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_f.unsqueeze(2)                           # [batch_size, num_f, 3]
        # 进入mlp网络并进行体积渲染 rgbs_f: [B, 3] depth_f: [B]  opacity_f: [B]
        rgbs_f, depths_f, opacitys_f, weights_f, sigmas_f, deltas_f = self.render_pts(xyz_f, z_vals_f, rays_d, self.fine_net, idx_fine2, mode="test")
        return rgbs_c, depths_c, rgbs_f, depths_f
    
    ########### intrs_inv: [B, 3, 3] c2ws: [B, 4, 4] ###########
    def get_rays(self, img_h, img_w, intrs_inv, c2ws):
        # 生成像素坐标网格
        with torch.no_grad():
            x_range = torch.arange(0, img_w, dtype=torch.float32, device=intrs_inv.device).add_(0.5)
            y_range = torch.arange(0, img_h, dtype=torch.float32, device=intrs_inv.device).add_(0.5)
            grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')        # [H, W]
            pixel_coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)     # [H*W, 2]
        pixel_coords = pixel_coords.unsqueeze(0)                                    # [1, H*W, 2]
        # 像素坐标系转相机坐标系
        pixel_coords_hom = self.pix2hom(pixel_coords)                               # [1, H*W, 3]
        cam_coords = self.pix2cam(pixel_coords_hom, intrs_inv)                      # [B, H*W, 3]
        cam_orgins = torch.zeros_like(cam_coords)                                   # [B, H*W, 3]
        # 相机坐标系转世界坐标系
        cam_coords_hom = self.cam2hom(cam_coords)                                   # [B, H*W, 4]
        cam_orgins_hom = self.cam2hom(cam_orgins)                                   # [B, H*W, 4]
        world_coords = self.cam2world(cam_coords_hom, c2ws)                         # [B, H*W, 3]
        world_origns = self.cam2world(cam_orgins_hom, c2ws)                         # [B, H*W, 3]
        # 光线原点和方向
        rays_o = world_origns                                                       # [B, H*W, 3]
        rays_d = world_coords - world_origns                                        # [B, H*W, 3]
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        return rays_o, rays_d

    ########### 进行网络训练和体渲染 ###########
    # xyz: [B, N, 3]  z_vals: [B, N]  rays_d: [B, 3]
    def render_pts(self, xyz, z_vals, rays_d, mlp_model, idx_fine=None, mode="train"):
        ##### 获取三维点对应的体密度和颜色 #####
        view_dirs = rays_d.unsqueeze(1).expand(-1, xyz.shape[1], -1)                                # [B, N, 3]
        if idx_fine is not None:
            xyz_ = xyz[idx_fine[:, 0], idx_fine[:, 1]]                                              # [idx_num*fine_scale, 3]
            view_dirs = view_dirs[idx_fine[:, 0], idx_fine[:, 1]]                                   # [idx_num*fine_scale, 3]
        else:
            xyz_ = xyz.reshape(-1, 3)                                                               # [B*N, 3]
            view_dirs = view_dirs.reshape(-1, 3)                                                    # [B*N, 3]
        # 位置和方向编码
        xyz_emb = self.embedding_xyz(xyz_)                                                          # [idx_num*fine_scale, 63]
        view_dirs = self.embedding_view(view_dirs)                                                  # [idx_num*fine_scale, 27]
        # 进入mlp网络
        if self.use_view_dirs:
            outputs = mlp_model(xyz_emb, view_dirs)                                                 # [B*N, 4] / [idx_num*fine_scale, 4]
        else:
            outputs = mlp_model(xyz_emb)                                                            # [B*N, 4] / [idx_num*fine_scale, 4]
        # 恢复输出维度
        if idx_fine is not None:
            out_rgb = torch.full((rays_d.shape[0], self.num_fine_samples, 3), 1.0, device=self.device)                     # [B, N, 3]
            out_sigma = torch.full((rays_d.shape[0], self.num_fine_samples, 1), self.sigmas_default, device=self.device)   # [B, N, 1]
            outputs_final = torch.cat([out_rgb, out_sigma], dim=-1)                                 # [B, N, 4]
            outputs_final[idx_fine[:, 0], idx_fine[:, 1]] = outputs                                 # [B, N, 4]
        else:
            outputs_final = outputs.reshape(rays_d.shape[0], self.num_coarse_samples, 4)            # [B, N, 4]

        # 分离颜色和密度
        rgbs, sigmas = outputs_final[..., :3], outputs_final[..., 3]                                # [B, N, 3], [B, N]
        ##### 体积渲染 #####
        # 求解采样点透明度
        deltas = z_vals[:, 1:] - z_vals[:, :-1]                                                     # [B, N-1]
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1])                                           # [B, 1]
        deltas = torch.cat([deltas, delta_inf], dim=-1)                                             # [B, N]
        rays_d_norm = torch.norm(rays_d.unsqueeze(1), dim=-1)                                       # [B, 1]
        deltas = deltas * rays_d_norm                                                               # [B, N]
        weights = self.sigmas2weights(deltas, sigmas, mode=mode)
        # 求解像素颜色、深度和光线透明度
        depth_map = torch.sum(z_vals * weights, dim=-1)                                             # [B]
        opacity_map = torch.sum(weights, dim=-1)                                                    # [B]
        rgbs_map = torch.sum(rgbs * weights.unsqueeze(-1), dim=1)                                   # [B, 3]
        if self.white_background:
            rgbs_map = rgbs_map + 1 - opacity_map.unsqueeze(-1)                                     # [B, 3]

        return rgbs_map, depth_map, opacity_map, weights, sigmas, deltas
    
    def sigmas2weights(self, deltas, sigmas, mode="train"):
        if mode == "train":
            noise = torch.randn(sigmas.shape, device=self.device)                                   # [B, N]
            sigmas = sigmas + noise
        else:
            sigmas = sigmas
        alphas = 1.0 - torch.exp(-torch.nn.Softplus()(sigmas) * deltas)                             # [B, N]
        # 求解采样点权重
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1-alphas+1e-10], dim=-1)      # [B, N+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]                          # [B, N]
        return weights

    # 保存模型
    def save_model(self, step):
        model_name = "{}-step-{}.ckpt".format(self.data_name, step)
        file_path = os.path.join(self.weights_path, model_name)
        save_dict = {'model_nerf': self.state_dict()}
        torch.save(save_dict, file_path)
        print("\nSave model: {}".format(model_name))

    # 加载已有权重
    def load_ckpt(self, ckpt_dict, nerf_model, coarse=False):
        nerf_ckpt_dict = ckpt_dict['model_nerf']
        print(nerf_ckpt_dict.keys())
        new_nerf_ckpt_dict = nerf_ckpt_dict.copy()
        # 粗网络
        if coarse:
            net_name = "coarse_net"
            # net_name = "nerf_coarse"
        # 细网络
        else:
            net_name = "fine_net"
            # net_name = "nerf_fine"
        for key in nerf_ckpt_dict:
            # 需要的网络权重参数重命名
            if net_name in key.split("."):
                # 先删掉旧权重参数
                del(new_nerf_ckpt_dict[key])
                # 参数重命名，删除前缀
                idx_name = key.split(".").index(net_name)
                new_name_list = key.split(".")[idx_name+1:]
                new_name = ""
                for i, name in enumerate(new_name_list):
                    new_name += name
                    if i != len(new_name_list) - 1:
                        new_name += "."
                new_nerf_ckpt_dict[new_name] = nerf_ckpt_dict[key]
            # 不需要的网络权重参数直接删除
            else:
                del(new_nerf_ckpt_dict[key])
        nerf_model.load_state_dict(new_nerf_ckpt_dict)
        
    def find_lasted_ckpt(self, folder):
        if not os.path.isdir(folder):
            return None
        pattern = re.compile(r'^' + re.escape(self.data_name) + r'-step-(\d+)\.ckpt$')
        best_file = None 
        best_step = -1
        for fn in os.listdir(folder):
            m = pattern.match(fn)
            if not m:
                continue
        step_str = m.group(1)
        step = int(step_str)
        full_path = os.path.join(folder, fn)
        if step > best_step:
            best_step = step
            best_file = full_path
        return best_file

    def pix2hom(self, pixel_coords):
        pixel_coords_hom = torch.cat([pixel_coords, torch.ones_like(pixel_coords[..., :1])], dim=-1)    # [B, 3]
        return pixel_coords_hom
    
    def cam2hom(self, cam_coords):
        cam_coords_hom = torch.cat([cam_coords, torch.ones_like(cam_coords[..., :1])], dim=-1)          # [B, 4]
        return cam_coords_hom
    
    def world2hom(self, world_coords):
        world_coords_hom = torch.cat([world_coords, torch.ones_like(world_coords[..., :1])], dim=-1)    # [B, 4]
        return world_coords_hom
    
    # intrs_inv: [B, 3, 3]  pixel_coords_hom: [B, N, 3]
    def pix2cam(self, pixel_coords_hom, intrs_inv):
        intrs_inv_T = torch.transpose(intrs_inv, -2, -1)        # [B, 3, 3]
        cam_coords = pixel_coords_hom @ intrs_inv_T             # [B, N, 3]
        return cam_coords        
    
    # c2w: [B, 4, 4]  cam_coords_hom: [B, N, 4]
    def cam2world(self, cam_coords_hom, c2ws):
        c2ws = c2ws[:, :3, :]                                   # [B, 3, 4]
        c2ws = torch.transpose(c2ws, -2, -1)                    # [B, 4, 3]
        world_coords = cam_coords_hom @ c2ws                    # [B, 3]
        return world_coords