import torch
import os
import re
from model import SinCosEmbedding, GridEncoder, NeRF_MLP, Prop_MLP

class NeRF_Model(torch.nn.Module):
    def __init__(self, sys_param, load_weights=False):
        super(NeRF_Model, self).__init__()
        self.sys_param = sys_param
        self.device = self.sys_param['device']
        self.pca_mode = self.sys_param['pca_mode']
        self.data_name = self.sys_param['data_name']
        self.white_background = self.sys_param['white_background']
        self.output_path = self.sys_param['output_path']
        self.weights_path = os.path.join(self.output_path, "weights")
        os.makedirs(self.weights_path, exist_ok=True)
        ################# 光线采样参数 #################
        # 理想化归一化的光线距离
        self.sdist_near = 0.0
        self.sdist_far = 1.0
        # 实际场景光线距离
        if self.pca_mode:
            self.near = 0.0
            self.far = 4.0
            # self.far = self.sys_param['radius']
        else:
            self.near = self.sys_param['near']
            self.far = self.sys_param['far']
        print("near far: ", self.near, self.far)
        # 采样点数量
        self.p_samples = self.sys_param['prop_samples']
        self.r_samples = self.sys_param['render_samples']
        ################# 编码参数 #################
        self.freqs_view = self.sys_param['render_freqs_view']
        self.prop_num_level = self.sys_param['prop_num_level']
        self.prop_level_dim = self.sys_param['prop_level_dim']
        self.render_num_level = self.sys_param['render_num_level']
        self.render_level_dim = self.sys_param['render_level_dim']
        ################# 例化编码和网络 #################
        # 方向编码
        self.embedding_view = SinCosEmbedding(self.sys_param, self.freqs_view)
        # 例化prop阶段的编码和mlp网络
        for lv in range(len(self.p_samples)):
            self.register_module(f'prop_hash_{lv}', GridEncoder(self.sys_param, lv=lv, prop=True))
            self.register_module(f'prop_mlp_{lv}', Prop_MLP(self.sys_param, lv=lv))
        # 例化NeRF阶段的编码和mlp网络
        self.render_hash = GridEncoder(self.sys_param, lv=0, prop=False)
        self.render_mlp = NeRF_MLP(self.sys_param)
    
    def forward(self, rays_o, rays_d):
        batch_size = rays_d.shape[0]                            # rays_d: [B, 3]
        # 生成光线采样模板  [B, 1, 1, 2]  
        prop_sdist = self.generate_base_sample(batch_size)
        # 生成与采样匹配的权重  [B, 1, 1, 1]  
        prop_weight = self.generate_base_weight(batch_size)
        # 光线格式一致
        rays_o = rays_o.unsqueeze(1).unsqueeze(1)               # [B, 1, 1, 3]
        rays_d = rays_d.unsqueeze(1).unsqueeze(1)               # [B, 1, 1, 3]

        outputs = {}
        enc_idx_list = []
        enc_embds_list = []
        ################# prop采样阶段 #################
        prop_weights_list = []
        prop_sdist_list = []
        for lv in range(len(self.p_samples)):
            # 进行prop阶段采样 
            # prop_sigma: 预测每个采样点的体密度 [B, 1, 1, p_samples]  
            # prop_tdist: 真实场景尺度区间端点 [B, 1, 1, p_samples+1] 
            # prop_sdist: 归一化尺度区间端点 [B, 1, 1, p_samples+1]
            prop_sigma, prop_tdist, prop_sdist, prop_idx, prop_embds = self.prop_stage(prop_sdist, prop_weight, lv, rays_o, rays_d)
            # 计算权重，供下一阶段使用 prop_weight: [B, 1, 1, p_samples]
            prop_weight = self.get_volumetric_weights(prop_sigma, prop_tdist, rays_d)
            prop_weights_list.append(prop_weight)
            prop_sdist_list.append(prop_sdist)
            enc_idx_list.append(prop_idx)
            enc_embds_list.append(prop_embds)
            outputs['prop_weights'] = prop_weights_list     # len=L  每项为[B, 1, 1, p_samples]
            outputs['prop_sdist'] = prop_sdist_list         # len=L  每项为[B, 1, 1, p_samples+1]
        ################# NeRF渲染阶段 #################
        # 依据最后一阶段prop结果进行重要性采样 
        # render_sigma: [B, 1, 1, r_samples, 1]  render_rgb: [B, 1, 1, r_samples, 3]  render_tdist: [B, 1, 1, r_samples+1]  render_sdist: [B, 1, 1, r_samples+1]
        render_sigma, render_rgb, render_tdist, render_sdist, render_idx, render_embds = self.render_stage(prop_sdist, prop_weight, 0, rays_o, rays_d)
        # 计算采样点权重    render_weight: [B, 1, 1, r_samples]
        render_weight = self.get_volumetric_weights(render_sigma[..., 0], render_tdist, rays_d)
        # 计算像素颜色、深度和光线透明度
        # render_opacity: [B, 1, 1, 1]    render_rgb: [B, 1, 1, 3]    render_depth: [B, 1, 1, 1]
        render_opacity, render_rgb, render_depth = self.volumetric_rendering(render_weight, render_rgb, render_tdist)
        enc_idx_list.append(render_idx)
        enc_embds_list.append(render_embds)

        outputs['pd_rgbs'] = render_rgb                     # [B, 1, 1, 3]
        outputs['render_weights'] = render_weight           # [B, 1, 1, r_samples]
        outputs['render_sdist'] = render_sdist              # [B, 1, 1, r_samples+1]
        outputs['enc_idx'] = enc_idx_list                   # len=L  
        outputs['enc_embds'] = enc_embds_list               # len=L  
        return outputs
    
    ################# prop阶段 #################
    # cur_sdist: [B, 1, 1, N+1]  cur_weight: [B, 1, 1, N]  rays_o: [B, 1, 1, 3]  rays_d: [B, 1, 1, 3]
    # 注：第一阶段cur_sdist为[B, 1, 1, 2]  cur_weight为[B, 1, 1, 1]
    def prop_stage(self, cur_sdist, cur_weight, cur_stage, rays_o, rays_d):
        # 保证光线采样点合理性,并且为后续的softmax做准备
        weights_logits = self.logits_resample(cur_sdist, cur_weight, exp_schlick=1.0)                       # [B, 1, 1, N] 
        # 根据cur_sdist和weights_logits进行重要性采样，范围在0-1之间
        # samples_s: 采样点[B, 1, 1, p_samples]     sdist: 采样区间点[B, 1, 1, p_samples+1]
        samples_s, sdist = self.sample_intervals(cur_sdist, weights_logits, self.p_samples[cur_stage], domain=[self.sdist_near, self.sdist_far])   
        # 从0-1范围映射到实际场景距离范围
        samples_t = self.near * (1.0 - samples_s) + self.far * samples_s                                    # [B, 1, 1, p_samples]
        tdist = self.near * (1.0 - sdist) + self.far * sdist                                                # [B, 1, 1, p_samples+1]
        # 计算采样点三维位置
        samples_xyz = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * samples_t.unsqueeze(-1)                 # [B, 1, 1, p_samples, 3]
        # 将采样点转换到 [-1, 1] 范围内
        contract_xyz = self.contract_xyz(samples_xyz)                                                       # [B, 1, 1, p_samples, 3]
        # 位置编码
        self.prop_hash = self.get_submodule(f'prop_hash_{cur_stage}')
        features_xyz = self.prop_hash(contract_xyz)                                                         # [B, 1, 1, p_samples, num_level*level_dim]
        # 进入mlp网络, 输出体密度
        self.prop_mlp = self.get_submodule(f'prop_mlp_{cur_stage}')
        sigma = self.prop_mlp(features_xyz)                                                                 # [B, 1, 1, p_samples]
        return sigma, tdist, sdist.detach(), self.prop_hash.idx, self.prop_hash.embeddings
    
    ################# NeRF渲染阶段 #################
    # cur_sdist: [B, 1, 1, N+1]  cur_weight: [B, 1, 1, N]  rays_o: [B, 1, 1, 3]  rays_d: [B, 1, 1, 3]
    def render_stage(self, cur_sdist, cur_weight, cur_stage, rays_o, rays_d):
        # 保证光线采样点合理性,并且为后续的softmax做准备
        weights_logits = self.logits_resample(cur_sdist, cur_weight, exp_schlick=1.0)                       # [B, 1, 1, N]
        # 根据cur_sdist和weights_logits进行重要性采样，范围在0-1之间
        # samples_s: [B, 1, 1, r_samples]     sdist: [B, 1, 1, r_samples+1]
        samples_s, sdist = self.sample_intervals(cur_sdist, weights_logits, self.r_samples[cur_stage], domain=[self.sdist_near, self.sdist_far])   
        # 从0-1范围映射到实际场景距离范围
        samples_t = self.near * (1.0 - samples_s) + self.far * samples_s                                    # [B, 1, 1, r_samples]
        tdist = self.near * (1.0 - sdist) + self.far * sdist                                                # [B, 1, 1, r_samples+1]
        # 计算采样点三维位置
        samples_xyz = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * samples_t.unsqueeze(-1)                 # [B, 1, 1, r_samples, 3]
        # 将采样点转换到 [-1, 1] 范围内
        contract_xyz = self.contract_xyz(samples_xyz)                                                       # [B, 1, 1, r_samples, 3]
        # 位置编码
        features_xyz = self.render_hash(contract_xyz)                                                       # [B, 1, 1, r_samples, num_level*level_dim]
        # 方向编码
        rays_d_samples = rays_d.unsqueeze(-2).repeat(1, 1, 1, self.r_samples[cur_stage], 1)                 # [B, 1, 1, r_samples, 3]
        view_dirs_emb = self.embedding_view(rays_d_samples)                                                 # [B, 1, 1, r_samples, view_dim]
        # 进入mlp网络, 输出体密度和颜色
        sigma, rgb = self.render_mlp(view_dirs_emb, features_xyz)                                           # [B, 1, 1, r_samples, 1], [B, 1, 1, r_samples, 3]
        return sigma, rgb, tdist, sdist.detach(), self.render_hash.idx, self.render_hash.embeddings

    # 生成基础采样光线的两个端点，然后再进行prop采样
    def generate_base_sample(self, batch):
        # zipnerf中的格式
        sdist = torch.cat([
            torch.full([batch, 1, 1, 1], self.sdist_near, device=self.device),
            torch.full([batch, 1, 1, 1], self.sdist_far,  device=self.device)
        ], dim=-1)            
        return sdist
    
    # 生成基础权重
    def generate_base_weight(self, batch):
        weight = torch.full([batch, 1, 1, 1], 1.0, device=self.device)
        return weight

    def logits_resample(self, sdist, weights, exp_schlick):
        # 检查是否有无效点，防止无效点对softmax的影响
        valid_mask = sdist[..., 1:] > sdist[..., :-1]                                   # [B, 1, 1, N]
        # valid_mask为true则取前一项 prop阶段第一轮为0
        resample = torch.where(valid_mask,
                               exp_schlick*torch.log(weights),
                               torch.full_like(weights, -torch.inf))
        return resample

    # 依据sdist和权重进行区间采样
    def sample_intervals(self, sdist, w_logits, num_samples, domain=[-torch.inf, torch.inf]):
        #----- 1. 生成均匀采样点 -----#
        # 极小正数并计算间隔区间长度
        eps = torch.finfo(sdist.dtype).eps
        inter_length = eps + (1 - eps) / num_samples
        # 生成采样点位置并扩展到各个光线    [B, 1, 1, num_samples]
        inter_position = torch.linspace(0, 1 - inter_length, num_samples, device=self.device)
        inter_position = inter_position.expand(sdist.shape[:-1] + (num_samples,))
        # 计算扰动最大范围并添加随机扰动
        max_jitter = (1 - inter_length) / (num_samples - 1) - eps
        shift_position = torch.rand(inter_position.shape[-1], device=self.device) * max_jitter               
        shift_position = shift_position.expand(sdist.shape[:-1] + (num_samples,))
        # 计算最终采样位置, 采样点在0-1范围内   [B, 1, 1, num_samples]
        final_position = inter_position + shift_position
        #----- 2. 根据 w_logits 重新分配均匀采样点 -----#
        new_samples = self.invert_cdf(final_position, sdist, w_logits)                      # [B, 1, 1, num_samples]
        #----- 3. 调整采样点 -----#
        # 计算采样点中点位置
        mid_samples = (new_samples[..., 1:] + new_samples[..., :-1]) / 2.0                  # [B, 1, 1, num_samples-1]
        # 添加边界点(利用对称点求解)
        minval, maxval = domain
        first_samples = (2 * new_samples[..., :1] - mid_samples[..., :1]).clamp_min(minval)    # [B, 1, 1, 1]
        last_samples = (2 * new_samples[..., -1:] - mid_samples[..., -1:]).clamp_max(maxval)   # [B, 1, 1, 1]
        # 拼接最终采样点
        final_sdist = torch.cat([first_samples, mid_samples, last_samples], dim=-1)          # [B, 1, 1, num_samples+1]
        return new_samples, final_sdist.detach()

    # 根据 w_logits 重新分配均匀采样点
    # samples: [B, 1, 1, num_samples]  sdist: [B, 1, 1, N+1]  w_logits: [B, 1, 1, N]
    def invert_cdf(self, samples, sdist, w_logits):
        # 用softmax将logits转换为概率分布
        weights = torch.softmax(w_logits, dim=-1)                                               # [B, 1, 1, N]
        # 计算累积分布函数(CDF)
        cw = torch.cumsum(weights[..., :-1], dim=-1).clamp_max(1)                               # [B, 1, 1, N-1]
        cdf = torch.cat([torch.zeros_like(weights[..., :1]), cw,
                         torch.ones_like(weights[..., :1])], dim=-1)                            # [B, 1, 1, N+1]
        # 在CDF上进行反向采样
        mask = samples[..., None, :] >= cdf[..., :, None]                                       # [B, 1, 1, N+1, num_samples]
        # 计算采样点所在新采样区间的起点和终点  [B, 1, 1, num_samples]
        fp0 = torch.max(torch.where(mask, sdist[..., None], sdist[..., :1, None]), dim=-2).values
        fp1 = torch.min(torch.where(~mask, sdist[..., None], sdist[..., -1:, None]), dim=-2).values
        # 计算采样点所在新采样区间的cdf起点和终点  [B, 1, 1, num_samples]
        xp0 = torch.max(torch.where(mask, cdf[..., None], cdf[..., :1, None]), dim=-2).values
        xp1 = torch.min(torch.where(~mask, cdf[..., None], cdf[..., -1:, None]), dim=-2).values
        # 线性插值计算最终采样点位置
        offset = torch.clip(torch.nan_to_num((samples - xp0) / (xp1 - xp0), 0), 0.0, 1.0)
        new_samples = fp0 + offset * (fp1 - fp0)                                               # [B, 1, 1, num_samples]
        return new_samples
        
    def contract_xyz(self, xyz):
        eps = torch.finfo(xyz.dtype).eps
        # 求三维点到原点的距离
        xyz_sq = torch.sum(xyz ** 2, dim=-1, keepdim=True).clamp_min(eps)                                   # [B, 1, 1, num_samples, 1]
        xyz_sqrt = torch.sqrt(xyz_sq)                                                                       # [B, 1, 1, num_samples, 1]
        # 计算收缩后的三维点位置 范围：[-2, 2]
        mask = xyz_sq <= 1.0
        xyz_contracted = torch.where(mask, xyz, (2.0 - 1.0 / xyz_sqrt) * (xyz / xyz_sqrt))                  # [B, 1, 1, num_samples, 3]   
        # 收缩到[-1, 1]
        xyz_contracted = xyz_contracted / 2.0                                                               # [B, 1, 1, num_samples, 3]
        return xyz_contracted

    def get_volumetric_weights(self, sigma, tdist, rays_d):
        # 计算采样点间隔
        deltas = tdist[..., 1:] - tdist[..., :-1]                                                               # [B, 1, 1, num_samples]
        deltas = deltas * torch.norm(rays_d[..., None, :], dim=-1)                                              # [B, 1, 1, num_samples]
        sigma_deltas = sigma * deltas                                                                           # [B, 1, 1, num_samples]
        # 计算点的不透明度
        alphas = 1.0 - torch.exp(-sigma_deltas)                                                                 # [B, 1, 1, num_samples]
        trans = torch.exp(-torch.cat([torch.zeros_like(sigma_deltas[..., :1]), 
                                      torch.cumsum(sigma_deltas[..., :-1], dim=-1)], dim=-1))                   # [B, 1, 1, num_samples]
        weights = alphas * trans                                                                                # [B, 1, 1, num_samples]
        return weights

    # 体积渲染 weight: [B, 1, 1, N]  rgb: [B, 1, 1, N, 3]  tdist: [B, 1, 1, N+1]
    def volumetric_rendering(self, weight, rgb, tdist):
        eps = torch.finfo(tdist.dtype).eps
        # 计算光线透明度
        opacity = torch.sum(weight, dim=-1, keepdim=True)                                                   # [B, 1, 1, 1]
        # 计算像素颜色
        rgb_map = torch.sum(weight[..., None] * rgb, dim=-2)                                                # [B, 1, 1, 3]
        if self.white_background:
            bgw = (1.0 - opacity).clamp_min(0.0)                                                            # [B, 1, 1, 1]
            rgb_map = rgb_map + bgw * 1.0                                                                   # [B, 1, 1, 3]
        # 计算像素深度
        samples_t = 0.5 * (tdist[..., :-1] + tdist[..., 1:])                                                # [B, 1, 1, N]
        depth_map = torch.sum(weight * samples_t, dim=-1, keepdim=True)                                     # [B, 1, 1, 1]
        depth_map = depth_map / opacity.clamp_min(eps)                                                      # [B, 1, 1, 1]
        # 对深度信息进行处理，如果数据中有nan，就会被替换成0，如果有正负无穷，会被替换成torch.inf
        depth_map = torch.nan_to_num(depth_map, torch.inf)
        # 限定数值范围，最小为采样最近点，最大为采样最远点
        depth_map = torch.clip(depth_map, tdist[..., 0], tdist[..., -1])                                  # [B, 1, 1, 1]
        return opacity, rgb_map, depth_map
    
    ########### intrs_inv: [B, 3, 3] c2ws: [B, 4, 4] ###########
    def get_rays(self, img_h, img_w, intrs_inv, c2ws):
        # 生成像素坐标网格
        with torch.no_grad():
            x_range = torch.arange(0, img_w, dtype=torch.float32, device=intrs_inv.device).add_(0.5)
            y_range = torch.arange(0, img_h, dtype=torch.float32, device=intrs_inv.device).add_(0.5)
            grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')        # [H, W]
            pixeloords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)     # [H*W, 2]
        pixeloords = pixeloords.unsqueeze(0)                                    # [1, H*W, 2]
        # 像素坐标系转相机坐标系
        pixeloords_hom = self.pix2hom(pixeloords)                               # [1, H*W, 3]
        camoords = self.pix2cam(pixeloords_hom, intrs_inv)                      # [B, H*W, 3]
        cam_orgins = torch.zeros_like(camoords)                                   # [B, H*W, 3]
        # 相机坐标系转世界坐标系
        camoords_hom = self.cam2hom(camoords)                                   # [B, H*W, 4]
        cam_orgins_hom = self.cam2hom(cam_orgins)                                   # [B, H*W, 4]
        worldoords = self.cam2world(camoords_hom, c2ws)                         # [B, H*W, 3]
        world_origns = self.cam2world(cam_orgins_hom, c2ws)                         # [B, H*W, 3]
        # 光线原点和方向
        rays_o = world_origns                                                       # [B, H*W, 3]
        rays_d = worldoords - world_origns                                        # [B, H*W, 3]
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        return rays_o, rays_d

    # 保存模型
    def save_model(self, step):
        model_name = "{}-step-{}.ckpt".format(self.data_name, step)
        file_path = os.path.join(self.weights_path, model_name)
        save_dict = {'model_nerf': self.state_dict()}
        torch.save(save_dict, file_path)
        print("\nSave model: {}".format(model_name))
        
    def pix2hom(self, pixeloords):
        pixeloords_hom = torch.cat([pixeloords, torch.ones_like(pixeloords[..., :1])], dim=-1)    # [B, 3]
        return pixeloords_hom
    
    def cam2hom(self, camoords):
        camoords_hom = torch.cat([camoords, torch.ones_like(camoords[..., :1])], dim=-1)          # [B, 4]
        return camoords_hom
    
    def world2hom(self, worldoords):
        worldoords_hom = torch.cat([worldoords, torch.ones_like(worldoords[..., :1])], dim=-1)    # [B, 4]
        return worldoords_hom
    
    # intrs_inv: [B, 3, 3]  pixeloords_hom: [B, N, 3]
    def pix2cam(self, pixeloords_hom, intrs_inv):
        intrs_inv_T = torch.transpose(intrs_inv, -2, -1)        # [B, 3, 3]
        camoords = pixeloords_hom @ intrs_inv_T                 # [B, N, 3]
        return camoords        
    
    # c2w: [B, 4, 4]  camoords_hom: [B, N, 4]
    def cam2world(self, camoords_hom, c2ws):
        c2ws = c2ws[:, :3, :]                                   # [B, 3, 4]
        c2ws = torch.transpose(c2ws, -2, -1)                    # [B, 4, 3]
        worldoords = camoords_hom @ c2ws                        # [B, 3]
        return worldoords