import torch
from torch_scatter import segment_coo

class Loss_dict(torch.nn.Module):
    def __init__(self, sys_param):
        super(Loss_dict, self).__init__()
        self.sys_param = sys_param
        self.prop_pulse_width = self.sys_param['prop_pulse_width']
        self.rgb_weight = self.sys_param['rgb_loss_weight']
        self.dist_weight = self.sys_param['dist_loss_weight']
        self.inter_weight = self.sys_param['inter_loss_weight']
        self.hash_weight = self.sys_param['hash_loss_weight']
        self.loss_l2 = torch.nn.MSELoss(reduction='mean')
    
    def forward(self, outputs):
        # 计算RGB损失
        self.loss_rgb = self.get_rgb_loss(outputs)
        # 计算proposal-render权重损失
        self.loss_inter = self.get_inter_loss(outputs, self.prop_pulse_width)
        # 计算正则项损失，约束floater
        self.loss_dist = self.get_distortion_loss(outputs)
        # 计算哈希网格衰减损失
        self.loss_hash = self.get_hash_decay_loss(outputs)
        # 总损失
        self.loss_final = self.rgb_weight*self.loss_rgb + self.inter_weight*self.loss_inter +\
                     self.dist_weight*self.loss_dist + self.hash_weight*self.loss_hash
        return self.loss_final

    def get_rgb_loss(self, outputs):
        pd_rgbs = outputs['pd_rgbs']          # [B, 1, 1, 3]
        gt_rgbs = outputs['gt_rgbs']          # [B, 1, 1, 3]
        loss_rgb = self.loss_l2(pd_rgbs, gt_rgbs)
        return loss_rgb
    
    def get_inter_loss(self, outputs, pulse_info):
        # 获取proposal阶段的采样点和权重
        prop_sdist = outputs["prop_sdist"]
        prop_weights = outputs["prop_weights"]
        # 获取render阶段的采样点和权重
        render_sdist = outputs["render_sdist"]
        render_weights = outputs["render_weights"]
        # 权重归一化，计算单位权重，目的是
        render_weights_norm = render_weights / (render_sdist[..., 1:] - render_sdist[..., :-1] + 1e-8)
        loss_anti_interlevel = 0
        
        # 循环前面的proposal阶段
        for i in range(len(prop_sdist)):
            # 获取prop阶段的采样点和权重
            cur_prop_sdist = prop_sdist[i]
            cur_prop_weights = prop_weights[i]
            # 模糊阶跃函数
            nerf_dist_, render_weights_norm_ = self.blur_step_function(render_sdist, render_weights_norm, pulse_info[i])                    # [B, 1, 1, 2*N+2]  [B, 1, 1, 2*N+2]
            # 梯形积分函数
            area = 0.5 * (render_weights_norm_[..., 1:] + render_weights_norm_[..., :-1]) * (nerf_dist_[..., 1:] - nerf_dist_[..., :-1])    # [B, 1, 1, 2*N+1]
            cdf = torch.cat([torch.zeros_like(area[..., :1]), torch.cumsum(area, dim=-1)], dim=-1)                                          # [B, 1, 1, 2*N+2]     
            # 把cdf映射到prop采样点上，计算对应的权重
            cdf_interp = self.sorted_interp_quad(cur_prop_sdist, nerf_dist_, render_weights_norm_, cdf)
            w_s = torch.diff(cdf_interp, dim=-1)

            loss_anti_interlevel += ((w_s - cur_prop_weights).clamp_min(0) ** 2 / (cur_prop_weights + 1e-5)).mean()

        return loss_anti_interlevel

    def get_distortion_loss(self, outputs):
        # 获取nerf阶段的采样点和权重
        render_weights = outputs["render_weights"]                                                                      # [B, 1, 1, N]
        render_sdist = outputs["render_sdist"]                                                                          # [B, 1, 1, N+1]
        # 获取采样点的中点坐标
        render_sdist_intervals = (render_sdist[..., 1:] + render_sdist[..., :-1]) / 2                                   # [B, 1, 1, N]                               
        # 计算每个采样点与其他所有采样点见的距离
        sdist_all = torch.abs(render_sdist_intervals[..., :, None] - render_sdist_intervals[..., None, :])              # [B, 1, 1, N, N]
        # 损失函数15的第一个部分
        loss_part1 = torch.sum(render_weights * torch.sum(render_weights[..., None, :] * sdist_all, dim=-1), dim=-1)    # [B, 1, 1]
        # 损失函数15的第二个部分
        loss_part2 = torch.sum(render_weights ** 2 * (render_sdist[..., 1:] - render_sdist[..., :-1]), dim=-1) / 3      # [B, 1, 1]
        # 两项相加
        loss_value = loss_part1.abs() + loss_part2.abs()
        
        return loss_value.mean()

    def get_hash_decay_loss(self, outputs):
        encoder_param = outputs["enc_embds"]
        encoder_idx = outputs["enc_idx"]
        loss_value = 0
        for i in range(len(encoder_param)):
            cur_idx = encoder_idx[i]
            cur_param = encoder_param[i]
            loss_hash_decay = segment_coo(cur_param**2, cur_idx,
                                torch.zeros(cur_idx.max() + 1, cur_param.shape[-1], device=cur_param.device),
                                reduce='mean').mean()
            loss_value += loss_hash_decay
        return loss_value

    # render_sdist: [B, 1, 1, N+1]  render_weights_norm: [B, 1, 1, N]
    def blur_step_function(self, render_sdist, render_weights_norm, pulse_width):
        # 按照宽度向两边扩张采样边界, 采样点数量扩大为2倍
        render_sdist_bound_expand, sort_idx = torch.sort(torch.cat([render_sdist - pulse_width, render_sdist + pulse_width], dim=-1))   # [B, 1, 1, 2N+2]
        # 计算每一个相邻采样点权重的差分增量(变化量)
        render_weights_expd1 = torch.cat([render_weights_norm, torch.zeros_like(render_weights_norm[..., :1])], dim=-1)                 # [B, 1, 1, N+1]
        render_weights_expd2 = torch.cat([torch.zeros_like(render_weights_norm[..., :1]), render_weights_norm], dim=-1)                 # [B, 1, 1, N+1]
        render_weights_diff = render_weights_expd1 - render_weights_expd2                                                               # [B, 1, 1, N+1]
        # 计算归一化的变化量(变化率)，保证在不同的模糊半径下(pulse_width)，都能够统一的表示变化率
        render_weights_radio = render_weights_diff / (2*pulse_width)                                                                    # [B, 1, 1, N+1]
        # 合并正负变化率
        render_weights_radio_double = torch.cat([render_weights_radio, -render_weights_radio], dim=-1)                                  # [B, 1, 1, 2N+2]
        # 按照之前扩张排序对变化率排序进行调整，保证一一对应
        render_weights_radio_double_sort = render_weights_radio_double.take_along_dim(sort_idx[..., :-1], dim=-1)                       # [B, 1, 1, 2N+1]
        # 计算所有采样点相对于前一个采样点的距离
        expd_sdist_to_start = render_sdist_bound_expand[..., 1:] - render_sdist_bound_expand[..., :-1]                                  # [B, 1, 1, 2N+1]
        # 计算所有采样点相对于第一个采样点的权重变化率
        expd_radio_to_start = torch.cumsum(render_weights_radio_double_sort, dim=-1)                                                    # [B, 1, 1, 2N+1]
        # 将上面两者相乘，摆脱归一化
        # 说明：在第33行代码里除以了2r，得到的是归一化数据，这里再乘回去，变成非归一化
        # 也是通过这个步骤，将原来的变化率扩展到了整条光线上，实现平滑过程
        recover_weights_diff = expd_sdist_to_start * expd_radio_to_start                                                                # [B, 1, 1, 2N+1]
        # 再累加，从权重差分变成权重
        recover_weights = torch.cumsum(recover_weights_diff, dim=-1).clamp_min(0)                                                       # [B, 1, 1, 2N+1]                                         
        blur_weights = torch.cat([torch.zeros_like(recover_weights[..., :1]), recover_weights], dim=-1)                                 # [B, 1, 1, 2N+2]

        return render_sdist_bound_expand, blur_weights
    
    def sorted_interp_quad(self, x, xp, fpdf, fcdf):
        mask = x[..., None, :] >= xp[..., :, None]

        def find_interval(x, return_idx=False):
            # Grab the value where `mask` switches from True to False, and vice versa.
            # This approach takes advantage of the fact that `x` is sorted.
            x0, x0_idx = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
            x1, x1_idx = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)
            if return_idx:
                return x0, x1, x0_idx, x1_idx
            return x0, x1

        fcdf0, fcdf1, fcdf0_idx, fcdf1_idx = find_interval(fcdf, return_idx=True)
        fpdf0 = fpdf.take_along_dim(fcdf0_idx, dim=-1)
        fpdf1 = fpdf.take_along_dim(fcdf1_idx, dim=-1)
        xp0, xp1 = find_interval(xp)

        offset = torch.clip(torch.nan_to_num((x - xp0) / (xp1 - xp0), 0), 0, 1)
        ret = fcdf0 + (x - xp0) * (fpdf0 + fpdf1 * offset + fpdf0 * (1 - offset)) / 2
        return ret
