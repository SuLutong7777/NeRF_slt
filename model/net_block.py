import torch 
import torch.nn.functional as F
from .net_utils import eval_sh

# 位置编码
class SinCosEmbedding(torch.nn.Module):
    def __init__(self, sys_param, freqs):
        super(SinCosEmbedding, self).__init__()
        self.device = sys_param['device']
        # 输入维度
        self.input_dim = 3
        # 输出维度
        self.output_dim = self.input_dim * freqs * 2 + self.input_dim
        # 编码维度
        self.freqs = 2 ** torch.arange(freqs, dtype=torch.float32, device=self.device)     # [F]
        
    def forward(self, x):
        batch_size = x.shape[0]                                         # x: [B, 3]
        x_freqs = x[..., None] * self.freqs                             # [B, 3, F]
        x_sin = torch.sin(x_freqs)                                      # [B, 3, F]
        x_cos = torch.cos(x_freqs)                                      # [B, 3, F]
        x_freqs_sincos = torch.stack([x_sin, x_cos], dim=-2)            # [B, 3, 2, F]
        x_freqs_sincos = x_freqs_sincos.view(batch_size, -1)            # [B, 3*2*F]
        x_emb = torch.cat([x, x_freqs_sincos], dim=-1)                  # [B, 3 + 3*2*F]
        return x_emb

# NeRF MLP网络块
class CoarseFine_Net(torch.nn.Module):
    def __init__(self, sys_param, MLP_type='Coarse'):
        super(CoarseFine_Net, self).__init__()
        ### 初始化参数
        self.sys_param = sys_param
        self.use_view_dirs = sys_param['use_view_dirs']
        self.view_dim = 3 + 3 * 2 * sys_param['freqs_view']     # 方向编码后的输入维度
        # self.view_dim = 3
        self.input_dim = 3 + 3 * 2 * sys_param['freqs_xyz']     # 位置编码后的输入维度
        self.output_dim = 4                                     # 输出维度(r,g,b,sigma)
        if MLP_type == 'Coarse':
            self.MLP_depth = sys_param['coarse_MLP_depth']
            self.MLP_width = sys_param['coarse_MLP_width']
            self.MLP_skips = sys_param['coarse_MLP_skips']
        elif MLP_type == 'Fine':
            self.MLP_depth = sys_param['fine_MLP_depth']
            self.MLP_width = sys_param['fine_MLP_width']
            self.MLP_skips = sys_param['fine_MLP_skips']
        
        #### 构建MLP网络
        for i in range(self.MLP_depth):
            # 第一层
            if i == 0:
                layer = torch.nn.Linear(self.input_dim, self.MLP_width)
            # 跳跃连接层
            elif i in self.MLP_skips:
                layer = torch.nn.Linear(self.MLP_width + self.input_dim, self.MLP_width)
            # 其他隐藏层
            else:
                layer = torch.nn.Linear(self.MLP_width, self.MLP_width)
            layer = torch.nn.Sequential(layer, torch.nn.ReLU(True))         # 直接改输入，节省内存
            setattr(self, 'layer_{}'.format(i), layer)
        # 输出层
        if self.use_view_dirs:          # 使用视角方向
            self.sigma_layer = torch.nn.Sequential(torch.nn.Linear(self.MLP_width, self.MLP_width),
                                                   torch.nn.ReLU(True),
                                                   torch.nn.Linear(self.MLP_width, 1))
            self.feature_layer = torch.nn.Linear(self.MLP_width, self.MLP_width)
            # self.feature_layer = torch.nn.Sequential(torch.nn.Linear(self.MLP_width, self.MLP_width),
            #                                          torch.nn.ReLU(True))
            self.rgb_layer = torch.nn.Sequential(torch.nn.Linear(self.MLP_width + self.view_dim, self.MLP_width // 2),
                                                 torch.nn.ReLU(True),
                                                 torch.nn.Linear(self.MLP_width // 2, 3))
        else:                           # 不使用视角方向
            self.output_layer = torch.nn.Linear(self.MLP_width, self.output_dim)

    def forward(self, xyz, view_dirs=None):
        xyz_ = xyz                                                          # xyz: [B, input_dim]
        # MLP隐藏层
        for i in range(self.MLP_depth):
            layer = getattr(self, 'layer_{}'.format(i))
            # 跳跃连接
            if i in self.MLP_skips:
                xyz_ = torch.cat([xyz_, xyz], dim=-1)
            xyz_ = layer(xyz_)
        # 输出层
        if self.use_view_dirs:
            sigma = self.sigma_layer(xyz_)                                  # [B, 1]
            feature = self.feature_layer(xyz_)                              # [B, W]
            rgb = self.rgb_layer(torch.cat([feature, view_dirs], dim=-1))   # [B, 3]
            rgb = torch.sigmoid(rgb)                                        # [B, 3]
            output = torch.cat([rgb, sigma], dim=-1)                        # [B, 4]
        else:
            output = self.output_layer(xyz_)                                # [B, 4]   
        return output