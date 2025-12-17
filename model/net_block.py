import torch 
import torch.nn.functional as F
import numpy as np
import math
from extensions import CUDA_Backend
from torch.autograd import Function

# 自己定义一个前向和反向传播的计算过程
class Grid_Custom_Flow(Function):
    # 继承自Function，必须定义，类似于写模型nn.model必须要定义forward一样
    # 参数说明：
    # ctx: 这个东西比较神奇，是由PyTorch在调用forward方法时自动提供的，
    #      用于在前向和反向传播之间保存状态信息，不需要处理，当做占位符(类比self)就行，backend才是传入的第一个参数.
    # backend: 自己定义的一个后端对象类，负责调用后端接口
    # inputs: 输入的数据，待进行编码的采样点坐标, 0-1的float类型 [N, 3]
    # embeddings: 哈希表的学习参数，float类型 [哈希表总长度，特征维度]
    # offsets: 每一层哈希表之前的累加尺寸和 [哈希表层数+1]
    # num_level: 哈希表的层数
    @staticmethod
    def forward(ctx, backend, inputs, embeddings, offsets, num_level, level_scale, base_resolution):
        # 获取输入维度信息
        batch  = inputs.shape[0]            # 批大小
        in_dim = inputs.shape[1]            # 输入维度(3)
        feat_dim  = embeddings.shape[1]     # 特征维度(2)
        # 取log在原文中说是方便cuda计算
        log_level_scale = math.log2(level_scale)
        # 转成内存连续的变量
        inputs = inputs.contiguous()
        # 定义一个占位变量，承接输出数据
        outputs = torch.empty([num_level, batch, feat_dim], device=inputs.device, dtype=embeddings.dtype)
        # 如果输入需要梯度，分配空间存储雅可比矩阵
        if inputs.requires_grad:
            dy_dx = torch.empty(batch, num_level*in_dim*feat_dim, device=inputs.device, dtype=embeddings.dtype)
        else:
            dy_dx = None

        # 同步一下进程，等待所有GPU中的任务完毕
        backend.synchronize()
        # 调用API进行计算
        # gridtype = 0 表示哈希
        # align_corners 默认True
        # interpolation = 0 表示线性插值
        backend.grid_encode_forward(inputs, embeddings, offsets, outputs, batch, in_dim, feat_dim, 
                                    num_level, log_level_scale, base_resolution, dy_dx, 0, True, 0)

        # 将输出数据的尺寸转换回原来的尺寸
        outputs = outputs.permute(1, 0, 2).reshape(batch, num_level*feat_dim)
        # 保存属性，与backward交互使用
        # 是PyTorch为实现自定义前向和后向传播提供的一种机制。通过这种方式，
        # 你可以在forward方法中计算并保存那些对于梯度计算（即backward方法）来说重要的数据和状态。
        ctx.backend = backend
        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        # gridtype = 0 表示哈希（-2）
        # interpolation = 0 表示线性插值(-1)
        ctx.dims = [batch, in_dim, feat_dim, num_level, log_level_scale, base_resolution, 0, 0]
        ctx.align_corners = True

        return outputs
    
    @staticmethod
    def backward(ctx, grad):
        # ctx: 这个东西比较神奇，是由PyTorch在调用forward方法时自动提供的，
        #      用于在前向和反向传播之间保存状态信息，不需要处理，当做占位符(类比self)就行，backend才是传入的第一个参数.
        backend = ctx.backend
        # 承接各种输出
        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        # gridtype = 0 表示哈希（-2）
        # interpolation = 0 表示线性插值(-1)
        batch, in_dim, feat_dim, num_level, log_level_scale, base_resolution, gridtype, interpolation = ctx.dims
        align_corners = ctx.align_corners
        # 梯度重塑
        # grad: [num_level, num_level*feat_dim] --> [num_level, batch, feat_dim]
        grad = grad.reshape(batch, num_level, feat_dim).permute(1, 0, 2).contiguous()
        # 梯度初始化
        grad_embeddings = torch.zeros_like(embeddings)                          # 哈希表参数的梯度
        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)      # 输入坐标的梯度
        else:
            grad_inputs = None
        # 同步一下进程，等待所有GPU中的任务完毕
        backend.synchronize()
        # cuda反向传播计算
        backend.grid_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, batch, in_dim, feat_dim,\
                                    num_level, log_level_scale, base_resolution, dy_dx, grad_inputs, gridtype, align_corners, interpolation)
        
        if dy_dx is not None:
            grad_inputs = grad_inputs.to(inputs.dtype)

        return None, grad_inputs, grad_embeddings, None, None, None, None, None, None, None

# 哈希编码
class GridEncoder(torch.nn.Module):
    def __init__(self, sys_param):
        super(GridEncoder, self).__init__()
        #################### 参数始化 ######################
        self.sys_param = sys_param
        # 输入尺寸,xyz三项
        self.input_dim = 3
        # 哈希表层数，INGP论文中的L=10
        self.num_level = self.sys_param["num_level"]
        # 哈希表每一层特征宽度，INGP论文中的F=4
        self.level_dim = self.sys_param["level_dim"]
        # 最低的哈希表分辨率，INGP论文中的N_min=16
        self.base_resolution = self.sys_param['base_resolution']
        # 哈希表每一层特征长度对2的对数，用于恢复特征长度，用于计算论文中的T=[21]
        self.log2_hashmap_size = self.sys_param["log2_hashmap_size"]
        # 每一层间哈希表格的分辨率倍数=2
        self.level_scale = self.sys_param['level_scale']

        # 经过哈希编码后的输出维度
        self.out_dim = self.num_level * self.level_dim
        # 每层哈希表提取特征的长度，论文中的T
        self.max_length = 2 ** self.log2_hashmap_size
        # 获取后端的加速计算方法，自己写的CUDA代码
        # CUDA写的 .cu/.cpp插件代码会被编译成一个Python可加载的.so文件
        self.backend = CUDA_Backend()
        # 自定义计算过程, 注意这个apply必须添加，是由继承的Function决定的使用方法
        self.grid_encode = Grid_Custom_Flow().apply
        
        # 获取哈希网格的一些整理信息
        # offsets：所有层网格的哈希表尺寸累加和
        # idx：和offsets一样的尺寸，每个数据表示当前位置是哪一层表格
        # resolutions：每一层网格的分辨率
        # self.embeddings: 哈希表tensor为可学习的参数，大小为[哈希表总长度，特征维度]
        offsets, idx, resolutions, self.embeddings = self.get_grid_info()
        # print(self.embeddings, "mnmmnnmnmnmnm")

        # 为上面的参数注册一个缓冲区，缓冲区的作用就是
        # 当我们调用 model.cuda() 或者 model.to('cuda') 将模型移动到 GPU 上时，
        # 这个张量也会自动被移到GPU上，而不需要手动处理
        self.register_buffer('offsets', offsets)
        self.register_buffer('idx', idx)
        self.register_buffer('grid_sizes', torch.tensor(resolutions, dtype=torch.int32))

    # 输入需要是[-1,1]的范围
    # input_means：[batch, 1, 1, N, 6, 3]
    def forward(self, input_means):
        # 保存一下输入数据的维度，后面需要还原这个尺寸
        # 最后一个维度不用保留，因为经过MLP会变
        ori_shape = input_means.shape[:-1]
        # 转换到0~1
        # print("!!!!!!!!!!!!!!xyz_max: ", input_means.max().item())
        # print("!!!!!!!!!!!!!!xyz_min: ", input_means.min().item())
        inputs = (input_means + 1)/2.0
        # print("!!!!!!!!!!!!!!xyz_max: ", inputs.max().item())
        # print("!!!!!!!!!!!!!!xyz_min: ", inputs.min().item())
        # 将尺寸展平
        inputs = inputs.reshape(-1, 3)
        # 输入编码器, 包含了后端
        outputs = self.grid_encode(self.backend, inputs, self.embeddings, self.offsets,
                                   self.num_level, self.level_scale, self.base_resolution)
        # 将输出还原尺寸
        outputs = outputs.reshape(ori_shape + (self.out_dim,))
        return outputs

    # 根据给定参数计算哈希网格的一些数值
    def get_grid_info(self):
        # 每一层的分辨率
        resolutions = []
        # 每一层哈希表之前的累加尺寸和 [哈希表层数+1]
        offsets = []
        # 当前哈希表的尺寸
        offset = 0
        # 对每层哈希表进行处理计算
        for i in range(self.num_level):
            # 当前层网格的分辨率 16 x 2 ^ i
            resolution = int(np.ceil(self.base_resolution * self.level_scale ** i))
            # 限制网格特征的长度限制到最大长度
            params_in_level = min(self.max_length, resolution ** self.input_dim) 
            # 哈希表的尺寸，调整一下尺寸，让他能够被8整除
            params_in_level = int(np.ceil(params_in_level / 8) * 8)                 
            # 存一下信息
            # resolutions存的是每一层网格的分辨率
            resolutions.append(resolution)
            # offsets存的是之前所有层网格的哈希表尺寸和（论文中是T）
            # 有点难理解需要看下面idx的含义
            # 例如：tensor([       0,     4920,    40864,   315496,  2412648,  4509800,  6606952, 
            # 8704104, 10801256, 12898408, 14995560], dtype=torch.int32)
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        # 转换成tensor
        offsets = torch.tensor(offsets, dtype=torch.int32)
        # 注意，这个idx是offset大小，就是最高分辨率的那层表格对应的
        # 哈希表大小，不是offsets(注意结尾的s)
        idx = torch.empty(offset, dtype=torch.long)
        # 对哈整个大的哈希表进行分块，每一块代表一层，并对每一块进行id赋值
        for i in range(self.num_level):
            idx[offsets[i]:offsets[i+1]] = i
        # level_dim 表示哈希表的的特征宽度（论文中是F）
        # self.n_params表示总的哈希表参数量
        self.n_params = offsets[-1] * self.level_dim
        # 注册哈希表tensor为可学习的参数，大小为[哈希表总长度，特征维度]
        embeddings = torch.nn.Parameter(torch.empty(offset, self.level_dim))
        # 初始化0.001
        embeddings.data.uniform_(-0.01, 0.01)

        return offsets, idx, resolutions, embeddings

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
    
# NeRF MLP小网络块
class NeRF_MLP2(torch.nn.Module):
    def __init__(self, sys_param):
        super(NeRF_MLP2, self).__init__()
        ### 初始化参数
        self.sys_param = sys_param
        self.MLP_depth = sys_param['MLP_depth']
        self.MLP_width = sys_param['MLP_width']
        self.MLP_skips = sys_param['MLP_skips']
        self.num_level = self.sys_param['num_level']
        self.level_dim = self.sys_param['level_dim']
        self.hash_feature_dim = self.num_level * self.level_dim
        self.view_dim = self.sys_param['freqs_view'] * 2 * 3 + 3

        #### 构建MLP网络
        for i in range(self.MLP_depth):
            # 第一层
            if i == 0:
                layer = torch.nn.Linear(self.hash_feature_dim, self.MLP_width)
            # 跳跃连接层
            elif i in self.MLP_skips:
                layer = torch.nn.Linear(self.MLP_width + self.hash_feature_dim, self.MLP_width)
            # 其他隐藏层
            else:
                layer = torch.nn.Linear(self.MLP_width, self.MLP_width)
            # 初始化并且例化网络
            torch.nn.init.kaiming_uniform_(layer.weight)
            layer = torch.nn.Sequential(layer, torch.nn.LeakyReLU(inplace=True)) 
            setattr(self, 'layer_{}'.format(i), layer)
        # 输出层
        # self.rgb_layer = torch.nn.Sequential(torch.nn.Linear(self.MLP_width, 3),
        #                                      torch.nn.Sigmoid())
        # self.sigma_layer = torch.nn.Linear(self.MLP_width, 1)
        self.sigma_layer = torch.nn.Sequential(torch.nn.Linear(self.MLP_width, self.MLP_width),
                                               torch.nn.LeakyReLU(inplace=True),
                                               torch.nn.Linear(self.MLP_width, 1))
        self.feature_layer = torch.nn.Linear(self.MLP_width, self.MLP_width)
        self.rgb_layer = torch.nn.Sequential(torch.nn.Linear(self.MLP_width + self.view_dim, self.MLP_width // 2),
                                             torch.nn.ReLU(True),
                                             torch.nn.Linear(self.MLP_width // 2, 3),
                                             torch.nn.Sigmoid())
    
    def forward(self, view_dirs, features):
        # xyz_view = torch.cat([view_dirs, features], dim=-1)
        # x = xyz_view
        x = features
        # MLP隐藏层
        for i in range(self.MLP_depth):
            layer = getattr(self, 'layer_{}'.format(i))
            # 跳跃连接
            if i in self.MLP_skips:
                x = torch.cat([x, features], dim=-1)
            x = layer(x)
        # 输出层
        feature = self.feature_layer(x)                              # [B, W]
        rgb = self.rgb_layer(torch.cat([feature, view_dirs], dim=-1))
        sigma = self.sigma_layer(x)
        output = torch.cat([rgb, sigma], dim=-1)                            # [B, 4]
        return output

# NeRF MLP小网络块
class NeRF_MLP(torch.nn.Module):
    def __init__(self, sys_param, MLP_type="coarse"):
        super(NeRF_MLP, self).__init__()
        ### 初始化参数
        self.sys_param = sys_param
        if MLP_type == "coarse":
            self.MLP_depth = sys_param['coarse_MLP_depth']
            self.MLP_width = sys_param['coarse_MLP_width']
            self.MLP_skips = sys_param['coarse_MLP_skips']
        elif MLP_type == "fine":
            self.MLP_depth = sys_param['fine_MLP_depth']
            self.MLP_width = sys_param['fine_MLP_width']
            self.MLP_skips = sys_param['fine_MLP_skips']
        self.num_level = self.sys_param['num_level']
        self.level_dim = self.sys_param['level_dim']
        self.hash_feature_dim = self.num_level * self.level_dim
        self.view_dim = self.sys_param['freqs_view'] * 2 * 3 + 3

        #### 构建MLP网络
        for i in range(self.MLP_depth):
            # 第一层
            if i == 0:
                layer = torch.nn.Linear(self.hash_feature_dim + self.view_dim, self.MLP_width)
            # 跳跃连接层
            elif i in self.MLP_skips:
                layer = torch.nn.Linear(self.MLP_width + self.hash_feature_dim + self.view_dim, self.MLP_width)
            # 其他隐藏层
            else:
                layer = torch.nn.Linear(self.MLP_width, self.MLP_width)
            # 初始化并且例化网络
            torch.nn.init.kaiming_uniform_(layer.weight)
            layer = torch.nn.Sequential(layer, torch.nn.LeakyReLU(inplace=True)) 
            # layer = torch.nn.Sequential(layer, torch.nn.ReLU(True))
            setattr(self, 'layer_{}'.format(i), layer)
        # 输出层
        self.rgb_layer = torch.nn.Sequential(torch.nn.Linear(self.MLP_width, 3),
                                             torch.nn.Sigmoid())
        self.sigma_layer = torch.nn.Linear(self.MLP_width, 1)
    
    def forward(self, view_dirs, features):
        xyz_view = torch.cat([view_dirs, features], dim=-1)
        x = xyz_view
        # MLP隐藏层
        for i in range(self.MLP_depth):
            layer = getattr(self, 'layer_{}'.format(i))
            # 跳跃连接
            if i in self.MLP_skips:
                x = torch.cat([x, xyz_view], dim=-1)
            x = layer(x)
        # 输出层
        rgb = self.rgb_layer(x)
        sigma = self.sigma_layer(x)
        output = torch.cat([rgb, sigma], dim=-1)                            # [B, 4]
        return output
