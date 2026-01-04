import torch 
import torch.nn.functional as F
import numpy as np
import math
from .cuda.B_cuda_setup import Grid_cuda_tools

# 自己定义一个前向和反向传播的计算过程
class Grid_Custom_Flow(torch.autograd.Function):
    # 继承自Function，必须定义，类似于写模型nn.model必须要定义forward一样
    # 参数说明：(ctx具体是啥自己查pytorch文档)
    # in_means: 输入的数据，待进行编码的采样点坐标, 必须要是在 0-1 的 float32 类型 [N, 3]
    # in_embds: 哈希表的学习参数，float32 类型 [哈希表总长度，特征维度]
    # in_offst: 每一层哈希表之前的累加尺寸和 [哈希表层数+1]
    # in_level: 哈希表的层数
    # in_scale: 每一层间哈希表格的分辨率倍数
    # in_baser: 最低的哈希表分辨率
    @staticmethod
    def forward(ctx, in_means, in_embds, in_offst, in_level, in_scale, in_baser):
        # 输入合并后的总量大小
        in_batch = in_means.shape[0]
        # 输入数据的维度
        in_dimen = in_means.shape[1]
        # 哈希特征编码后的长度
        ft_dimen = in_embds.shape[1]
        # 取log在原文中说是方便cuda计算
        lg_scale = math.log2(in_scale)
        # 转成内存连续的变量
        in_means = in_means.contiguous()
        # 定义函数插值的方式
        in_intrp = 0    # 线性插值
        # in_intrp = 1  # 平滑权重插值
        # 定义是否对齐角点
        in_align = True
        # 定义一个占位变量，承接输出数据
        # 注意，这里必须初始化是0，不能改成其他数值，因为cuda代码里有计算逻辑用到全是0
        ot_encod = torch.zeros([in_level, in_batch, ft_dimen], device = in_means.device, dtype = in_embds.dtype)
        # 输入如果需要梯度
        # 就初始化一个 输出特征对输入坐标的导数 容器
        if in_means.requires_grad:
            # 注意，这里必须初始化是0，不能改成其他数值，因为cuda代码里有计算逻辑用到全是0
            in_jacob = torch.zeros([in_batch, in_level * in_dimen * ft_dimen], device = in_means.device, dtype = in_embds.dtype)
        else:
            in_jacob = None            
        # 同步一下进程，等待所有GPU中的任务完毕
        torch.cuda.synchronize()
        # 调用API进行计算
        Grid_cuda_tools.grid_encode_forward(in_means,   # 输入采样点位置
                                            in_embds,   # 输入哈希表可学习参数
                                            in_offst,   # 输入每一层在哈希表里的偏移 
                                            in_batch,   # 输入采样点数量B 
                                            in_dimen,   # 输入采样点维度D
                                            ft_dimen,   # 输入哈希特征维度C
                                            in_level,   # 输入哈希表格层数L
                                            in_baser,   # 输入哈希表最低层分辨率H
                                            lg_scale,   # 输入哈希表层级缩放率S
                                            in_intrp,   # 输入插值方式，0是线性，1是平滑
                                            in_align,   # 输入是否对齐角点
                                            ot_encod,   # 输出编码后的新特征
                                            in_jacob)   # 输出哈希特征对位置的梯度（可能没有）
    
        # 将输出数据的尺寸转换回原来的尺寸
        ot_encod = ot_encod.permute(1, 0, 2).reshape(in_batch, in_level*ft_dimen)
        # 保存属性，与backward交互使用
        # 是PyTorch为实现自定义前向和后向传播提供的一种机制。通过这种方式，
        # 你可以在forward方法中计算并保存那些对于梯度计算（即backward方法）来说重要的数据和状态。
        # ctx.backend = backend
        ctx.save_for_backward(in_means, in_embds, in_offst, in_jacob)
        # gridtype = 0 表示哈希（-2）
        # interpolation = 0 表示线性插值(-1)
        ctx.in_tempd = [in_batch, in_dimen, ft_dimen, in_level, lg_scale, in_baser]
        ctx.in_align = in_align
        ctx.in_intrp = in_intrp
        return ot_encod
    
    @staticmethod
    def backward(ctx, in_grads):
        # 承接各种输出
        in_means, in_embds, in_offst, in_jacob = ctx.saved_tensors
        in_batch, in_dimen, ft_dimen, in_level, lg_scale, in_baser = ctx.in_tempd
        in_align = ctx.in_align
        in_intrp = ctx.in_intrp
        # grad: [in_level, in_level*ft_dimen] --> [in_level, in_batch, ft_dimen]
        in_grads = in_grads.reshape(in_batch, in_level, ft_dimen).permute(1, 0, 2).contiguous()
        # 承接的占位符
        gd_embds = torch.zeros_like(in_embds)
        # 承接的占位符  
        if in_jacob is not None:
            # 注意，这里必须初始化是0，不能改成其他数值，因为cuda代码里有计算逻辑用到全是0
            gd_means = torch.zeros_like(in_means, dtype = in_embds.dtype)
        else:
            gd_means = None
        # 同步一下进程，等待所有GPU中的任务完毕
        # backend.synchronize()
        torch.cuda.synchronize()
        # cuda反向传播计算
        Grid_cuda_tools.grid_encode_backward(in_grads,      # 从反向端传过来的梯度
                                             in_means,      # 输入采样点位置
                                             in_offst,      # 输入每一层在哈希表里的偏移
                                             in_batch,      # 输入采样点数量B
                                             in_dimen,      # 输入采样点维度D
                                             ft_dimen,      # 输入哈希特征维度C
                                             in_level,      # 输入哈希表格层数L
                                             in_baser,      # 输入哈希表最低层分辨率H
                                             lg_scale,      # 输入哈希表层级缩放率S
                                             in_intrp,      # 输入插值方式，0是线性，1是平滑
                                             in_align,      # 输入是否对齐角点
                                             gd_embds,      # 输出到哈希表的梯度
                                             in_jacob,      # 哈希特征对位置的梯度（可能没有）
                                             gd_means)      # 输出到采样点位置的梯度（可能没有） 

        if in_jacob is not None:
            gd_means = gd_means.to(in_jacob.dtype)

        return gd_means, gd_embds, None, None, None, None, None, None, None

# 哈希编码
class GridEncoder(torch.nn.Module):
    def __init__(self, sys_param, lv, prop=False):
        print("Initializing Grid Encoder...")
        super(GridEncoder, self).__init__()
        #################### 参数始化 ######################
        self.sys_param = sys_param
        # 输入尺寸,xyz三项
        self.input_dim = 3
        # prop阶段的哈希表参数
        if prop:
            # 哈希表层数，INGP论文中的L=8
            self.num_level = self.sys_param["prop_num_level"][lv]
            # 哈希表每一层特征宽度，INGP论文中的F=2
            self.level_dim = self.sys_param["prop_level_dim"][lv]
            # 最低的哈希表分辨率，INGP论文中的N_min=16
            self.base_resolution = self.sys_param['prop_base_resolution'][lv]
            # 哈希表每一层特征长度对2的对数，用于恢复特征长度，用于计算论文中的T=[21]
            self.log2_hashmap_size = self.sys_param["prop_log2_hashmap_size"][lv]
            # 每一层间哈希表格的分辨率倍数=2
            self.level_scale = self.sys_param['prop_level_scale'][lv]
        # NeRF阶段的哈希表参数
        else:
            # 哈希表层数，INGP论文中的L=10
            self.num_level = self.sys_param["render_num_level"][lv]
            # 哈希表每一层特征宽度，INGP论文中的F=4
            self.level_dim = self.sys_param["render_level_dim"][lv]
            # 最低的哈希表分辨率，INGP论文中的N_min=16
            self.base_resolution = self.sys_param['render_base_resolution'][lv]
            # 哈希表每一层特征长度对2的对数，用于恢复特征长度，用于计算论文中的T=[21]
            self.log2_hashmap_size = self.sys_param["render_log2_hashmap_size"][lv]
            # 每一层间哈希表格的分辨率倍数=2
            self.level_scale = self.sys_param['render_level_scale'][lv]

        # 经过哈希编码后的输出维度
        self.out_dim = self.num_level * self.level_dim
        # 每层哈希表提取特征的长度，论文中的T
        self.max_length = 2 ** self.log2_hashmap_size
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
        inputs = (input_means + 1)/2.0
        # 将尺寸展平
        inputs = inputs.reshape(-1, 3)
        # 输入编码器, 包含了后端
        outputs = self.grid_encode(inputs, self.embeddings, self.offsets,
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

# 正余弦编码
class SinCosEmbedding(torch.nn.Module):
    def __init__(self, sys_param, freqs):
        print("Initializing SinCos Embedding...")
        super(SinCosEmbedding, self).__init__()
        self.device = sys_param['device']
        # 输入维度
        self.input_dim = 3
        # 输出维度
        self.output_dim = self.input_dim * freqs * 2 + self.input_dim
        # 编码维度
        self.freqs = 2 ** torch.arange(freqs, dtype=torch.float32, device=self.device)     # [F]
        
    def forward(self, x):                                               # [B, 1, 1, 3]            
        batch_size = x.shape[:-1]                                       # [B, 1, 1]
        x_freqs = x[..., None] * self.freqs                             # [B, 1, 1, 3, F]
        x_sin = torch.sin(x_freqs)                                      # [B, 1, 1, 3, F]
        x_cos = torch.cos(x_freqs)                                      # [B, 1, 1, 3, F]
        x_freqs_sincos = torch.stack([x_sin, x_cos], dim=-2)            # [B, 1, 1, 3, 2, F]
        x_freqs_sincos = x_freqs_sincos.reshape(batch_size + (-1,))     # [B, 1, 1, 3*2*F]
        x_emb = torch.cat([x, x_freqs_sincos], dim=-1)                  # [B, 1, 1, 3 + 3*2*F]
        return x_emb
    
# NeRF MLP小网络块
class NeRF_MLP(torch.nn.Module):
    def __init__(self, sys_param):
        print("Initializing NeRF MLP...")
        super(NeRF_MLP, self).__init__()
        ### 初始化参数
        self.sys_param = sys_param
        self.MLP_depth = sys_param['render_MLP_depth'][0]
        self.MLP_width = sys_param['render_MLP_width'][0]
        self.MLP_skips = sys_param['render_MLP_skips'][0]
        self.num_level = self.sys_param['render_num_level'][0]
        self.level_dim = self.sys_param['render_level_dim'][0]
        self.hash_feature_dim = self.num_level * self.level_dim
        self.sigma_bias = 1.0
        self.view_dim = self.sys_param['render_freqs_view'] * 2 * 3 + 3

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
            self.register_module(f"rgb_layer_{i}", layer)
        # 输出层
        self.rgb_layer = torch.nn.Sequential(torch.nn.Linear(self.MLP_width, 3),
                                             torch.nn.Sigmoid())
        self.sigma_layer = torch.nn.Linear(self.MLP_width, 1)
    
    def forward(self, view_dirs, features):
        xyz_view = torch.cat([view_dirs, features], dim=-1)
        x = xyz_view
        # MLP隐藏层
        for i in range(self.MLP_depth):
            # 跳跃连接
            if i in self.MLP_skips:
                x = torch.cat([x, xyz_view], dim=-1)
            x = self.get_submodule(f"rgb_layer_{i}")(x)
            x = F.leaky_relu(x)
        # 输出层
        rgb = self.rgb_layer(x)
        sigma = self.sigma_layer(x)
        return sigma, rgb

class Prop_MLP(torch.nn.Module):
    def __init__(self, sys_param, lv):
        print("Initializing prop MLP...")
        super(Prop_MLP, self).__init__()
        ### 初始化参数
        self.sys_param = sys_param
        self.MLP_width = self.sys_param['prop_MLP_width'][lv]
        self.num_level = self.sys_param['prop_num_level'][lv]
        self.level_dim = self.sys_param['prop_level_dim'][lv]
        self.hash_feature_dim = self.num_level * self.level_dim
        self.sigma_bias = 1.0
        #### 体密度预测层
        self.sigma_layer = torch.nn.Sequential(torch.nn.Linear(self.hash_feature_dim, self.MLP_width),
                                                torch.nn.LeakyReLU(inplace=True),
                                                torch.nn.Linear(self.MLP_width, 1))
    
    def forward(self, features):
        sigma = self.sigma_layer(features)[..., 0] + self.sigma_bias
        sigma = F.softplus(sigma)
        return sigma