#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <algorithm>
#include <stdexcept>
#include <stdint.h>
#include <cstdio>
#include "hash_enc_bd.h"

// 向上取整的除法
template <typename T>
__host__ __device__ inline T div_round_up(T val, T divisor) 
{
    return (val + divisor - 1) / divisor;
}

// 平滑插值函数，左侧插值权重的定义是 3t^2 - 2t^3
// t 在 0-1 之间的位置
template <typename T>
__device__ inline T smoothstep(T val) 
{
    return val*val*(3.0f - 2.0f * val);
}

// 把整数格点坐标通过 “乘不同常数 + XOR 混合” 压成一个 32-bit 整数，用来当哈希表索引。
template <uint32_t in_dimen>
__device__ uint32_t fast_hash(const uint32_t pos_grid[in_dimen]) {
    
    // coherent type of hashing
    // 哈希常数，这个选择有讲究，不能随便给
    constexpr uint32_t primes[7] = {1u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u};
    // 输出数据容器，一个32位的哈希值
    uint32_t result = 0;
    #pragma unroll
    for (uint32_t i = 0; i < in_dimen; ++i)
    {
        // hash = (x⋅p0​)⊕(y⋅p1​)⊕(z⋅p2​)
        // ^= 表示按位异或操作
        result ^= pos_grid[i] * primes[i];
    }

    return result;
}

// 将计算出来的网格点映射成哈希表的索引，细节到特征的第几位，一般取0，表示起始位置
// 如果网格的索引没有超过哈希表的尺寸，就直接使用线性网格的索引过程
// 如果网格的索引超过哈希表的尺寸，就使用哈希函数来计算索引
template <uint32_t in_dimen, uint32_t ft_dimen>
__device__ uint32_t get_grid_index(
    const bool     in_align,            // 是否对齐角点
    const uint32_t ft_ele_idx,          // 特征维度内的第几个特征索引，如果是0，就是当前哈希特征的起始位置
    const uint32_t hashmap_size,        // 当前层哈希表的尺寸
    const uint32_t resolution,          // 当前层的分辨率
    const uint32_t pos_grid[in_dimen])  // 当前坐标所在网格的某角点索引
{
    uint32_t stride = 1;
    uint32_t index = 0;

    #pragma unroll
    // (stride <= hashmap_size) 这一条件是为了防止 stride 过大，导致 index 计算错误
    // 如果 stride 超过 hashmap_size, 说明当前层哈希表已经装不下这么多格点了
    // 这种情况下就要切换到哈希函数计算索引
    for (uint32_t d = 0; (d < in_dimen) && (stride <= hashmap_size); d++)
    {
        // 按照维度展平计算索引，d=0 代表x维度，d=1 代表y维度，d=2 代表z维度
        // 以二维为例子进行一下说明，比如二维的坐标是 [10, 15]
        // d=0, index = 0 + 10*1 = 10, stride = 1 * resolution = 16
        // d=1, index = 10 + 15*16 = 250, stride = 16 * resolution = 256
        // 和我们正常将某个[x,y]坐标在图像层面flatten展平成一维是一个原理，三维就是这种情况的推广
        index += pos_grid[d] * stride;
        stride *= in_align ? resolution: (resolution + 1);
    }
    // 注意这个逻辑，判定用stride，而不用index的原因:
    // stride在上面的循环后，其最终的结果等于当前网格的总点数，比如上面二维的例子，stride最后等于256=16^2
    // 注意，在上面循环的判定条件中，有一个 (stride <= hashmap_size)，如果网格的总数大于哈希表当前层的容量，这个循环会停止
    // 此时，index还没有计算完毕，会损失很多信息。
    // 所以如果index没有计算完毕，就使用下面的哈希函数重新获得index索引数值
    if (stride > hashmap_size) 
    {
        // 通过哈希得到索引，不可避免的会出现一些碰撞
        index = fast_hash<ft_dimen>(pos_grid);
    }
    // 因为哈希表格在存储的时候是一块连续的一维内存
    // (index % hashmap_size) 表示在哈希表的第几行
    // 乘以 ft_dimen 表示每一行有 ft_dimen 个特征值，转换为每个特征起始位置的数组的索引
    // ft_ele_idx 表示拿到第几个特征
    return (index % hashmap_size) * ft_dimen + ft_ele_idx;
}

// 计算网格的反传梯度
// 模板函数中 hd_dimen 表示每个线程处理几个哈希特征的梯度
// hd_dimen在ft_dimen=1的时候取1，其他情况都是2，仅支持 ft_dimen = 1, 2, 4, 8
template <uint32_t in_dimen, uint32_t ft_dimen, uint32_t hd_dimen>
__global__ void kernel_grid_bd(
    const float* __restrict__ in_grads,
    const float* __restrict__ in_means,
    const int*   __restrict__ in_offst,
    const uint32_t in_batch, 
    const uint32_t in_level,
    const uint32_t in_baser, 
    const float    lg_scale,
    const uint32_t in_intrp,
    const bool     in_align,
    float* __restrict__ gd_embds
)
{
    // 在launch总线程的时候，总线程数应该是 in_batch * ft_dimen / hd_dimen
    // 如果想让idx恢复成只代表空间点in_batch的数量，就需要再乘以 hd_dimen / ft_dimen
    // idx 表示当前线程落在哪一个具体的样本里
    const uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * hd_dimen / ft_dimen;
    if (idx >= in_batch) return;
    // 当前线程负责处理具体哪一层哈希表的坐标点
    const uint32_t cur_level = blockIdx.y;
    // 当前线程需要处理的输出特征里负责部分的起始地址，也就是这个样本里从哪个通道开始计算
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * hd_dimen - idx * ft_dimen;

    // 将当前线程的指针，指向该线程负责层 cur_level 的对应梯度参数块的起始位置（移动指针地址）
    gd_embds += in_offst[cur_level] * ft_dimen;
    // 将当前线程的指针，指向输入坐标的第 idx 个坐标起始位置，这个坐标就是当前线程负责处理的坐标
    in_means += idx * in_dimen;
    // 将当前线程的指针，跳到上游梯度张量 in_grads 中，当前层 cur_level、当前样本 idx、当前线程负责的起始通道 ch 位置
    // in_grads 的尺寸是 [in_level, in_batch, ft_dimen]
    in_grads += cur_level * in_batch * ft_dimen + idx * ft_dimen + ch;
    // 当前层level哈希网格的尺寸
    const uint32_t hashmap_size = in_offst[cur_level + 1] - in_offst[cur_level];
    // exp2f(cur_level * lg_scale) 计算当前层相比较于基础层的缩放比例
    // scale 近似等于这一层网格最大的格点尺寸，最后的 - 1.0f 是因为索引从0开始
    // 例如，0层，scale就是15, 2层scale就是31...
    const float scale = exp2f(cur_level * lg_scale) * in_baser - 1.0f;
    // 拿到当前层的分辨率
    const uint32_t resolution = (uint32_t)ceil(scale) + 1;

    // 输入坐标数据的越界检查，必须要在 0-1 之间
    #pragma unroll
    for (uint32_t d = 0; d < in_dimen; d++) {
        if (in_means[d] < 0 || in_means[d] > 1)
            // 因为梯度的容器初始化全是数值0，所以这里可以直接返回
            return;
    }

    // 初始化插值权重相关的容器数组
    float pos[in_dimen];
    uint32_t pos_grid[in_dimen];

    #pragma unroll
    for (uint32_t d = 0; d < in_dimen; d++) 
    {
        // 将归一化的坐标恢复到当前网格下的具体位置
        pos[d] = in_means[d] * scale + (in_align ? 0.0f : 0.5f);
        // floorf 表示向下取整, pos_grid[d] 拿到的是当前坐标所在网格的最左下角坐标索引（三维空间里就是所有索引最小的角点）
        pos_grid[d] = floorf(pos[d]);
        // 计算相对于格点的偏移
        pos[d] -= (float)pos_grid[d];
        // 如果使用平滑插值，进行平滑插值的计算，如果使用线性插值，就直接保持pos[d]原数值就行
        if (in_intrp == 1)
        {
            pos[d] = smoothstep(pos[d]);
        }
    }

    // 初始化一个缓存当前线程处理特征的梯度容器
    float grad_cur[hd_dimen] = {0}; // fetch to register

    #pragma unroll
    for (uint32_t c = 0; c < hd_dimen; c++)
    {
        // 把对应位置的梯度拿出来存到缓存里
        grad_cur[c] = in_grads[c];
    }

    #pragma unroll
    // 插值计算，循环8次，每次取一个包围框的角点
    for (uint32_t ii = 0; ii < (1 << in_dimen); ii++)
    {
        
        float w = 1;
        // 当前循环轮数拿到角点坐标的缓存容器
        uint32_t pos_grid_local[in_dimen];

        #pragma unroll
        // 在前向里面说过了，拿到当前循环具体处理的角点坐标
        // 拿到坐标的同时处理角点的插值权重
        for (uint32_t d = 0; d < in_dimen; d++)
        {
            if ((ii & (1 << d)) == 0)
            {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } 
            else
            {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        // 拿到选定网格对应位置的特征元素索引
        uint32_t index = get_grid_index<in_dimen, ft_dimen>(in_align, ch, hashmap_size, resolution, pos_grid_local);

        // 将梯度累加到对应的输出中
        #pragma unroll
        for (uint32_t c = 0; c < hd_dimen; c++)
        {
            // 其实反向梯度核心就这一个公式，in_grads是后端计算（MLP）传回来的梯度，是dLoss/dy，表示对输出特征的梯度
            // grad_cur 是把in_grads中与当前线程匹配的梯度拿了出来
            // gd_embds 是对哈希表参数的梯度，直接用来更新哈希网格数据，是dL/dembds
            // 两者的关系是在前向中是：y = ∑ w * embds[index(corner)]
            // 所以反向关系就是：dL/dembds += w * dL/dy 
            atomicAdd(&gd_embds[index + c], (float)(w) * grad_cur[c]);
        }
    }    
}

// 计算输入位置的反传梯度
// 总的线程数量就是 in_batch*in_dimen 数量（向上取整）
template <uint32_t in_dimen, uint32_t ft_dimen>
__global__ void kernel_means_bd(
    const float* __restrict__ in_grads,     // 从反向端传过来的梯度
    const float* __restrict__ in_jacob,     // 哈希特征对位置的梯度
    uint32_t in_batch,                      // 输入采样点数量B
    uint32_t in_level,                      // 输入哈希表格层数L
    float* __restrict__ gd_means            // 输出到采样点位置的梯度
) 
{
    // 获取当前线程的id，如果超过总的需要处理的数据量，就直接返回
    const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= in_batch * in_dimen) return;

    // 把线程id映射回需要处理的数据索引
    // b 表示具体需要处理哪个坐标点
    // d 表示具体处理这个点的哪个维度
    const uint32_t b = idx / in_dimen;
    const uint32_t d = idx - b * in_dimen;
    // 调整雅可比矩阵的指针，将指针指向当前线程需要处理的数据位置
    in_jacob += b * in_level * in_dimen * ft_dimen;
    // 定义一个当前线程缓存结果的容器
    float result = 0;
    // 计算梯度
    # pragma unroll
    // 对所有 level 和 channel 做链式法则求和       
    for (int l = 0; l < in_level; l++)
    {
        # pragma unroll
        for (int ch = 0; ch < ft_dimen; ch++)
        {
            // dL/dmeans = in_grads * in_jacob
            result += in_grads[l * in_batch * ft_dimen + b * ft_dimen + ch] * in_jacob[l * in_dimen * ft_dimen + d * ft_dimen + ch];
        }
    }

    gd_means[idx] = result;
}


// 反向传播函数
void grid_encode_backward_cuda(
    const float*   in_grads,    // 从反向端传过来的梯度
    const float*   in_means,    // 输入采样点位置
    const int*     in_offst,    // 输入每一层在哈希表里的偏移
    const uint32_t in_batch,    // 输入采样点数量B
    const uint32_t in_dimen,    // 输入采样点维度D
    const uint32_t ft_dimen,    // 输入哈希特征维度C
    const uint32_t in_level,    // 输入哈希表格层数L
    const uint32_t in_baser,    // 输入哈希表最低层分辨率H
    const float    lg_scale,    // 输入哈希表层级缩放率S 
    const uint32_t in_intrp,    // 输入插值方式，0是线性，1是平滑
    const bool     in_align,    // 输入是否对齐角点
    float*         gd_embds,    // 输出到哈希表的梯度
    float*         in_jacob,    // 哈希特征对位置的梯度（可能没有）
    float*         gd_means     // 输出到采样点位置的梯度（可能没有） 
)
{
    // 每个 block 512 个线程，处理 512 个输入坐标
    static constexpr uint32_t N_THREAD = 512;
    // 每个线程串行处理特征维度的数量（2u代表无符号数）
    const uint32_t hd_dimen = std::min(2u, ft_dimen);
    // 总的线程数量是 in_batch * ft_dimen，每个线程处理hd_dimen个特征维度，所以
    const dim3 blocks_hashgrid = {div_round_up(in_batch * ft_dimen / hd_dimen, N_THREAD), in_level, 1};

    // 仅支持 in_dimen = 2, 3
    // 仅支持 ft_dimen = 1, 2, 4, 8
    // 注意，用实例 switch 的原因是 template <uint32_t in_dimen, uint32_t ft_dimen> 模版调用的时候必须是编译期常量
    switch (in_dimen)
    {
        // 2D 实验情况，输入坐标是2维
        case 2:
            switch (ft_dimen)
            {   
                // 计算相对于网格数据的梯度
                case 1: 
                    kernel_grid_bd<2, 1, 1><<<blocks_hashgrid, N_THREAD>>>(in_grads, in_means, in_offst, in_batch, in_level, in_baser, lg_scale, in_intrp, in_align, gd_embds); 
                    // 如果需要输出对输入位置xyz的梯度，就计算该项
                    if (in_jacob) kernel_means_bd<2, 1><<<div_round_up(in_batch * in_dimen, N_THREAD), N_THREAD>>>(in_grads, in_jacob, in_batch, in_level, gd_means);
                    break;
                case 2: 
                    kernel_grid_bd<2, 2, 2><<<blocks_hashgrid, N_THREAD>>>(in_grads, in_means, in_offst, in_batch, in_level, in_baser, lg_scale, in_intrp, in_align, gd_embds); 
                    if (in_jacob) kernel_means_bd<2, 2><<<div_round_up(in_batch * in_dimen, N_THREAD), N_THREAD>>>(in_grads, in_jacob, in_batch, in_level, gd_means);
                    break;
                case 4: 
                    kernel_grid_bd<2, 4, 2><<<blocks_hashgrid, N_THREAD>>>(in_grads, in_means, in_offst, in_batch, in_level, in_baser, lg_scale, in_intrp, in_align, gd_embds); 
                    if (in_jacob) kernel_means_bd<2, 4><<<div_round_up(in_batch * in_dimen, N_THREAD), N_THREAD>>>(in_grads, in_jacob, in_batch, in_level, gd_means);
                    break;
                case 8: 
                    kernel_grid_bd<2, 8, 2><<<blocks_hashgrid, N_THREAD>>>(in_grads, in_means, in_offst, in_batch, in_level, in_baser, lg_scale, in_intrp, in_align, gd_embds); 
                    if (in_jacob) kernel_means_bd<2, 8><<<div_round_up(in_batch * in_dimen, N_THREAD), N_THREAD>>>(in_grads, in_jacob, in_batch, in_level, gd_means);
                    break;
                default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
            } break;
        // 3D 实验情况，输入直接是三维空间点
        case 3:
            switch (ft_dimen)
            {
                // 计算相对于网格数据的梯度
                case 1: 
                    kernel_grid_bd<3, 1, 1><<<blocks_hashgrid, N_THREAD>>>(in_grads, in_means, in_offst, in_batch, in_level, in_baser, lg_scale, in_intrp, in_align, gd_embds); 
                    if (in_jacob) kernel_means_bd<3, 1><<<div_round_up(in_batch * in_dimen, N_THREAD), N_THREAD>>>(in_grads, in_jacob, in_batch, in_level, gd_means);
                    break;
                case 2: 
                    kernel_grid_bd<3, 2, 2><<<blocks_hashgrid, N_THREAD>>>(in_grads, in_means, in_offst, in_batch, in_level, in_baser, lg_scale, in_intrp, in_align, gd_embds);
                    if (in_jacob) kernel_means_bd<3, 2><<<div_round_up(in_batch * in_dimen, N_THREAD), N_THREAD>>>(in_grads, in_jacob, in_batch, in_level, gd_means);
                    break;
                case 4: 
                    kernel_grid_bd<3, 4, 2><<<blocks_hashgrid, N_THREAD>>>(in_grads, in_means, in_offst, in_batch, in_level, in_baser, lg_scale, in_intrp, in_align, gd_embds); 
                    if (in_jacob) kernel_means_bd<3, 4><<<div_round_up(in_batch * in_dimen, N_THREAD), N_THREAD>>>(in_grads, in_jacob, in_batch, in_level, gd_means);
                    break;
                case 8: 
                    kernel_grid_bd<3, 8, 2><<<blocks_hashgrid, N_THREAD>>>(in_grads, in_means, in_offst, in_batch, in_level, in_baser, lg_scale, in_intrp, in_align, gd_embds); 
                    if (in_jacob) kernel_means_bd<3, 8><<<div_round_up(in_batch * in_dimen, N_THREAD), N_THREAD>>>(in_grads, in_jacob, in_batch, in_level, gd_means);
                    break;
                default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
            } break;
        
        default:
            throw std::runtime_error("in_dimen must be 2 or 3.");
    }
}

