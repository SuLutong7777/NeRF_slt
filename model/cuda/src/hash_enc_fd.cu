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
#include "hash_enc_fd.h"

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

// 平滑插值函数的导数，根据上面的函数定义，求导是 6t - 6t^2
template <typename T>
__device__ inline T smoothstep_derivative(T val)
{
    return 6*val*(1.0f - val);
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
        index = fast_hash<in_dimen>(pos_grid);
    }
    // 因为哈希表格在存储的时候是一块连续的一维内存
    // (index % hashmap_size) 表示在哈希表的第几行
    // 乘以 ft_dimen 表示每一行有 ft_dimen 个特征值，转换为每个特征起始位置的数组的索引
    // ft_ele_idx 表示拿到第几个特征
    return (index % hashmap_size) * ft_dimen + ft_ele_idx;
}

template <uint32_t in_dimen, uint32_t ft_dimen>
__global__ void kernel_grid_fd(
    const float* __restrict__ in_means, 
    const float* __restrict__ in_embds, 
    const int*   __restrict__ in_offst,
    const uint32_t in_batch, 
    const uint32_t in_level,
    const uint32_t in_baser, 
    const float    lg_scale,
    const uint32_t in_intrp,
    const bool     in_align,
    float* __restrict__ ot_encod,
    float* __restrict__ in_jacob
) 
{
    // 该函数使用二维的 CUDA 网格阵列，X方向上有 ceil(in_batch / N_THREAD) 个 block，Y 方向上有 in_level 个 block
    // 获取当前的线程索引id，如果超过总的输入坐标数量，就直接返回
    // idx 是X方向上 in_batch 维度的索引，每个X方向都有 in_batch 个线程
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= in_batch) return;
    // 当前线程负责处理具体哪一层哈希表的坐标点
    const uint32_t cur_level = blockIdx.y;
    
    // 将当前线程的指针，指向该线程负责层 cur_level 的数据起始位置（移动指针地址）
    in_embds += (uint32_t)in_offst[cur_level] * ft_dimen;
    // 将当前线程的指针，指向输入坐标的第 idx 个坐标起始位置，这个坐标就是当前线程负责处理的坐标
    in_means += idx * in_dimen;
    // 将当前线程的指针，指向输出编码后特征的第 idx 个坐标在 cur_level 层的起始位置
    // ot_encod 初始化的形状是 [in_level, in_batch, ft_dimen]，所以要保证存储数据的位置正确
    ot_encod += cur_level * in_batch * ft_dimen + idx * ft_dimen;

    // 输入坐标数据的越界检查，必须要在 0-1 之间
    #pragma unroll
    for (uint32_t d = 0; d < in_dimen; d ++) 
    {
        // 因为雅可比和输出编码的容器初始化全是数值0，所以这里可以直接返回
        if (in_means[d] < 0. || in_means[d] > 1.) 
            return;
    }

    // 如果输入数据没有越界，进行正常计算
    // 当前层level哈希网格的尺寸
    const uint32_t hashmap_size = in_offst[cur_level + 1] - in_offst[cur_level];
    // exp2f(cur_level * lg_scale) 计算当前层相比较于基础层的缩放比例
    // scale 近似等于这一层网格最大的格点尺寸，最后的 - 1.0f 是因为索引从0开始
    // 例如，0层，scale就是15, 2层scale就是31...
    const float scale = exp2f(cur_level * lg_scale) * in_baser - 1.0f;
    // 拿到当前层的分辨率
    const uint32_t resolution = (uint32_t)ceil(scale) + 1;
    
    // 初始化插值权重相关的容器数组
    float pos[in_dimen];
    float pos_deriv[in_dimen]; 
    uint32_t pos_grid[in_dimen];

    // 计算查询哈希表及插值计算用的一些数据
    // pos_grid[d] 后面会用来索引查询哈希表
    // pos[d] 后面会用来计算插值，存的是插值权重
    // pos_deriv[d] 后面会用来计算输出特征对输入坐标的导数
    #pragma unroll
    for (uint32_t d = 0; d < in_dimen; d++)
    {
        // 将归一化的坐标恢复到当前网格下的具体位置
        pos[d] = in_means[d] * scale + (in_align ? 0.0f : 0.5f);
        // floorf 表示向下取整, pos_grid[d] 拿到的是当前坐标所在网格的最左下角坐标索引（三维空间里就是所有索引最小的角点）
        pos_grid[d] = floorf(pos[d]);
        // 计算相对于格点的偏移
        pos[d] -= (float)pos_grid[d];
        // 根据插值的要求，计算插值
        // in_intrp == 1 表示使用平滑插值
        if (in_intrp == 1) 
        {
            // 平滑插值的导数因子
            pos_deriv[d] = smoothstep_derivative(pos[d]);
            // 注意，pos在这里存的插值的左侧权重，右侧权重是 1 - pos
            pos[d] = smoothstep(pos[d]);
        }
        // 其他情况使用线性插值
        else
        {
            // 线性插值的导数因子是1
            // 线性插值左侧权重直接就是本身，所以pos数值不变，右侧权重直接就是 1 - pos
            // 线性插值的的左侧权重定义就是 t, 右侧权重是 1-t，所以左侧权重的导数就是 1
            pos_deriv[d] = 1.0f;
        }

    }

    //printf("[b=%d, l=%d] pos=(%f, %f)+(%d, %d)\n", b, level, pos[0], pos[1], pos_grid[0], pos_grid[1]);

    // 初始化插值计算结果的缓存数组
    float results[ft_dimen] = {0};

    // 计算插值数值
    #pragma unroll
    // 1 << in_dimen 表示 2 的 in_dimen 次方，C++ 中的常用写法，把 1 左移 in_dimen 位, 0001 -> 1000 二进制就等于 8
    // 如果 in_dimen = 3, 那么就是 8 个角点, 如果 in_dimen = 2, 那么就是 4 个角点
    for (uint32_t ii = 0; ii < (1 << in_dimen); ii++)
    {
        float w = 1;
        // 坐标容器，缓存包围当前坐标点的box框 8 个角点中的某一个
        // 以 3 维为例进行该函数的说明
        uint32_t pos_grid_local[in_dimen];
        
        #pragma unroll
        // 遍历每一个维度的坐标, x,y,z, d=0代表x维度，d=1代表y维度，d=2代表z维度
        for (uint32_t d = 0; d < in_dimen; d++)
        {
            // 这部分有点难理解，是在用二进制位的方式来计算角点数据
            // ii 是从 0 到 7 的数字，二进制就是 000 到 111
            // 1 << d 表示把 1 左移 d 位，一共有 3 个数值 001, 010, 100
            // & 符号表示按位与计算，(ii & (1 << d)) 表示查看 ii 的第 d 位是不是 1
            // 如果 ii & (1 << d) 是 0, 表示第 d 位是 0, 如果不等于 0, 表示第 d 位是 1
            // 总之，外层的 for 会循环 8 次，表示 8 个角点，内层的 for 会循环 3 次，表示 x,y,z 三个维度
            // 在前面我们拿到了 pos_grid, 是当前网格的左下角坐标索引，已经拿到1个角点了
            // 这里的循环是在拿另外的 7 个角点（ii = 0 数值会自动等于左下角的角点，原本数值）的坐标索引，并且存到 pos_grid_local 里面
            // 另外插值法的权重w是各个维度的权重累乘，可以自己推导一下
            if ((ii & (1 << d)) == 0)
            {
                // 如果当前维度 d 的那一位是0，表示使用左侧的格点
                // 左侧格点的插值权重是 1 - pos[d]，数值是pos[d]，数值和权重成反比（离边界点越近，数值越小，但是影响应该越大，所以反比）
                w *= 1 - pos[d];
                // 左侧格点直接取左下角的数值就行
                pos_grid_local[d] = pos_grid[d];
            }
            else
            {
                // 如果当前维度 d 的那一位是1，表示使用右侧的格点
                // 右侧格点的插值权重是 pos[d]，数值是 pos[d] + 1
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        // 此时 pos_grid_local 拿到的是包围当前坐标点的box框 8 个角点中的某一个
        // 计算得到这个角点在哈希表中的索引位置，这个索引位置指向对应特征的首地址，可以拿到相应的特征参数
        uint32_t index = get_grid_index<in_dimen, ft_dimen>(in_align, 0, hashmap_size, resolution, pos_grid_local);
        
        // 根据ft_dimen维度，对每一位的特征值循环计算
        #pragma unroll
        for (uint32_t ch = 0; ch < ft_dimen; ch++)
        {
            // 坐标每一位的特征值，乘以插值权重，累加到结果中
            // 注意这是循环内，所以每个角点都会根据权重，累加进来一部分内容，结果就是8个角点对应特征值的加权和
            results[ch] += w * in_embds[index + ch];
        }

        //printf("[b=%d, l=%d] int %d, idx %d, w %f, val %f\n", b, level, idx, index, w, grid[index]);
    }    

    // 将计算的最终特征结果赋值给最终要输出的特征容器，当前坐标点的特征结果计算完毕
    #pragma unroll
    for (uint32_t ch = 0; ch < ft_dimen; ch++)
    {
        ot_encod[ch] = results[ch]; 
    }

    // differentiable (soft) indexing: https://discuss.pytorch.org/t/differentiable-indexing/17647/9
    if (in_jacob)
    {
        // 调整雅可比梯度的起始指针位置
        in_jacob += idx * in_dimen * in_level * ft_dimen + cur_level * in_dimen * ft_dimen; // B L D C
        
        #pragma unroll
        // 对每个输入坐标xyz分别求导
        for (uint32_t gd = 0; gd < in_dimen; gd++)
        {
            // 输出数据的缓存容器
            float results_grad[ft_dimen] = {0};
            
            #pragma unroll
            // (1 << (in_dimen - 1)) 这个部分难理解，举个例子说明：
            // 一维线性插值公式: f(t) = (1-t)*(插值左端点数值v0) + t*(插值右端点数值v1)
            // 求导：df/dt = v1-v0，这里只有1个差分项
            // 推广到二维插值公式: f(tx, ty) = (1-tx)*(1-ty)*v00 + tx*(1-ty)*v10 + (1-tx)*ty*v01 + tx*ty*v11
            // 求导：df/dtx = (1-ty)*(v10-v00) + ty*(v11-v01)，注意这里出现了两个差分，(v10-v00)和(v11-v01)
            // 当插值推广到三维的时候，上面的差分项就变成了4个
            // 所以这部分循环的次数就是 2^(in_dimen-1) 次，2D 情况就是循环上面推导的两次，3D 情况就是上面的4次
            for (uint32_t ii = 0; ii < (1 << (in_dimen - 1)); ii++)
            {
                float w = scale;
                // 容器，
                uint32_t pos_grid_local[in_dimen];

                #pragma unroll
                // 这个部分和上面的计算插值相似
                // nd 表示除了当前 gd 维度之外的维度
                // 比如 gd=0 表示 x 维度，nd就表示y维度和z维度，nd=0,1遍历yz
                // 所以结论就是：这段代码最后会拿到 gd 维度数据没有填，而额外两个维度填好的坐标点
                // 例如: 假设当前输入的xyz点被[0,0,0]和[1,1,1]这个网格包围，
                // 则gd=0表示x维度, 这段代码拿到的就是 [?, 0, 0] [?, 0, 1], [?, 1, 0], [?, 1, 1] 这四个顶点坐标
                for (uint32_t nd = 0; nd < in_dimen - 1; nd++)
                {
                    const uint32_t d = (nd >= gd) ? (nd + 1) : nd;

                    if ((ii & (1 << nd)) == 0)
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
                // 前面的 ? 维度会在这里填上，表示需要用来作差计算梯度的两个数据，如 v1-v0
                pos_grid_local[gd] = pos_grid[gd];
                uint32_t index_left = get_grid_index<in_dimen, ft_dimen>(in_align, 0, hashmap_size, resolution, pos_grid_local);
                pos_grid_local[gd] = pos_grid[gd] + 1;
                uint32_t index_right = get_grid_index<in_dimen, ft_dimen>(in_align, 0, hashmap_size, resolution, pos_grid_local);

                #pragma unroll
                // 计算得到最终的导数梯度
                for (uint32_t ch = 0; ch < ft_dimen; ch++)
                {   
                    results_grad[ch] += w * (in_embds[index_right + ch] - in_embds[index_left + ch]) * pos_deriv[gd];
                }
            }

            #pragma unroll
            // 把计算出来的梯度存回全局梯度的大表
            for (uint32_t ch = 0; ch < ft_dimen; ch++)
            {
                in_jacob[gd * ft_dimen + ch] = results_grad[ch];
            }
        }
    }
}


// 前向传播函数
void grid_encode_forward_cuda(
    const float*   in_means,    // 输入的坐标，0-1范围
    const float*   in_embds,    // 哈希表数据
    const int*     in_offst,    // 每一层哈希表的偏移量
    const uint32_t in_batch,    // 输入坐标的数量
    const uint32_t in_dimen,    // 输入坐标的维度，2 或 3
    const uint32_t ft_dimen,    // 输出特征的维度，1,2,4 或 8
    const uint32_t in_level,    // 哈希网格的层数
    const uint32_t in_baser,    // 基础层的分辨率
    const float    lg_scale,    // 每一层相对于基础层的缩放比例的对数值
    const uint32_t in_intrp,    // 插值方式，0 线性插值，1 平滑插值
    const bool     in_align,    // 是否对齐角点
    float*         ot_encod,    // 输出编码后的特征
    float*         in_jacob     // 输出雅可比矩阵
)
{
    // 每个 block 512 个线程，处理 512 个输入坐标
    static constexpr uint32_t N_THREAD = 512;
    // 总共有向上取整 ceil(in_batch / 512) * in_level 个 block
    // 让每个线程负责处理一个坐标在某一层的计算
    const dim3 blocks_hashgrid = {div_round_up(in_batch, N_THREAD), in_level, 1};
    // 仅支持 in_dimen = 2, 3
    // 仅支持 ft_dimen = 1, 2, 4, 8
    // 注意，用实例 switch 的原因是 template <uint32_t in_dimen, uint32_t ft_dimen> 模版调用的时候必须是编译期常量
    switch (in_dimen)
    {
        // 2D 实验情况，输入坐标是2维
        case 2:
            switch (ft_dimen)
            {
                case 1: kernel_grid_fd<2, 1><<<blocks_hashgrid, N_THREAD>>>(in_means, in_embds, in_offst, in_batch, in_level, in_baser, lg_scale, in_intrp, in_align, ot_encod, in_jacob); break;
                case 2: kernel_grid_fd<2, 2><<<blocks_hashgrid, N_THREAD>>>(in_means, in_embds, in_offst, in_batch, in_level, in_baser, lg_scale, in_intrp, in_align, ot_encod, in_jacob); break;
                case 4: kernel_grid_fd<2, 4><<<blocks_hashgrid, N_THREAD>>>(in_means, in_embds, in_offst, in_batch, in_level, in_baser, lg_scale, in_intrp, in_align, ot_encod, in_jacob); break;
                case 8: kernel_grid_fd<2, 8><<<blocks_hashgrid, N_THREAD>>>(in_means, in_embds, in_offst, in_batch, in_level, in_baser, lg_scale, in_intrp, in_align, ot_encod, in_jacob); break;
                default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
            } break;
        // 3D 实验情况，输入直接是三维空间点
        case 3:
            switch (ft_dimen)
            {
                case 1: kernel_grid_fd<3, 1><<<blocks_hashgrid, N_THREAD>>>(in_means, in_embds, in_offst, in_batch, in_level, in_baser, lg_scale, in_intrp, in_align, ot_encod, in_jacob); break;
                case 2: kernel_grid_fd<3, 2><<<blocks_hashgrid, N_THREAD>>>(in_means, in_embds, in_offst, in_batch, in_level, in_baser, lg_scale, in_intrp, in_align, ot_encod, in_jacob); break;
                case 4: kernel_grid_fd<3, 4><<<blocks_hashgrid, N_THREAD>>>(in_means, in_embds, in_offst, in_batch, in_level, in_baser, lg_scale, in_intrp, in_align, ot_encod, in_jacob); break;
                case 8: kernel_grid_fd<3, 8><<<blocks_hashgrid, N_THREAD>>>(in_means, in_embds, in_offst, in_batch, in_level, in_baser, lg_scale, in_intrp, in_align, ot_encod, in_jacob); break;
                default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
            } break;
        
        default:
            throw std::runtime_error("in_dimen must be 2 or 3.");
    }

}


