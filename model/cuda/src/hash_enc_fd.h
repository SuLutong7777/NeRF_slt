#ifndef _HASH_ENCODE_FD_H
#define _HASH_ENCODE_FD_H

#include <stdint.h>
#include <torch/torch.h>

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
);

#endif