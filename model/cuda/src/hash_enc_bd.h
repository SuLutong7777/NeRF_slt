#ifndef _HASH_ENCODE_BD_H
#define _HASH_ENCODE_BD_H

#include <stdint.h>
#include <torch/torch.h>

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
);

#endif