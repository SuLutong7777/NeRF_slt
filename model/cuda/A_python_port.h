#pragma once
#include <stdio.h>
#include <iostream>

// 与Python接口对接的前向函数
void grid_encode_forward(
    const torch::Tensor& in_means,             // 输入采样点位置
    const torch::Tensor& in_embds,             // 输入哈希表可学习参数
    const torch::Tensor& in_offst,             // 输入每一层在哈希表里的偏移
    const uint32_t in_batch,                   // 输入采样点数量B
    const uint32_t in_dimen,                   // 输入采样点维度D
    const uint32_t ft_dimen,                   // 输入哈希特征维度C
    const uint32_t in_level,                   // 输入哈希表格层数L
    const uint32_t in_baser,                   // 输入哈希表最低层分辨率H
    const float    lg_scale,                   // 输入哈希表层级缩放率S 
    const uint32_t in_intrp,                   // 输入插值方式，0是线性，1是平滑
    const bool     in_align,                   // 输入是否对齐角点 
    torch::Tensor& ot_encod,                   // 输出编码后的新特征
    c10::optional<torch::Tensor>& in_jacob     // 输出哈希特征对位置的梯度（可能没有） 
);

// 与Python接口对接的反向函数
void grid_encode_backward(
    const torch::Tensor& in_grads,                  // 从反向端传过来的梯度
    const torch::Tensor& in_means,                  // 输入采样点位置
    const torch::Tensor& in_offst,                  // 输入每一层在哈希表里的偏移
    const uint32_t in_batch,                        // 输入采样点数量B
    const uint32_t in_dimen,                        // 输入采样点维度D
    const uint32_t ft_dimen,                        // 输入哈希特征维度C
    const uint32_t in_level,                        // 输入哈希表格层数L
    const uint32_t in_baser,                        // 输入哈希表最低层分辨率H
    const float    lg_scale,                        // 输入哈希表层级缩放率S 
    const uint32_t in_intrp,                        // 输入插值方式，0是线性，1是平滑
    const bool     in_align,                        // 输入是否对齐角点
    torch::Tensor& gd_embds,                        // 输出到哈希表的梯度
    const c10::optional<torch::Tensor>& in_jacob,   // 哈希特征对位置的梯度（可能没有）
    c10::optional<at::Tensor> gd_means             // 输出到采样点位置的梯度（可能没有） 
);