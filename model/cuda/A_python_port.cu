#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "src/hash_enc_fd.h"
#include "src/hash_enc_bd.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING32(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be a float32 tensor")

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
    c10::optional<torch::Tensor>& in_jacob)    // 输出哈希特征对位置的梯度（可能没有）
{
    // 检查数据是否在CUDA上
    CHECK_CUDA(in_means);
    CHECK_CUDA(in_embds);
    CHECK_CUDA(in_offst);
    CHECK_CUDA(ot_encod);
    if (in_jacob.has_value()) CHECK_CUDA(in_jacob.value());
    
    // 检查数据是否连续存储
    CHECK_CONTIGUOUS(in_means);
    CHECK_CONTIGUOUS(in_embds);
    CHECK_CONTIGUOUS(in_offst);
    CHECK_CONTIGUOUS(ot_encod);
    if (in_jacob.has_value()) CHECK_CONTIGUOUS(in_jacob.value());

    // 检查数据是否满足指定的类型
    CHECK_IS_FLOATING32(in_means);
    CHECK_IS_FLOATING32(in_embds);
    CHECK_IS_FLOATING32(ot_encod);
    CHECK_IS_INT(in_offst);
    if (in_jacob.has_value()) CHECK_IS_FLOATING32(in_jacob.value());

    // 注意，仅仅支持float32类型的相关计算
    grid_encode_forward_cuda(in_means.data_ptr<float>(), 
                             in_embds.data_ptr<float>(), 
                             in_offst.data_ptr<int>(), 
                             in_batch,
                             in_dimen,
                             ft_dimen,
                             in_level,
                             in_baser,
                             lg_scale,
                             in_intrp,
                             in_align,
                             ot_encod.data_ptr<float>(),
                             in_jacob.has_value() ? in_jacob.value().data_ptr<float>() : nullptr);

}

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
    c10::optional<at::Tensor> gd_means)             // 输出到采样点位置的梯度（可能没有） 
{
    // 检查数据是否在CUDA上
    CHECK_CUDA(in_grads);
    CHECK_CUDA(in_means);
    CHECK_CUDA(in_offst);
    CHECK_CUDA(gd_embds);
    if (in_jacob.has_value()) CHECK_CUDA(in_jacob.value());
    if (gd_means.has_value()) CHECK_CUDA(gd_means.value());
    
    // 检查数据是否连续存储
    CHECK_CONTIGUOUS(in_grads);
    CHECK_CONTIGUOUS(in_means);
    CHECK_CONTIGUOUS(in_offst);
    CHECK_CONTIGUOUS(gd_embds);
    if (in_jacob.has_value()) CHECK_CONTIGUOUS(in_jacob.value());
    if (gd_means.has_value()) CHECK_CONTIGUOUS(gd_means.value());

    // 检查数据是否满足指定的类型
    CHECK_IS_FLOATING32(in_grads);
    CHECK_IS_FLOATING32(in_means);
    CHECK_IS_INT(in_offst);
    CHECK_IS_FLOATING32(gd_embds);
    if (in_jacob.has_value()) CHECK_IS_FLOATING32(in_jacob.value());
    if (gd_means.has_value()) CHECK_IS_FLOATING32(gd_means.value());

    // 反向传播函数
    // 注意，仅仅支持float32类型的相关计算
    grid_encode_backward_cuda(in_grads.data_ptr<float>(), 
                              in_means.data_ptr<float>(), 
                              in_offst.data_ptr<int>(), 
                              in_batch,
                              in_dimen,
                              ft_dimen,
                              in_level,
                              in_baser,
                              lg_scale,
                              in_intrp,
                              in_align,
                              gd_embds.data_ptr<float>(),
                              in_jacob.has_value() ? in_jacob.value().data_ptr<float>() : nullptr,
                              gd_means.has_value() ? gd_means.value().data_ptr<float>() : nullptr);
    
}