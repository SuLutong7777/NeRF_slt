import os
from torch.utils.cpp_extension import load

# 编译4090和A100平台
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0 8.9"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

# 配置路径为相对路径, 这样方便迁移
# 获取当前.py文件的路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
compile_dir = os.path.join(BASE_DIR, "cuda_compile")
os.makedirs(compile_dir, exist_ok=True)

Grid_cuda_tools = load(
    name="Grid_cuda_tools",
    sources=[os.path.join(BASE_DIR, "C_link_file.cpp"),
             os.path.join(BASE_DIR, "A_python_port.cu"),
             os.path.join(BASE_DIR, "src/hash_enc_fd.cu"),
             os.path.join(BASE_DIR, "src/hash_enc_bd.cu")],  # 替换为你的实际源文件名
    
    extra_cflags=["-std=c++17",
                  "-DCUDA_VERSION=12060"],
    # 可选优化项
    extra_cuda_cflags=["-std=c++17", 
                       "-lineinfo", 
                       "-use_fast_math",
                       "--expt-relaxed-constexpr",
                        # 指定CUDA版本是12.6
                       "-DCUDA_VERSION=12060"],  

    # extra_include_paths=["/home/gaoyu/GaoYu_lib/glm-1.0.2",
    #                      "/home/gaoyu/anaconda3/envs/torch26/lib/python3.12/site-packages/torch/include",
    #                      "/home/gaoyu/anaconda3/envs/torch26/lib/python3.12/site-packages/torch/include/torch/csrc/api/include"],
    
    verbose=True,
    build_directory=compile_dir
)