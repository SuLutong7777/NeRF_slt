import abc
import torch
import numpy as np

from inspect import getmembers, isfunction

class CUDA_Backend(object):
    def __init__(self):
        # gpu品牌供应商
        self.gpu_vendor = 'nvidia'
        # 链接导入编译生成的库文件（后端文件）
        backend_name = self.get_backend_name()
        self.function_list = self.get_functions(backend_name)
        # 将函数列表变成当前定义类的函数
        # 具体内容：
        # 'gpu_vendor': 'nvidia'
        # 'grad_total_variation'
        # 'grid_encode_backward'
        # 'grid_encode_forward'
        # 'sample_intervals'
        for func in self.function_list:
            self.__dict__[func.__name__] = func   #  i.e. self.func_name = func
        
    # 阻塞当前的代码，直到所有的GPU计算任务都完成 
    def synchronize(self):
        torch.cuda.synchronize()

    # 导入后端生打包生成的库文件，此处是：
    # /home/gaoyu/anaconda3/envs/Pytorch1.12.1/lib/python3.8/site-packages/_cuda_backend.cpython-38-x86_64-linux-gnu.so
    # 是需要自己用cuda代码编译生成的库
    def get_backend_name(self):
        try:
            backend_name = __import__('_cuda_backend')
        except ImportError:
            print("Import Backend Error !")

        return backend_name

    # 根据读取的库，提取出包含的函数
    def get_functions(self, backend_model):
        funcs = []
        for name, func in getmembers(backend_model):
            if not (str(name).startswith('__') and str(name).endswith('__')):
                funcs.append(func)
        return funcs        