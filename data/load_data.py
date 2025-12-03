import json
import os
import struct
import cv2
import torch
import math
import re
import time
import collections
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import init_show_figure, show_camera_position

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Images = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}

CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])

class Load_Blender():
    def __init__(self, sys_param):
        start = time.time()
        self.sys_param = sys_param
        self.data_path = self.sys_param['data_path']
        self.data_name = self.sys_param['data_name']
        print("Loading Blender /", self.data_name)
        # 从json文件中获取信息 imgs:[N, 3, H, W] intrs:[N, 3, 3] c2ws:[N, 4, 4] 
        self.imgs_train, self.name_train, self.intrs_train, self.intrs_inv_train, self.c2ws_train, self.train_num = self.load_json('transforms_train.json')
        self.imgs_test, self.name_test, self.intrs_test, self.intrs_inv_test, self.c2ws_test, self.test_num = self.load_json('transforms_test.json')
        self.imgs_val, self.name_val, self.intrs_val, self.intrs_inv_val, self.c2ws_val, self.val_num = self.load_json('transforms_val.json')
        _, _, self.img_h, self.img_w = self.imgs_train.shape
        # 相机外参进行pca操作
        self.c2ws_pca_train, transform_mat, scale_factor = transform_poses_pca(self.c2ws_train)
        self.c2ws_pca_test = apply_pca(self.c2ws_test, transform_mat, scale_factor)
        self.c2ws_pca_val = apply_pca(self.c2ws_val, transform_mat, scale_factor)
        self.idx_train, self.idx_test, self.idx_val = torch.arange(0, self.train_num), torch.arange(0, self.test_num), torch.arange(0, self.val_num)
        end = time.time()
        print(f"运行时间: {end - start:.2f}秒")
        # # 可视化调试
        # show_camera(self.intrs_train, self.intrs_test, self.c2ws_pca_train, self.c2ws_pca_test, self.img_w)

    def load_json(self, json_file, num_threads = 8):
        # 打开并读取 JSON 文件
        json_path = os.path.join(self.data_path, json_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 获取fov并转换为内参矩阵
        camera_angle_x = data['camera_angle_x']
        # 多线程读取相机参数和图像
        frames = data['frames']
        results = []
        # # 单线程
        # for frame in frames:
        #     future = self.load_frame(frame, camera_angle_x)
        #     results.append(future)
        # 多线程
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self.load_frame, frame, camera_angle_x)
                for frame in frames
            ]
            # 先完成的进程先返回值，防止等待阻塞
            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               desc="Loading frame",
                               ncols=100):
                results.append(future.result())

        # 处理获取的图像和相机参数信息(用sorted, .sort()会原地排序)
        results_sorted = sorted(results, key=lambda x: int(re.findall(r'\d+', x['img_name'])[0]))
        c2ws = []
        intrs = []
        intrs_inv = []
        imgs = []
        name_list = []
        for result in results_sorted:
            imgs.append(result['img'])
            intrs.append(result['intrinsic'])
            intrs_inv.append(torch.linalg.inv(result['intrinsic']))
            c2ws.append(result['c2w'])
            name_list.append(result['img_name'])
        cam_num = len(results_sorted)
        imgs = torch.stack(imgs, dim=0)
        intrs = torch.stack(intrs, dim=0)
        intrs_inv = torch.stack(intrs_inv, dim=0)
        c2ws = torch.stack(c2ws, dim=0)

        return imgs, name_list, intrs, intrs_inv, c2ws, cam_num
        
    # 获取相机内外参数和图像信息
    def load_frame(self, frame, camera_angle_x):
        # 获取图像路径和图像tensor
        img_path = os.path.join(self.data_path, frame['file_path'] + '.png')
        img_name = frame['file_path'].split("/")[-1]
        img_tensor, H, W = self.load_img(img_path)
        # 将视场角转换为内参矩阵
        intrinsic = self.fov2intr(camera_angle_x, W, H)             # [3, 3]
        # 读取外参矩阵并转换为OpenCV坐标系
        c2w_gl = torch.tensor(frame['transform_matrix'])
        c2w_cv = self.blender_OpenGL_2_OpenCV(c2w_gl)
        frame_info = {"img_name": img_name, 
                      "img": img_tensor, 
                      "intrinsic": intrinsic,
                      "c2w": c2w_cv}
        return frame_info

    ### opencv读取单张图像(快一些) ###
    def load_img(self, img_path):
        img_bgra_np = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)                                    # [H, W, 4] 取值范围 0~255
        img_rgba_np = cv2.cvtColor(img_bgra_np, cv2.COLOR_BGRA2RGBA)                                # [H, W, 4] 取值范围 0~255
        H, W, _ = img_rgba_np.shape
        img_tensor = torch.as_tensor(img_rgba_np, dtype=torch.float32) / 255.0                      # [H, W, 4] 取值范围 0~1.0
        img_tensor = img_tensor[:, :, :3]*img_tensor[:, :, 3:] + (1 - img_tensor[:, :, 3:])         # [H, W, 3] 取值范围 0~1.0
        img_tensor = img_tensor.permute(2, 0, 1)                                                    # [3, H, W] 取值范围 0~1.0
        return img_tensor, H, W

    ### pil读取单张图像(慢一些) ###
    def load_img_pil(self, img_path):
        img = Image.open(img_path)
        img_tensor = T.ToTensor()(img)                                                              # [4, H, W] 取值范围 0~1.0
        img_tensor = img_tensor[:3, :, :]*img_tensor[3:, :, :] + (1 - img_tensor[3:, :, :])         # [3, H, W] 取值范围 0~1.0
        _, H, W = img_tensor.shape
        return img_tensor, H, W

    ### 将视场角转换为内参矩阵 ###
    def fov2intr(self, camera_angle_x, W, H):
        camera_angle_x = torch.tensor(camera_angle_x, dtype=torch.float32)
        focal = 0.5 * W / math.tan(0.5 * camera_angle_x)
        intrinsic = torch.tensor([[focal, 0.0, W/2.0],
                                  [0.0, focal, H/2.0],
                                  [0.0,   0.0,   1.0]], dtype=torch.float32)
        return intrinsic
    
    ### 将内参矩阵转换为视场角 ###
    def intr2fov(self, intrinsic, W):
        intrinsic = torch.tensor(intrinsic, dtype=torch.float32)
        focal = intrinsic[0, 0]
        camera_angle_x = 2 * math.atan(0.5 * W / focal)
        return camera_angle_x
    
    ### 将外参矩阵从OpenGL坐标系转换到OpenCV坐标系
    def blender_OpenGL_2_OpenCV(self, c2w_gl):
        transform_matrix = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32))
        R_gl = c2w_gl[:3, :3]
        T_gl = c2w_gl[:3, 3:]
        R_cv = R_gl @ transform_matrix
        T_cv = T_gl
        c2w_cv = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32))
        c2w_cv[:3, :3] = R_cv
        c2w_cv[:3, 3:] = T_cv
        return c2w_cv

    ### 将外参矩阵从OpenGL坐标系转换到OpenCV坐标系
    def blender_OpenCV_2_OpenGL(self, c2w_cv):
        transform_matrix = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32))
        R_cv = c2w_cv[:3, :3]
        T_cv = c2w_cv[:3, 3:]
        R_gl = R_cv @ transform_matrix
        T_gl = T_cv
        c2w_gl = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32))
        c2w_gl[:3, :3] = R_gl
        c2w_gl[:3, 3:] = T_gl
        return c2w_gl
    
class Load_LLFF():
    def __init__(self, sys_param):
        start = time.time()
        self.sys_param = sys_param
        self.data_path = self.sys_param['data_path']
        self.data_name = self.sys_param['data_name']
        print("Loading LLFF /", self.data_name)
        self.scale = int(self.sys_param['img_scale'])
        # imgs:[N, 3, H, W] 取值:0~1.0;  intrs:[N, 3, 3];  c2ws:[N, 4, 4];  bounds:[N, 2]
        self.imgs, self.intrs, self.intrs_inv, self.c2ws, self.imgs_num, self.bounds, self.img_h, self.img_w = self.load_llff_data()
        self.c2ws_pca, _, _ = transform_poses_pca(self.c2ws)
        # 选取前百分之八十作为训练集
        self.test_num = int(self.imgs_num * 0.2)
        self.train_num = self.imgs_num - self.test_num
        self.val_num = self.train_num
        self.c2ws_pca_train, self.c2ws_pca_test, self.c2ws_pca_val = self.c2ws_pca[:self.train_num], self.c2ws_pca[self.train_num:], self.c2ws_pca[:self.train_num]
        self.imgs_train, self.imgs_test, self.imgs_val = self.imgs[:self.train_num], self.imgs[self.train_num:], self.imgs[:self.train_num] 
        self.intrs_train, self.intrs_test, self.intrs_val = self.intrs[:self.train_num], self.intrs[self.train_num:], self.intrs[:self.train_num]
        self.intrs_inv_train, self.intrs_inv_test, self.intrs_inv_val = self.intrs_inv[:self.train_num], self.intrs_inv[self.train_num:], self.intrs_inv[:self.train_num]
        self.idx_train, self.idx_test, self.idx_val = torch.arange(0, self.train_num), torch.arange(0, self.test_num), torch.arange(0, self.val_num)
        end = time.time()
        print(f"运行时间:{end-start:.2f}秒")
        # # 可视化相机位姿
        # show_camera(self.intrs_train, self.intrs_test, self.c2ws_pca_train, self.c2ws_pca_test, self.img_w)

    def load_llff_data(self):
        imgs, name_list, imgs_num, img_h, img_w = self.load_llff_imgs()
        c2ws, intrs, intrs_inv, bounds = self.load_llff_npy()
        return imgs, intrs, intrs_inv, c2ws, imgs_num, bounds, img_h, img_w

    # 加载图片
    def load_llff_imgs(self, num_threads=8):
        # 获取图片路径
        if self.scale == 1:
            imgs_root = os.path.join(self.data_path, "images")
        else:
            imgs_root = os.path.join(self.data_path, f"images_{self.scale}")
        imgs_name = [
            f for f in os.listdir(imgs_root)
            if not f.startswith('.')                                            # 跳过隐藏文件
            and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))  # 仅处理图片文件
        ]
        imgs_num = len(imgs_name)
        # 读取图像
        imgs_info = []
        # 多线程
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self.load_img, imgs_root, img_name)
                for img_name in imgs_name
            ]
            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               desc="Loading llff imgs",
                               ncols=100):
                imgs_info.append(future.result())
        # 将读取的图像信息排序并合成tensor矩阵
        imgs_sorted = sorted(imgs_info, key=lambda x:x['img_name'])
        imgs = []
        name_list = []
        for img in imgs_sorted:
            img_tensor = img['img']
            imgs.append(img_tensor)
            name_list.append(img['img_name'])
        imgs = torch.stack(imgs, dim=0)
        _, _, img_h, img_w = imgs.shape
        return imgs, name_list, imgs_num, img_h, img_w

    def load_img(self, img_root, img_name):
        img_path = os.path.join(img_root, img_name)
        img_bgr_np = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)                                     # [H, W, 3] 0~255
        img_rgb_np = cv2.cvtColor(img_bgr_np, cv2.COLOR_BGR2RGB)                                    # [H, W, 3] 0~255
        img_tensor = torch.as_tensor(img_bgr_np, dtype=torch.float32) / 255.0                       # [H, W, 3] 0~1.0
        img_tensor = img_tensor.permute(2, 0, 1)                                                    # [3, H, W] 0~1.0
        img_info = {"img_name": img_name, "img": img_tensor}
        return img_info

    # 加载npy文件
    def load_llff_npy(self):
        self.npy_path = os.path.join(self.data_path, "poses_bounds.npy")
        data_npy = torch.tensor(np.load(self.npy_path), dtype=torch.float32)
        camera_data = data_npy[:, :-2].reshape(-1, 3, 5)        # [N ,3, 5]
        # 获取外参矩阵并转换到OpenCV坐标系
        c2ws_llff = camera_data[:, :3, :4]                      # [N, 3, 4]
        c2ws_cv = self.LLFF_2_OpenCV(c2ws_llff)                 # [N, 4, 4]
        # 获取内参矩阵
        hwfs = camera_data[:, :3, 4].reshape(-1, 3)             # [N, 3]
        intrs = self.hwf_2_intrinsic(hwfs)                      # [N, 3, 3]
        intrs_inv = torch.linalg.inv(intrs)                     # [N, 3, 3]
        bounds = data_npy[:, -2:]                               # [N, 2]
        return c2ws_cv, intrs, intrs_inv, bounds

    # 从llff坐标系转换到OpenCV坐标系
    def LLFF_2_OpenCV(self, c2ws_llff):
        c2ws_cv_tensor = torch.cat([c2ws_llff[:, :, 1:2], c2ws_llff[:, :, 0:1], -c2ws_llff[:, :, 2:3], c2ws_llff[:, :, 3:4]], dim=-1)       # [N, 3, 4]
        norm = torch.tensor([0, 0, 0, 1], dtype=torch.float32).reshape(1, 4)                                                                # [1, 4] 
        norm = norm[None].repeat(c2ws_cv_tensor.shape[0], 1, 1)                                                                                    # [N, 1, 4]
        c2ws_cv = torch.cat([c2ws_cv_tensor, norm], dim=-2)                                                                                 # [N, 4, 4]
        return c2ws_cv
    
    # 将hwf转换为内参矩阵
    def hwf_2_intrinsic(self, hwfs):
        intrs = []
        for hwf in hwfs:
            h, w, focal = hwf
            intr = torch.tensor([
                [focal/self.scale, 0.0, w/self.scale],
                [0.0, focal/self.scale, h/self.scale],
                [0.0,              0.0,          1.0]
            ], dtype=torch.float32)
            intrs.append(intr)
        intrs = torch.stack(intrs, dim=0)
        return intrs

class Load_mip360():
    def __init__(self, sys_param):
        start = time.time()
        self.sys_param = sys_param
        self.data_path = self.sys_param['data_path']
        self.data_name = self.sys_param['data_name']
        print("Loading Mip360 /", self.data_name)
        self.scale = self.sys_param['img_scale']
        # imgs: [N, 3, H, W] 取值0~1.0  intrs: [N ,3, 3]  c2ws: [N, 4, 4]
        self.imgs, self.name_list, self.intrs, self.intrs_inv, self.c2ws, self.imgs_num, self.img_h, self.img_w = self.load_mip360_data()
        self.c2ws_pca, _, _ = transform_poses_pca(self.c2ws)
        # 每隔8张选1张作为测试集
        test_indices = []
        train_indices = []
        for idx in range(self.imgs_num):
            if idx % 8 == 0: 
                test_indices.append(idx)
            else:
                train_indices.append(idx)
        val_indices = train_indices
        # 根据索引划分
        self.imgs_train, self.imgs_test, self.imgs_val = self.imgs[train_indices], self.imgs[test_indices], self.imgs[val_indices] 
        self.c2ws_pca_train, self.c2ws_pca_test, self.c2ws_pca_val = self.c2ws_pca[train_indices], self.c2ws_pca[test_indices], self.c2ws_pca[val_indices]
        self.intrs_train, self.intrs_test, self.intrs_val = self.intrs[train_indices], self.intrs[test_indices], self.intrs[val_indices]
        self.intrs_inv_train, self.intrs_inv_test, self.intrs_inv_val = self.intrs_inv[train_indices], self.intrs_inv[test_indices], self.intrs_inv[val_indices]
        self.name_train = [self.name_list[i] for i in train_indices]
        self.name_test = [self.name_list[i] for i in test_indices]
        self.name_val = [self.name_list[i] for i in val_indices]
        self.idx_train, self.idx_test, self.idx_val = torch.arange(0, self.train_num), torch.arange(0, self.test_num), torch.arange(0, self.val_num)
        end = time.time()
        print(f"运行时间:{end-start:.2f}秒")
        # show_camera(self.intrs_train, self.intrs_test, self.c2ws_pca_train, self.c2ws_pca_test, self.img_w)
        
    def load_mip360_data(self):
        imgs, name_list, imgs_num, img_h, img_w = self.load_mip360_imgs()
        c2ws, intrs, intrs_inv = self.load_mip360_colmap()
        return imgs, name_list, intrs, intrs_inv, c2ws, imgs_num, img_h, img_w
        
    # 加载图片
    def load_mip360_imgs(self, num_threads=8):
        # 获取图片路径
        if self.scale == 1:
            imgs_root = os.path.join(self.data_path, "images")
        else:
            imgs_root = os.path.join(self.data_path, f"images_{self.scale}")
        imgs_name = [
            f for f in os.listdir(imgs_root)
            if not f.startswith('.')                                            # 跳过隐藏文件
            and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))  # 仅处理图片文件
        ]
        imgs_num = len(imgs_name)
        # 读取图像
        imgs_info = []
        # 多线程
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self.load_img, imgs_root, img_name)
                for img_name in imgs_name
            ]
            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               desc="Loading llff imgs",
                               ncols=100):
                imgs_info.append(future.result())
        # 将读取的图像信息排序并合成tensor矩阵
        imgs_sorted = sorted(imgs_info, key=lambda x:x['img_name'])
        imgs = []
        name_list = []
        for img in imgs_sorted:
            img_tensor = img['img']
            imgs.append(img_tensor)
            name_list.append(img['img_name'])
        imgs = torch.stack(imgs, dim=0)
        _, _, img_h, img_w = imgs.shape
        return imgs, name_list, imgs_num, img_h, img_w

    def load_img(self, img_root, img_name):
        img_path = os.path.join(img_root, img_name)
        img_bgr_np = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)                 # [H, W, 3] 0~255
        img_rgb_np = cv2.cvtColor(img_bgr_np, cv2.COLOR_BGR2RGB)                # [H, W, 3] 0~255
        H, W, _ = img_rgb_np.shape
        img_tensor = torch.as_tensor(img_bgr_np, dtype=torch.float32) / 255.0   # [H, W, 3] 0~1.0
        img_tensor = img_tensor.permute(2, 0, 1)                                # [3, H, W] 0~1.0
        img_info = {"img_name": img_name, "img": img_tensor}
        return img_info

    # 加载colmap文件
    def load_mip360_colmap(self):
        colmap_data = Colmap_data(self.data_path)
        extrs_colmap, intrs_colmap = colmap_data.cam_extrs, colmap_data.cam_intrs
        c2ws, intrs, intrs_inv = self.get_sorted_cam_params(extrs_colmap, intrs_colmap)
        return c2ws, intrs, intrs_inv

    def get_sorted_cam_params(self, cam_extrinsics, cam_intrinsics):
        # 提取所有 image 对象
        images_list = list(cam_extrinsics.values())
        images_list.sort(key=lambda x: x.name)
        
        c2ws = []
        intrs = []
        for img in images_list:
            # 外参矩阵
            R_w2c = qvec2rotmat(img.qvec)                                       # 旋转矩阵
            t_w2c = img.tvec.reshape(3, 1)                                      # 平移向量
            c2w = torch.diag(torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32))
            c2w[:3, :3] = R_w2c.T
            c2w[:3, 3] = (-R_w2c.T @ t_w2c).ravel()
            c2ws.append(c2w)
        
            # 内参矩阵
            cam = cam_intrinsics[img.camera_id]
            if cam.model == "PINHOLE":
                fx, fy, cx, cy = cam.params
                intr = torch.tensor([
                    [fx/self.scale, 0, cx/self.scale],
                    [0, fy/self.scale, cy/self.scale],
                    [0,             0,             1]
                ], dtype=torch.float32)
            elif cam.model == "SIMPLE_PINHOLE":
                f, cx, cy = cam.params
                intr = torch.tensor([
                    [f/self.scale, 0, cx/self.scale],
                    [0, f/self.scale, cy/self.scale],
                    [0,            0,             1]
                ], dtype=torch.float32)
            elif cam.model == "SIMPLE_RADIAL":
                f, cx, cy, k1 = cam.params  # 忽略畸变
                intr = torch.tensor([
                    [f/self.scale, 0, cx/self.scale],
                    [0, f/self.scale, cy/self.scale],
                    [0,            0,             1]
                ], dtype=torch.float32)
            else:
                raise NotImplementedError(f"Camera model {cam.model} not supported yet.")
            intrs.append(intr)
        
        c2ws_tensor = torch.stack(c2ws, dim=0)              # [N, 4, 4]
        intrs_tensor = torch.stack(intrs, dim=0)            # [N, 3, 3]
        intrs_tensor_inv = torch.linalg.inv(intrs_tensor)   # [N, 3, 3]
        return c2ws_tensor, intrs_tensor, intrs_tensor_inv           

class Colmap_data():
    def __init__(self, colmap_path):
        cameras_extr_file = os.path.join(colmap_path, 'sparse/0', 'images.bin')
        cameras_intr_file = os.path.join(colmap_path, 'sparse/0', 'cameras.bin')
        self.cam_extrs = self.read_extrinsics_binary(cameras_extr_file)
        self.cam_intrs = self.read_intrinsics_binary(cameras_intr_file)
    
        # 读取colmap生成的外参二进制文件
    def read_extrinsics_binary(self, cam_extrinsics_file):
        print("-----读取colmap数据集二进制外参文件")
        # 初始化一个图片数据字典
        images = {}
        with open(cam_extrinsics_file, 'rb') as fid:
            # 读取文件中图像的数量, Q代表8字节的无符号长整型
            num_images = self.read_next_bytes(fid, 8, 'Q')[0]
            # 遍历每个图像
            for _ in range(num_images):
                image_params = self.read_next_bytes(fid, num_bytes=64, format_char_sequence='idddddddi')
                # 图像唯一标识符
                image_id = image_params[0]
                # 图像的旋转矩阵四元数
                qvec = torch.tensor(image_params[1:5], dtype=torch.float32)
                # 图像的平移三维向量
                tvec = torch.tensor(image_params[5:8], dtype=torch.float32)
                # 对应相机id
                camera_id = image_params[8]
                # 读取图像名称
                image_name = ""
                current_char = self.read_next_bytes(fid, 1, 'c')[0]
                while current_char != b"\x00":
                    image_name += current_char.decode("utf-8")
                    current_char = self.read_next_bytes(fid, 1, 'c')[0]
                # 读取图像中2D点数量
                num_points2D = self.read_next_bytes(fid, 8, 'Q')[0]
                # 读取2D点坐标和对应3D点id
                x_y_id_s = self.read_next_bytes(fid, 24*num_points2D, 'ddq'*num_points2D)
                # 每隔3个数取一次，取第一个数和第二个数 使用 map(float, ...) 将这些坐标值转换为浮点数。
                # 最后，tuple(...) 将转换后的浮点数值组合成一个元组。
                x_y_id_s = torch.tensor(list(map(float, x_y_id_s)), dtype=torch.float32)
                xys = torch.stack([x_y_id_s[0::3], x_y_id_s[1::3]], dim=1)
                point3D_ids = x_y_id_s[2::3].long()
                images[image_id] = Images(
                    id = image_id, qvec = qvec, tvec = tvec,
                    camera_id = camera_id, name = image_name,
                    xys = xys, point3D_ids = point3D_ids 
                )
        return images
    
    # 读取colmap生成的内参二进制文件
    def read_intrinsics_binary(self, cam_intrinsics_file):
        print("-----读取colmap数据集二进制内参文件")
        # 初始化一个相机参数文件
        cameras = {}
        with open(cam_intrinsics_file, 'rb') as fid:
            # 读取相机数量
            num_cameras = self.read_next_bytes(fid, 8, 'Q')[0]
            # 遍历每个相机并读取属性
            for _ in range(num_cameras):
                camera_params = self.read_next_bytes(fid, 24, 'iiQQ')
                # 相机id
                camera_id = camera_params[0]
                # 相机模型id
                model_id = camera_params[1]
                # 利用model_id查找相机模型名称
                model_name = CAMERA_MODEL_IDS[model_id].model_name
                # 图像宽度
                width = camera_params[2]
                # 图像高度
                height = camera_params[3]
                # 查询该相机参数数量并获取内参值
                num_params = CAMERA_MODEL_IDS[model_id].num_params
                intr_params = self.read_next_bytes(fid, 8*num_params, "d"*num_params)
                # 存储相机类
                cameras[camera_id] = Camera(
                    id = camera_id,
                    model = model_name,
                    width = width,
                    height = height,
                    params = np.array(intr_params)
                )
        return cameras

    def read_next_bytes(self, fid, num_bytes, format_char_sequence, endian_character = "<"):
        data = fid.read(num_bytes)
        # 返回解包数据，一个元组
        return struct.unpack(endian_character + format_char_sequence, data)

# COLMAP产生的相机位姿的世界坐标系不一定是啥样
# 这个操作将COLMAP生成的坐标系进行转换，变成以环绕中心为世界坐标系原点的全新分布坐标
# pca是指主成分分析，Principal Component Analysis，一种数据降维方法
# 主成分分析可以看这个：https://zhuanlan.zhihu.com/p/37777074
# 这段代码实现PAC用的是上面这个链接中3.5的(1)方法
# 输入poses为[N, 4, 4], 必须为c2w矩阵，不能为w2c
def transform_poses_pca(poses_c2w):
    poses_c2w = poses_c2w.detach()
    # 获取所有相机的中心点
    trans = poses_c2w[:, :3, 3]
    # 取平均值
    trans_mean = torch.mean(trans, dim=0)
    # 中心化，相当于取所有点的平均中心为新坐标原点
    # 生成新的相机中心位置 [194, 3]
    trans = trans - trans_mean
    # 计算特征值eigval，和特征向量eigvec
    # 注意，这两个算出来是复数格式，有实部和虚部，即使虚部为0，也会保留
    # 所以这里要除去虚部(虚部全部算出来都是0)
    # trans.T @ trans: [3,3], 注意，这个过程在计算平移向量集合的协方差（正常有个除以n的系数，但是不影响特征向量）
    # eigval:[3], eigvec:[3,3]
    # eigval, eigvec = torch.linalg.eig(trans.T @ trans)
    # 转成Numpy做，pytorch版本的特征向量符号与Numpy不一致
    eigval, eigvec = np.linalg.eig(trans.cpu().numpy().T @ trans.cpu().numpy())
    eigval = torch.from_numpy(eigval)
    eigvec = torch.from_numpy(eigvec)
    # print(eigval, eigvec)
    # exit()
    # 对所有特征值进行从大到小的排序，获取排序的索引
    inds = torch.argsort(eigval.real, descending=True)
    # 同时排序特征向量
    # eigvec = eigvec[:, inds].real
    eigvec = eigvec[:, inds]
    # print(eigvec, "2222")
    # 将特征向量转置，构造投影矩阵，将所有坐标点投影到新的坐标系下
    # 这个新的坐标系的轴就是数据的主成分轴。
    # 这里eigvec为[3,3]，因为数据一共有三个主成分，分别为x,y,z，都需要保留，所以上面链接中的k值取3，就等同于不用筛选
    # eigvec中，每一列是特征向量，转置之后变成行，在进行投影的时候就是rot@trans，x,y,z维度能对应
    rot = eigvec.T
    # 保持坐标系变换后与原来规则相同
    # 在三维空间中，一个合法的旋转矩阵应该是正交的且行列式为1，这保证了坐标系变换保持了空间的右手规则。
    # 如果行列式小于0，表明旋转矩阵将导致坐标系翻转，违反了右手规则。
    # 一个矩阵的行列式（np.linalg.det(rot)）告诉我们这个矩阵是保持空间的定向（右手或左手）不变还是改变了空间的定向。具体来说：
    # 如果行列式大于0，说明变换后的坐标系保持原有的定向（即如果原坐标系是右手的，变换后仍然是右手的）。
    # 如果行列式小于0，说明变换后的坐标系改变了原有的定向（即从右手变为了左手，或从左手变为了右手）。
    if torch.linalg.det(rot) < 0:
        rot = torch.diag(torch.tensor([1.0, 1.0, -1.0])) @ rot

    # 构建完整的[R|T]变换矩阵，直接针对原始的pose信息，不再单纯考虑trans
    # 尺寸是[3, 4]
    transform_mat = torch.cat([rot, rot @ -trans_mean[:, None]], dim=-1)
    # 转为[4, 4]
    transform_mat = torch.cat([transform_mat, torch.tensor([[0, 0, 0, 1.]])], dim=0)
    # 整体RT矩阵转换[N, 4, 4]
    # poses_recentered = transform_mat @ poses_c2w
    poses_recentered = torch.matmul(transform_mat.unsqueeze(0), poses_c2w)
    # 检查坐标轴方向
    # 检查在新坐标系中，相机指向的平均方向的y分量是否向下。如果是的话，这意味着变换后的位姿与常规的几何或物理约定（例如，通常期望的y轴向上）不符。
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0])) @ poses_recentered
        transform_mat = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0])) @ transform_mat
        
    # 原始相机方向向量的平均（这里假设第三列是前向向量）
    orig_forward = poses_c2w[:, :3, 2].mean(0)
    new_forward = poses_recentered[:, :3, 2].mean(0)
    # 如果方向相反（dot product < 0），就翻转 Z 和 X（保持右手系）
    if (orig_forward @ new_forward) < 0:
        poses_recentered = torch.diag(torch.tensor([-1.0, 1.0, -1.0, 1.0])) @ poses_recentered
        transform_mat = torch.diag(torch.tensor([-1.0, 1.0, -1.0, 1.0])) @ transform_mat
    # 对数据进行归一化，收敛到[-1, 1]之间
    scale_factor = 1. / torch.max(torch.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    poses_recentered[:, 3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(poses_recentered.shape[0], 1)
    # transform_mat = torch.diag(torch.tensor([scale_factor] * 3 + [1])) @ transform_mat
    
    return poses_recentered, transform_mat, scale_factor

def apply_pca(poses_c2ws, trans, scale):
    c2ws_pca = trans @ poses_c2ws
    c2ws_pca[:, :3, 3] *= scale
    return c2ws_pca

def inverse_transform_pca(poses_recentered, transform_mat, scale_factor):
    poses_recentered[:, :3, 3] /= scale_factor
    transform_inv = torch.linalg.inv(transform_mat)
    poses_original = torch.matmul(transform_inv.unsqueeze(0), poses_recentered.clone())
    poses_original[:, 3, :] = torch.tensor([0, 0, 0, 1.0])
    return poses_original

def qvec2rotmat(qvec):
    return torch.tensor([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]], dtype=torch.float32)
