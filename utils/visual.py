import torch
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 初始化画布
def init_show_figure(show_info=True):
    # 创建一个新的matplotlib图像
    all_fig = plt.figure(figsize=(16, 9))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.default'] = 'regular'
    # 创建一个3D坐标轴对象
    ax = Axes3D(all_fig, auto_add_to_figure=False)
    all_fig.add_axes(ax)
    # 设置坐标平面背景颜色为纯白(默认灰色)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    
    if show_info:
        # 显示坐标轴标签
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        # 设置坐标轴范围
        ax.set_xlim(-1., 1.)
        ax.set_ylim(-1., 1.)
        ax.set_zlim(-1., 1.)
    else:
        # 隐藏网格和坐标轴
        ax.grid(False)
        ax.axis(False)

    # 开启交互式绘图模式
    plt.ion()
    # 确保3F坐标比例一致
    plt.gca().set_box_aspect((1, 1, 1))
    return ax

# 绘制相机轨迹
def draw_space_lines(c2w_mat, ax, line_w=0.5):
    pts_3d = np.array(c2w_mat[:, :3, 3].detach().cpu()) # [N, 3]
    ax.plot(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2], linestyle='--', color='red', marker='o', label='Dashed Line', linewidth=line_w)

# RT_mat:[N, 4, 4]
def show_camera_position(intr_mat, RT_mat, img_w, fig_ax, color = (0.7,0.2,0.7), cam_size=0.025, line_w=0.5):
    clip_pose = RT_mat[:, :3, :]
    draw_camera_shape(clip_pose, intr_mat, color, img_w, fig_ax, cam_size, line_w)

# extr_mat：[C2W]
def draw_camera_shape(extr_mat, intr_mat, color, img_w, ax, cam_size=0.25, line_w=0.5):
    # extr_mat: [84, 3, 4]
    # intr_mat: [84, 3, 3]
    cam_line = cam_size
    focal = intr_mat[:,0,0]*cam_line/img_w
    
    cam_pts_1 = torch.stack([-torch.ones_like(focal)*cam_line/2,
                             -torch.ones_like(focal)*cam_line/2,
                             focal], -1)[:,None,:].to(extr_mat.device)
    cam_pts_2 = torch.stack([-torch.ones_like(focal)*cam_line/2,
                             torch.ones_like(focal)*cam_line/2,
                             focal], -1)[:,None,:].to(extr_mat.device)
    cam_pts_3 = torch.stack([torch.ones_like(focal)*cam_line/2,
                             torch.ones_like(focal)*cam_line/2,
                             focal], -1)[:,None,:].to(extr_mat.device)
    cam_pts_4 = torch.stack([torch.ones_like(focal)*cam_line/2,
                             -torch.ones_like(focal)*cam_line/2,
                             focal], -1)[:,None,:].to(extr_mat.device)
    
    cam_pts_1 = cam_pts_1 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
    cam_pts_2 = cam_pts_2 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
    cam_pts_3 = cam_pts_3 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
    cam_pts_4 = cam_pts_4 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
    
    # [N, 5, 3]
    cam_pts = torch.cat([cam_pts_1, cam_pts_2, cam_pts_3, cam_pts_4, cam_pts_1], dim=-2)

    for i in range(4):
        # [84, 2, 3]
        cur_line_pts = torch.stack([cam_pts[:,i,:], cam_pts[:,i+1,:]], dim=-2).to('cpu')
        for each_cam in cur_line_pts:
            ax.plot(each_cam[:,0],each_cam[:,1],each_cam[:,2],color=color,linewidth=line_w)
    extr_T = extr_mat[:, :3, 3]
    
    for i in range(4):
        # [84, 2, 3]
        cur_line_pts = torch.stack([extr_T, cam_pts[:,i,:]], dim=-2).to('cpu')
        for each_cam in cur_line_pts:
            ax.plot(each_cam[:,0],each_cam[:,1],each_cam[:,2],color=color,linewidth=line_w)
    extr_T = extr_T.to('cpu')

    ax.scatter(extr_T[:,0],extr_T[:,1],extr_T[:,2],color=color,s=5) 
    ax.set_aspect('equal') 

# 可视化光线
def show_rays(rays_o, rays_d, ax, line_w=0.5):
    rays_end = rays_o + rays_d
    num_rays = len(rays_o)
    # 绘制每条光线
    for i in range(num_rays):
        ax.plot([rays_o[i, 0], rays_end[i, 0]],
                [rays_o[i, 1], rays_end[i, 1]],
                [rays_o[i, 2], rays_end[i, 2]], 'r-', lw=line_w)
        
# OpenCV存rgb图像   rgb_tensor: [3, h, w]
def save_img_cv2(rgb_tensor, save_path):
    rgb_tensor = rgb_tensor.clamp(0.0, 1.0)
    img_np = (rgb_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

# PIL存rgb图像      rgb_tensor: [3, h, w]
def save_img_pil(rgb_tensor, save_path):
    rgb_tensor = rgb_tensor.clamp(0.0, 1.0)
    img_np = (rgb_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(img_np).save(save_path)

# OpenCV存depth图像   depth_tensor: [h, w]
def save_depth_cv2(depth_tensor, save_path):
    # depth_tensor = depth_tensor.clamp(0.0, 1.0)
    depth_np = (depth_tensor.cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(save_path, depth_np)

# PIL存depth图像      depth_tensor: [h, w]
def save_depth_pil(depth_tensor, save_path):
    # depth_tensor = depth_tensor.clamp(0.0, 1.0)
    depth_np = (depth_tensor.cpu().numpy().squeeze(0) * 255).astype(np.uint8)
    Image.fromarray(depth_np, mode='L').save(save_path)