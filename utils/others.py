import torch
import numpy as np

def SE3_exp(tau):
    device = tau.device
    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho
    T = torch.eye(4, device=device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )

def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V
    
def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm

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
