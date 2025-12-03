import torch
import os
import io
import lpips
import warnings
import contextlib
from .pohsun_ssim import pytorch_ssim

# 屏蔽所有额外输出（如用户未显式 print 的）
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"

def quiet_lpips_model():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return lpips.LPIPS(net="alex")

# get lpips score
def lpips_score(image_pred, image_gt):
    # lpips_loss = lpips.LPIPS(net="alex")
    lpips_loss = quiet_lpips_model()  # 屏蔽LPIPS初始化输出
    lpips_loss = lpips_loss.to(image_pred.device)
    with torch.no_grad():
        lpips_value = lpips_loss(image_pred*2-1, image_gt*2-1).item()
    return lpips_value

# get PSNR score
def psnr_score(image_pred, image_gt, valid_mask=None, reduction='mean'):
    def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
        value = (image_pred-image_gt)**2
        if valid_mask is not None:
            value = value[valid_mask]
        if reduction == 'mean':
            return torch.mean(value)
        return value
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

# get SSIM score
def ssim_score(image_pred, image_gt):
    image_pred = image_pred.unsqueeze(0)
    image_gt = image_gt.unsqueeze(0)
    ssim = pytorch_ssim.ssim(image_pred, image_gt).item()
    return ssim

