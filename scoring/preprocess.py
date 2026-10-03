import numpy as np
from PIL import Image
TOLERANCES = (2.0, 5.0, 10.0, 20.0)
FIDELITY_TOLERANCES = (5.0, 10.0, 20.0)
_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.072175], [0.0193339, 0.119192, 0.9503041]])

def rgb2lab(arr_u8):
    rgb = arr_u8.astype(np.float32) / 255.0
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = rgb @ _M.T.astype(np.float32)
    (Xn, Yn, Zn) = (0.95047, 1.0, 1.08883)
    xyz[..., 0] /= Xn
    xyz[..., 2] /= Zn
    (eps, kap) = (0.008856, 903.3)
    fxyz = np.where(xyz > eps, xyz ** (1 / 3), (kap * xyz + 16) / 116)
    (fx, fy, fz) = (fxyz[..., 0], fxyz[..., 1], fxyz[..., 2])
    L = np.where(xyz[..., 1] > eps, 116 * fy - 16, kap * xyz[..., 1])
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1).astype(np.float32)

def deltaE76(lab_a, lab_b):
    d = lab_a - lab_b
    return np.sqrt((d * d).sum(axis=-1)).astype(np.float32)

def load_triple(source_path, gt_path, pred_path, mask_path):
    orig = Image.open(source_path).convert('RGB')
    gt = Image.open(gt_path).convert('RGB')
    orig = np.asarray(orig, dtype=np.uint8)
    gt = np.asarray(gt, dtype=np.uint8)
    detail = {'scale_applied': None, 'letterboxed': False, 'pred_size_original': None, 'gt_size': list(gt.shape[:2])}
    pred = Image.open(pred_path).convert('RGB')
    detail['pred_size_original'] = list(pred.size)
    if pred.size != (gt.shape[1], gt.shape[0]):
        (ow, oh) = pred.size
        (tw, th) = (gt.shape[1], gt.shape[0])
        (ar_p, ar_g) = (ow / oh, tw / th)
        if abs(ar_p - ar_g) > 0.02:
            detail['letterboxed'] = True
            if ar_p > ar_g:
                new_h = int(round(ow / ar_g))
                pad_top = (new_h - oh) // 2
                canvas = Image.new('RGB', (ow, new_h), (0, 0, 0))
                canvas.paste(pred, (0, pad_top))
                detail['scale_applied'] = [tw / ow, th / new_h]
            else:
                new_w = int(round(oh * ar_g))
                pad_left = (new_w - ow) // 2
                canvas = Image.new('RGB', (new_w, oh), (0, 0, 0))
                canvas.paste(pred, (pad_left, 0))
                detail['scale_applied'] = [tw / new_w, th / oh]
            pred = canvas.resize((tw, th), Image.BICUBIC)
        else:
            pred = pred.resize((tw, th), Image.BICUBIC)
            detail['scale_applied'] = [tw / ow, th / oh]
    pred = np.asarray(pred, dtype=np.uint8)
    mask = np.asarray(Image.open(mask_path).convert('L')) > 127
    return (orig, gt, pred, mask, detail)
