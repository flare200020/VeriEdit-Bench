import numpy as np
from .preprocess import TOLERANCES
SCORER_VERSION = '1.0'
TAU_EFF = 6.0
R_MIN_GATE = 0.05
MAG_R_BAND = 0.15
TAU_MEDIAN_GATE = 8.0
HI_DELTA_GATE_E = 20.0
HI_OUTSIDE_MASK_RATIO = 2.0
COHERENCE_MIN = 0.25
COHERENCE_BLOCK = 16
REBLUR_SIGMAS = (1.0, 2.0, 3.0)
REBLUR_MEDIAN_MAX = 2.0
STRUCTURAL_GATE_R_MAX = 0.9
BG_MIN_AREA_FRAC = 0.0005
BG_CAP_MULT = 4.0
GX_BG_RESID_MAX = 6.0
GX_MIN_BG_PX = 4096
GX_SAMPLE_MAX = 65536
GX_BG_COV_SMIN_MIN = 0.3
GX_A_ENVELOPE = 1.0
GX_T_ENVELOPE = 40.0
GX_CORE_RESID_MAX = 2.0
REGISTRY = {}

def register_core(fn):
    REGISTRY[fn.__name__] = fn
    return fn

def _alphabeta(fid_raw, base):
    return min(max((fid_raw - base) / (1.0 - base), 0.0), 1.0)

@register_core
def edit_fidelity(pred_path, gt_path, source_path, edit_meta, extras):
    import cv2
    from .preprocess import FIDELITY_TOLERANCES, rgb2lab, deltaE76
    mask = extras['mask']
    if mask.sum() == 0:
        return (0.0, {'reason': 'empty mask'})
    pred_blur = cv2.GaussianBlur(extras['pred'], (3, 3), 0.6)
    gt_blur = cv2.GaussianBlur(extras['gt'], (3, 3), 0.6)
    orig_blur = cv2.GaussianBlur(extras['orig'], (3, 3), 0.6)
    mask_d = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1).astype(bool)
    gates = _did_change_gates(extras)
    if gates['gate'] is not None:
        return (0.0, {'reason': f"gated:{gates['gate']}", 'scorer': SCORER_VERSION})
    lab_p = rgb2lab(pred_blur)
    lab_g = rgb2lab(gt_blur)
    lab_o = rgb2lab(orig_blur)
    dE_pg = deltaE76(lab_p, lab_g)
    dE_og = deltaE76(lab_o, lab_g)
    per_t = [float(np.mean(dE_pg[mask_d] <= t)) for t in FIDELITY_TOLERANCES]
    fid_raw = float(np.mean(per_t))
    base = float(np.mean([float(np.mean(dE_og[mask_d] <= t)) for t in FIDELITY_TOLERANCES]))
    detail = {'fidelity_pass_rate_raw': fid_raw, 'per_tolerance': per_t, 'tolerances': list(FIDELITY_TOLERANCES), 'baseline_do_nothing': base, 'fidelity_baseline': extras.get('fidelity_baseline'), 'blur_sigma': 0.6, 'region_dilate_px': 1, 'scorer': SCORER_VERSION}
    if base > 0.995:
        detail.update({'normalised': 0.0, 'reason': 'fidelity_saturated_baseline'})
        return (0.0, detail)
    score01 = _alphabeta(fid_raw, base)
    detail['normalised'] = score01
    return (100.0 * score01, detail)

@register_core
def preservation(pred_path, gt_path, source_path, edit_meta, extras):
    dE_po = extras['dE_pred_orig']
    outside = extras['outside_mask']
    if outside.sum() == 0:
        return (100.0, {'note': 'mask covers whole image; preservation vacuous', 'preservation_pass_rate': 1.0})
    per_t = [float(np.mean(dE_po[outside] <= t)) for t in TOLERANCES]
    return (100.0 * float(np.mean(per_t)), {'preservation_pass_rate': float(np.mean(per_t)), 'per_tolerance': per_t})

def _clean_background(cand, mask):
    import cv2
    outside = (cand & ~mask).astype(np.uint8)
    (H, W) = cand.shape
    min_area = max(16, int(BG_MIN_AREA_FRAC * H * W))
    (n, lab, stats, _) = cv2.connectedComponentsWithStats(outside, connectivity=8)
    keep = np.zeros_like(outside)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    return (keep.astype(bool), min_area)

def _did_change_gates(extras):
    if '_gates' in extras:
        return extras['_gates']
    import cv2
    from .preprocess import rgb2lab, deltaE76
    mask = extras['mask']
    lab_o = extras['lab_orig']
    lab_p = extras['lab_pred']
    lab_g = extras['lab_gt']
    dE_go = extras['dE_gt_orig']
    dE_po = extras['dE_pred_orig']
    core = mask & (dE_go > TAU_EFF)
    core_kind = 'tau_eff'
    if core.sum() < max(20, mask.sum() * 0.2):
        core = mask & (dE_go > 2.0)
        core_kind = 'tau2'
    if core.sum() < max(20, mask.sum() * 0.2):
        core = mask
        core_kind = 'mask'
    d_pred = np.abs((lab_p[core] - lab_o[core]).astype(np.float64)).sum()
    d_gt = np.abs((lab_g[core] - lab_o[core]).astype(np.float64)).sum()
    med_pred = float(np.median(dE_po[core])) if core.sum() else 0.0
    med_gt = float(np.median(dE_go[core])) if core.sum() else 0.0
    hi_out_area = float((~mask & (dE_po > HI_DELTA_GATE_E)).sum())
    hi_out_over_mask = hi_out_area / max(float(mask.sum()), 1.0)
    if d_gt < 1e-06:
        r = 1.0 if d_pred < 1e-06 else 0.0
    else:
        r = float(d_pred / d_gt)
    out = {'gate': None, 'core': core, 'core_kind': core_kind, 'r': r, 'med_pred': med_pred, 'med_gt': med_gt, 'coherence': None, 'hi_out_area_px': round(hi_out_area, 1), 'hi_out_over_mask': round(hi_out_over_mask, 4)}
    if mask.sum() == 0 or core.sum() == 0:
        out['gate'] = 'empty'
    elif r < R_MIN_GATE:
        out['gate'] = 'r<0.05'
    else:
        thr = min(REBLUR_MEDIAN_MAX, 0.8 * med_gt) if med_gt > 0 else REBLUR_MEDIAN_MAX
        for s in REBLUR_SIGMAS:
            b = cv2.GaussianBlur(extras['orig'], (0, 0), s)
            lab_b = rgb2lab(b)
            med_b_pred = float(np.median(deltaE76(lab_b, lab_p)[core]))
            med_b_gt = float(np.median(deltaE76(lab_b, lab_g)[core]))
            if med_b_gt < thr:
                out['blur_confusable_gt'] = True
            residual_ratio = float(np.abs(lab_p[core].astype(np.float64) - lab_b[core]).sum()) / max(d_gt, 1e-06)
            if med_b_pred < thr and med_b_gt >= thr and (residual_ratio < R_MIN_GATE) and (med_b_pred < float(np.median(extras['dE_pred_gt'][core]))):
                out['gate'] = 'reblur-match'
                out['reblur_sigma'] = s
                break
        if out['gate'] is None and r < STRUCTURAL_GATE_R_MAX:
            (coh, _) = _block_coherence(lab_p, lab_o, core)
            out['coherence'] = round(coh, 4)
            if coh < COHERENCE_MIN:
                out['gate'] = 'coherence'
        if out['gate'] is None:
            gx = _global_xform_fit(lab_o, lab_p, lab_g, mask, core, med_gt)
            out['global_xform'] = gx
            if gx.get('fired'):
                out['gate'] = 'global-xform'
    extras['_gates'] = out
    return out

def _global_xform_fit(lab_o, lab_p, lab_g, mask, core, med_gt):
    (H, W) = mask.shape
    bg_idx = np.flatnonzero(~mask.ravel())
    if bg_idx.size < GX_MIN_BG_PX or mask.sum() > 0.8 * mask.size:
        return {'fired': False, 'reason': 'background_too_small', 'bg_px': int(bg_idx.size)}
    step = max(1, bg_idx.size // GX_SAMPLE_MAX)
    idx = bg_idx[::step][:GX_SAMPLE_MAX]
    O = lab_o.reshape(-1, 3)[idx].astype(np.float64)
    P = lab_p.reshape(-1, 3)[idx].astype(np.float64)
    Oc_cov = O - O.mean(axis=0)
    sv = np.linalg.svd(Oc_cov, compute_uv=False) / np.sqrt(max(len(O) - 1, 1))
    smin = float(sv[-1])
    if smin < GX_BG_COV_SMIN_MIN:
        (oL, pL) = (O[:, 0], P[:, 0])
        vl = float(np.var(oL))
        if vl < 1.0:
            return {'fired': False, 'reason': 'bg_underdetermined', 'bg_px': int(bg_idx.size), 'bg_cov_smin': round(smin, 4), 'bg_L_var': round(vl, 4)}
        a1 = float(np.cov(oL, pL, ddof=0)[0, 1] / vl)
        b1 = float(pL.mean() - a1 * oL.mean())
        med_bg1 = float(np.median(np.abs(a1 * oL + b1 - pL)))
        detail = {'bg_px': int(bg_idx.size), 'sample_px': int(idx.size), 'fit_kind': 'L1', 'bg_cov_smin': round(smin, 4), 'bg_L_var': round(vl, 3), 'a1': round(a1, 4), 'b1': round(b1, 3), 'med_resid_bg': round(med_bg1, 3)}
        if not (0.8 <= a1 <= 1.25 and abs(b1) <= 40.0 and (med_bg1 < GX_BG_RESID_MAX)):
            detail['reason'] = 'L_fit_not_global'
            return {'fired': False, **detail}
        fitted_core = lab_o[core].astype(np.float64).copy()
        fitted_core[:, 0] = a1 * fitted_core[:, 0] + b1
        target_strength = float(np.abs(lab_g[core].astype(np.float64) - lab_o[core]).sum())
        residual_ratio = float(np.abs(lab_p[core].astype(np.float64) - fitted_core).sum()) / max(target_strength, 1e-06)
        detail['residual_change_ratio'] = residual_ratio
        if residual_ratio >= R_MIN_GATE:
            return {'fired': False, 'reason': 'meaningful_change_remains_after_fit', **detail}
        res_c = np.abs(a1 * lab_o[core, 0].astype(np.float64) + b1 - lab_p[core, 0].astype(np.float64))
        med_res_core = float(np.median(res_c))
        bar = min(GX_CORE_RESID_MAX, 0.75 * med_gt) if med_gt > 0 else GX_CORE_RESID_MAX
        detail.update({'med_resid_core': round(med_res_core, 3), 'bar': round(bar, 3), 'med_gt': round(med_gt, 3)})
        if med_res_core >= bar:
            detail['reason'] = 'core_unexplained_by_T'
            return {'fired': False, **detail}
        res_g = np.abs(a1 * lab_o[core, 0].astype(np.float64) + b1 - lab_g[core, 0].astype(np.float64))
        med_res_gt = float(np.median(res_g))
        bar_gt = min(TAU_MEDIAN_GATE, 0.75 * med_gt) if med_gt > 0 else TAU_MEDIAN_GATE
        detail.update({'med_resid_gt': round(med_res_gt, 3), 'bar_gt': round(bar_gt, 3)})
        if med_res_gt < bar_gt:
            detail['reason'] = 'gt_is_transform_like'
            return {'fired': False, **detail}
        return {'fired': True, **detail}
    X = np.concatenate([O, np.ones((O.shape[0], 1))], axis=1)
    (Wmat, *_) = np.linalg.lstsq(X, P, rcond=None)
    a_off = float(np.abs(Wmat[:3] - np.eye(3)).max())
    t_abs = float(np.abs(Wmat[3]).max())
    if a_off > GX_A_ENVELOPE or t_abs > GX_T_ENVELOPE:
        return {'fired': False, 'reason': 'transform_out_of_envelope', 'bg_px': int(bg_idx.size), 'bg_cov_smin': round(smin, 4), 'A_offdiag_max': round(a_off, 4), 't_abs_max': round(t_abs, 3)}
    resid_bg = np.sqrt(((X @ Wmat - P) ** 2).sum(axis=1))
    med_bg = float(np.median(resid_bg))
    detail = {'bg_px': int(bg_idx.size), 'sample_px': int(idx.size), 'med_resid_bg': round(med_bg, 3), 'bg_cov_smin': round(smin, 4), 'A_offdiag_max': round(a_off, 4), 't_abs_max': round(t_abs, 3)}
    if med_bg >= GX_BG_RESID_MAX:
        detail['reason'] = 'no_global_transform_fit'
        return {'fired': False, **detail}
    Oc = lab_o[core].astype(np.float64)
    Pc = lab_p[core].astype(np.float64)
    Xc = np.concatenate([Oc, np.ones((Oc.shape[0], 1))], axis=1)
    target_strength = float(np.abs(lab_g[core].astype(np.float64) - Oc).sum())
    residual_ratio = float(np.abs(Pc - Xc @ Wmat).sum()) / max(target_strength, 1e-06)
    detail['residual_change_ratio'] = residual_ratio
    if residual_ratio >= R_MIN_GATE:
        return {'fired': False, 'reason': 'meaningful_change_remains_after_fit', **detail}
    med_res_core = float(np.median(np.sqrt(((Xc @ Wmat - Pc) ** 2).sum(axis=1))))
    bar = min(GX_CORE_RESID_MAX, 0.75 * med_gt) if med_gt > 0 else GX_CORE_RESID_MAX
    detail.update({'med_resid_core': round(med_res_core, 3), 'bar': round(bar, 3), 'med_gt': round(med_gt, 3)})
    if med_res_core >= bar:
        detail['reason'] = 'core_unexplained_by_T'
        return {'fired': False, **detail}
    bar_gt = min(TAU_MEDIAN_GATE, 0.75 * med_gt) if med_gt > 0 else TAU_MEDIAN_GATE
    Gc = lab_g[core].astype(np.float64)
    med_res_gt = float(np.median(np.sqrt(((Xc @ Wmat - Gc) ** 2).sum(axis=1))))
    detail['med_resid_gt'] = round(med_res_gt, 3)
    detail['bar_gt'] = round(bar_gt, 3)
    if med_res_gt < bar_gt:
        detail['reason'] = 'gt_is_transform_like'
        return {'fired': False, **detail}
    return {'fired': True, **detail}

@register_core
def localization(pred_path, gt_path, source_path, edit_meta, extras):
    dE_po = extras['dE_pred_orig']
    dE_go = extras['dE_gt_orig']
    mask = extras['mask']
    if mask.sum() == 0:
        return (0.0, {'reason': 'empty mask'})
    gates = _did_change_gates(extras)
    if gates['gate'] is not None:
        return (0.0, {'gate': gates['gate'], 'recall': 0.0, 'precision': 0.0, 'scorer': SCORER_VERSION})
    gt6 = dE_go > TAU_EFF
    tau = TAU_EFF if gt6.sum() >= max(16, 0.05 * mask.sum()) else 2.0
    ch_gt = dE_go > tau
    if ch_gt.sum() == 0:
        ch_gt = mask
    cand = dE_po > tau
    (keep, min_area) = _clean_background(cand, mask)
    ch = cand & mask | keep
    detail = {'tau_loc': tau, 'changemap_area': int(ch.sum()), 'gt_changemap_area': int(ch_gt.sum()), 'background_pixels_after_cleanup': int((keep & ~mask).sum()), 'min_component_area': int(min_area), 'scorer': SCORER_VERSION}
    if ch.sum() == 0:
        detail.update({'recall': 0.0, 'precision': 0.0})
        return (0.0, detail)
    recall = float((ch & ch_gt).sum() / ch_gt.sum())
    px_in = float((ch & mask).sum())
    px_bg = float((ch & ~mask).sum())
    bg_capped = min(px_bg, BG_CAP_MULT * float(mask.sum()))
    precision = px_in / (px_in + bg_capped) if px_in + bg_capped > 0 else 0.0
    detail.update({'recall': recall, 'precision': precision, 'background_pixels_raw': int(px_bg), 'background_pixels_capped': int(bg_capped)})
    if recall <= 0.0 or precision <= 0.0:
        return (0.0, detail)
    f1 = 2.0 * recall * precision / (recall + precision)
    return (100.0 * f1, detail)

def _block_coherence(lab_p, lab_o, region, block=COHERENCE_BLOCK):
    import cv2
    k = 2 * (block // 2) + 1
    region = cv2.dilate(region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))).astype(bool)
    d = lab_p.astype(np.float64) - lab_o.astype(np.float64)
    d[~region] = 0.0
    (H, W) = region.shape
    (ph, pw) = (-H % block, -W % block)
    if ph or pw:
        d = np.pad(d, ((0, ph), (0, pw), (0, 0)))
    (Hb, Wb) = (d.shape[0] // block, d.shape[1] // block)
    tiles = d.reshape(Hb, block, Wb, block, 3).sum(axis=(1, 3))
    denom = np.abs(d).sum(axis=(0, 1))
    numer = np.abs(tiles).sum(axis=(0, 1))
    ratios = [float(numer[c] / denom[c]) if denom[c] > 1e-09 else 0.0 for c in range(3)]
    return (max(ratios), ratios)

@register_core
def magnitude(pred_path, gt_path, source_path, edit_meta, extras):
    mask = extras['mask']
    if mask.sum() == 0:
        return (0.0, {'reason': 'empty mask'})
    lab_o = extras['lab_orig']
    lab_p = extras['lab_pred']
    lab_g = extras['lab_gt']
    gates = _did_change_gates(extras)
    core = gates['core']
    r = gates['r']
    detail = {'r': r, 'med_dE_pred': round(gates['med_pred'], 3), 'med_dE_gt': round(gates['med_gt'], 3), 'core_kind': gates['core_kind'], 'core_px': int(core.sum()), 'coherence': gates['coherence'], 'hi_out_over_mask': gates.get('hi_out_over_mask'), 'scorer': SCORER_VERSION}
    if gates['gate'] is not None:
        detail.update({'r_score': 0.0, 'magnitude_only': 0.0, 'direction_cos': None, 'did_change': False, 'gate': gates['gate']})
        return (0.0, detail)
    dev = abs(r - 1.0)
    r_score = 1.0 if dev <= MAG_R_BAND else max(0.0, 1.0 - (dev - MAG_R_BAND))
    v_pred = (lab_p[core].astype(np.float64) - lab_o[core]).ravel()
    v_gt = (lab_g[core].astype(np.float64) - lab_o[core]).ravel()
    (n_pred, n_gt) = (float(np.linalg.norm(v_pred)), float(np.linalg.norm(v_gt)))
    if n_pred < 1e-06 or n_gt < 1e-06:
        direction = 0.0 if (n_pred < 1e-06) != (n_gt < 1e-06) else 1.0
    else:
        direction = float((v_pred * v_gt).sum() / (n_pred * n_gt))
        direction = max(-1.0, min(1.0, direction))
    magnitude_only = 100.0 * r_score
    mag_score = 100.0 * (0.5 * r_score + 0.5 * ((direction + 1.0) / 2.0))
    detail.update({'r_score': r_score, 'magnitude_only': magnitude_only, 'direction_cos': direction, 'direction_score': (direction + 1.0) / 2.0, 'did_change': True, 'gate': None})
    return (mag_score, detail)
