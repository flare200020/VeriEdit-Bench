import json
from pathlib import Path
import numpy as np
from .core import SCORER_VERSION
from .preprocess import rgb2lab
TARGET_SCHEMA = 'target_meta/v1'
_RULE_PATH = Path(__file__).resolve().parent / 'targets_rule.json'
_DEFAULT_TAU = None

def load_tau():
    global _DEFAULT_TAU
    if _DEFAULT_TAU is None:
        _DEFAULT_TAU = dict(json.loads(_RULE_PATH.read_text())['params'])
    return _DEFAULT_TAU

def deltaE00(lab1, lab2):
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    (l1, a1, b1) = (lab1[..., 0], lab1[..., 1], lab1[..., 2])
    (l2, a2, b2) = (lab2[..., 0], lab2[..., 1], lab2[..., 2])
    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)
    cbar = (c1 + c2) / 2
    g = 0.5 * (1 - np.sqrt(cbar ** 7 / (cbar ** 7 + 25.0 ** 7)))
    a1p = (1 + g) * a1
    a2p = (1 + g) * a2
    c1p = np.sqrt(a1p * a1p + b1 * b1)
    c2p = np.sqrt(a2p * a2p + b2 * b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dl = l2 - l1
    dc = c2p - c1p
    dhp = h2p - h1p
    dhp = np.where(c1p * c2p == 0, 0.0, np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp)))
    dh = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dhp / 2))
    lbar = (l1 + l2) / 2
    cpbar = (c1p + c2p) / 2
    hsum = h1p + h2p
    hbar = np.where(c1p * c2p == 0, hsum, np.where(np.abs(h1p - h2p) <= 180, hsum / 2, np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2)))
    t = 1 - 0.17 * np.cos(np.radians(hbar - 30)) + 0.24 * np.cos(np.radians(2 * hbar)) + 0.32 * np.cos(np.radians(3 * hbar + 6)) - 0.2 * np.cos(np.radians(4 * hbar - 63))
    dtheta = 30 * np.exp(-((hbar - 275) / 25) ** 2)
    rc = 2 * np.sqrt(cpbar ** 7 / (cpbar ** 7 + 25.0 ** 7))
    sl = 1 + 0.015 * (lbar - 50) ** 2 / np.sqrt(20 + (lbar - 50) ** 2)
    sc = 1 + 0.045 * cpbar
    sh = 1 + 0.015 * cpbar * t
    rt = -np.sin(np.radians(2 * dtheta)) * rc
    return np.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (dh / sh) ** 2 + rt * (dc / sc) * (dh / sh))

def _hex_or_name_to_rgb(v):
    if not isinstance(v, str):
        return None
    s = v.strip().lstrip('#')
    if len(s) == 6:
        try:
            return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)
        except ValueError:
            pass
    try:
        from matplotlib import colors as mcolors
        (r, g, b, _a) = mcolors.to_rgba(v)
        return np.array([int(round(r * 255)), int(round(g * 255)), int(round(b * 255))], dtype=np.uint8)
    except Exception:
        return None

def _rgb_to_lab(rgb_u8):
    return rgb2lab(np.asarray(rgb_u8, dtype=np.uint8).reshape(1, 1, 3))[0, 0]

def _regions_from_meta(objects, lab_orig, allowed_mask):
    explicit = ['source_region_rle' in obj for obj in objects]
    if any(explicit) or any((obj.get('src_find_rule') == 'explicit_source_region_rle_v1' for obj in objects)):
        from .source_regions import regions_from_explicit
        return regions_from_explicit(objects, lab_orig, allowed_mask)
    rule = objects[0].get('src_find_rule', 'dE76<=8')
    src_labs = np.stack([_rgb_to_lab(_hex_or_name_to_rgb(o['from_resolved_hex'])) for o in objects])
    if rule == 'hue_sector_nn':
        ab_o = lab_orig[..., 1:3]
        ab_s = src_labs[:, 1:3]
        dist = np.sqrt(((ab_o[..., None, :] - ab_s) ** 2).sum(axis=-1))
        assign = dist.argmin(axis=-1)
        return ([allowed_mask & (assign == p) for p in range(len(objects))], rule)
    regions = []
    for p in range(len(objects)):
        dE = np.sqrt(((lab_orig - src_labs[p]) ** 2).sum(axis=-1))
        regions.append(allowed_mask & (dE <= 8.0))
    return (regions, rule)

def _tgt_refs(obj):
    gr = obj['gt_render']
    peaks = gr.get('tgt_peaks')
    if gr.get('repr_kind') == 'multi_hist' and peaks:
        labs = np.asarray([pk[:3] for pk in peaks], dtype=np.float64)
        shs = np.asarray([pk[3] for pk in peaks], dtype=np.float64)
        return (labs, shs, 'multi_hist')
    return (np.asarray([gr['tgt_lab_med']], dtype=np.float64), np.asarray([1.0]), 'median')

def _src_refs(obj):
    sr = obj.get('src_render')
    if sr and sr.get('repr_kind') == 'multi_hist' and sr.get('src_peaks'):
        labs = np.asarray([pk[:3] for pk in sr['src_peaks']], dtype=np.float64)
        shs = np.asarray([pk[3] for pk in sr['src_peaks']], dtype=np.float64)
        return (labs, shs, 'multi_hist')
    return (np.asarray([obj['gt_render']['src_lab_med']], dtype=np.float64), np.asarray([1.0]), 'median')

def _dE00_to_refs(px, refs_lab):
    d = deltaE00(px[:, None, :], refs_lab[None, :, :])
    return d.min(axis=1)

def _tgt_eq_bg_nondiscriminative(tgt_labs, tgt_shares, bg_lab, tau_gap):
    gaps = deltaE00(tgt_labs, bg_lab[None, :])
    near = float(tgt_shares[gaps <= tau_gap].sum())
    return (near >= 0.9, gaps, near)

def _core_from_region(region, kind):
    import cv2
    ru = region.astype(np.uint8)
    if kind == 'skeleton':
        dt = cv2.distanceTransform(ru, cv2.DIST_L2, 5)
        if dt.max() <= 0:
            return np.zeros_like(region)
        nb = cv2.dilate(dt, np.ones((3, 3), np.uint8))
        return (dt >= nb - 1e-06) & (dt >= 1.0)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.erode(ru, k, iterations=2).astype(bool)

def _na(coverage_status, meta=None, note=None):
    det = {'coverage_status': coverage_status, 'attain': None, 'n_objects': 0, 'per_object': [], 'scorer': SCORER_VERSION, 'target_schema': TARGET_SCHEMA, 'family': (meta or {}).get('family'), 'uncertifiable_reason': (meta or {}).get('uncertifiable_reason')}
    if note:
        det['note'] = note
    return (None, det)

def target_attainment(pred_path, gt_path, source_path, edit_meta, extras, case):
    try:
        return _target_attainment_inner(pred_path, gt_path, source_path, edit_meta, extras, case)
    except Exception as e:
        import traceback
        tb = ' | '.join(traceback.format_exc().splitlines()[-2:])
        meta = (extras or {}).get('target_meta')
        det = {'coverage_status': 'error:scorer_internal', 'attain': None, 'n_objects': 0, 'per_object': [], 'scorer': SCORER_VERSION, 'target_schema': TARGET_SCHEMA, 'family': (meta or {}).get('family'), 'uncertifiable_reason': (meta or {}).get('uncertifiable_reason'), 'note': f'{type(e).__name__}: {e} | {tb}'[:600]}
        return (None, det)
def _nearest_reference_distances(pixels, references):
    (distances, assignments) = ([], [])
    for start in range(0, len(pixels), 4096):
        d = deltaE00(pixels[start:start + 4096, None, :], references[None, :, :])
        ix = d.argmin(axis=1)
        assignments.append(ix)
        distances.append(d[np.arange(len(ix)), ix])
    return (np.concatenate(distances), np.concatenate(assignments))

def _core_target_references(extras, tau):
    import copy
    meta = copy.deepcopy(extras['target_meta'])
    objects = meta['objects']
    (regions, _) = _regions_from_meta(objects, extras['lab_orig'], extras['allowed_mask'])
    for (obj, region) in zip(objects, regions):
        core = _core_from_region(region, obj['core_kind'])
        pixels = extras['lab_gt'][core].astype(np.float64)
        if not len(pixels):
            continue
        (refs, _, kind) = _tgt_refs(obj)
        tolerance = tau['tau_pass']
        separation = obj['gt_render'].get('src_to_tgt_dE00')
        if tau.get('weak_sep_mode', 'abs') == 'relative' and separation is not None:
            tolerance = min(tolerance, tau['weak_sep_alpha'] * float(separation))
        (ds, _) = _nearest_reference_distances(pixels, refs)
        if kind != 'multi_hist' and np.mean(ds <= tolerance) >= tau['cov_hi']:
            continue
        (colors, counts) = np.unique(pixels, axis=0, return_counts=True)
        (ds, assignments) = _nearest_reference_distances(colors, refs)
        while np.any(ds > tolerance):
            ix = int(np.argmax(np.where(ds > tolerance, counts, -1)))
            refs = np.vstack([refs, colors[ix]])
            d = deltaE00(colors, colors[ix][None, :])
            assignments[d < ds] = len(refs) - 1
            ds = np.minimum(ds, d)
            if len(refs) > 2048:
                raise RuntimeError('Target reference exceeds 2048 colors')
        weights = np.bincount(assignments, weights=counts, minlength=len(refs))
        refs = refs[weights > 0]
        (ds, assignments) = _nearest_reference_distances(colors, refs)
        shares = np.bincount(assignments, weights=counts, minlength=len(refs)) / len(pixels)
        assert np.max(ds) <= tolerance + 1e-09 and abs(shares.sum() - 1) < 1e-10
        obj['gt_render']['repr_kind'] = 'multi_hist'
        obj['gt_render']['tgt_peaks'] = [[*map(float, color), float(share)] for (color, share) in zip(refs, shares)]
    return meta

def _target_attainment_inner(pred_path, gt_path, source_path, edit_meta, extras, case):
    tau = load_tau()
    meta = extras.get('target_meta')
    if meta is None:
        return _na('unavailable:no_target_meta', None, 'target metadata unavailable')
    cov_status = meta.get('coverage_status')
    if cov_status != 'covered':
        return _na(cov_status or 'unknown', meta)
    core_union = extras.get('target_core')
    if core_union is None:
        return _na('unavailable:no_target_core_mask', meta, 'target core mask unavailable')
    meta = _core_target_references(extras, tau)
    (score, detail) = _attain_impl(extras['lab_pred'], extras['lab_orig'], extras['allowed_mask'], core_union, meta, tau)
    return (score, detail)

def _attain_impl(lab_p, lab_o, allowed, core_union, meta, tau):
    cov_status = meta.get('coverage_status')
    objects = meta['objects']
    (regions, rule) = _regions_from_meta(objects, lab_o, allowed)
    per_object = []
    attain_vals = []
    px_recon_union = np.zeros_like(allowed)
    n_core_px_match = 0
    for (p, obj) in enumerate(objects):
        gr = obj['gt_render']
        (tgt_labs, tgt_shares, tgt_kind) = _tgt_refs(obj)
        (src_labs, _src_sh, src_kind) = _src_refs(obj)
        bg_lab = obj.get('bg_ring_lab_med')
        bg_lab = np.asarray(bg_lab, dtype=np.float64) if bg_lab is not None else None
        d_s2t = gr.get('src_to_tgt_dE00')
        tau_pass_eff = tau['tau_pass']
        if tau.get('weak_sep_mode', 'abs') == 'relative' and d_s2t is not None:
            tau_pass_eff = min(tau['tau_pass'], tau['weak_sep_alpha'] * float(d_s2t))
        region = regions[p]
        core = _core_from_region(region, obj.get('core_kind'))
        core_px = int(core.sum())
        px_recon_union |= core
        rec = {'pair_id': int(obj['pair_id']), 'core_px': core_px, 'core_px_recorded': obj.get('core_px'), 'core_px_match': bool(core_px == obj.get('core_px')), 'region_px': int(region.sum()), 'core_kind': obj.get('core_kind'), 'src_find_rule': rule, 'repr_kind_tgt': tgt_kind, 'repr_kind_src': src_kind, 'n_tgt_refs': int(len(tgt_labs)), 'n_src_refs': int(len(src_labs)), 'tau_pass_eff': round(float(tau_pass_eff), 4), 'src_to_tgt_dE00': round(float(d_s2t), 4) if d_s2t is not None else None}
        n_core_px_match += int(core_px == obj.get('core_px'))
        if core_px == 0:
            rec.update({'state': 'unscored_empty_core', 'attain': None, 'med_dE00_pred_vs_gtTarget': None, 'p90_dE00_pred_vs_gtTarget': None, 'cov_pred_at_target': None, 'src_residue': None, 'bg_exposed': None, 'vanished': False})
            per_object.append(rec)
            continue
        px = lab_p[core].astype(np.float64)
        d_tgt = _dE00_to_refs(px, tgt_labs)
        d_src = _dE00_to_refs(px, src_labs)
        med_tgt = float(np.median(d_tgt))
        med_src = float(np.median(d_src))
        cov = float(np.mean(d_tgt <= tau_pass_eff))
        src_res = float(np.mean(d_src <= tau['tau_src']))
        peak_attain = None
        peak_detail = None
        peak_missing = []
        if tgt_kind == 'multi_hist' and len(tgt_labs):
            dmat = deltaE00(px[:, None, :], tgt_labs[None, :, :])
            assign = dmat.argmin(axis=1)
            inband = d_tgt <= tau_pass_eff
            maj = float(tau.get('peak_major_share', 0.08))
            miss_sh = float(tau.get('peak_missing_share', 0.1))
            miss_thr = float(tau.get('peak_miss_ratio', 0.5))
            kpx = float(len(d_tgt))
            peak_detail = []
            ratios = []
            for (r, s_r) in enumerate(tgt_shares):
                sh_att = float(((assign == r) & inband).sum() / kpx)
                denom = max(float(s_r), maj)
                ratio = sh_att / denom
                peak_detail.append({'peak': r, 'share_recorded': round(float(s_r), 6), 'share_attained': round(sh_att, 6), 'attain_ratio': round(ratio, 6)})
                if float(s_r) >= maj:
                    ratios.append(ratio)
                if float(s_r) >= miss_sh and ratio < miss_thr:
                    peak_missing.append(r)
            peak_attain = float(min(ratios)) if ratios else 1.0
        if bg_lab is not None:
            d_bg = deltaE00(px, bg_lab[None, :])
            med_bg = float(np.median(d_bg))
            bg_exp = float(np.mean(d_bg <= tau['tau_bg']))
            reg_px = lab_p[region].astype(np.float64)
            new_ink = float(np.mean(deltaE00(reg_px, bg_lab[None, :]) > tau['tau_bg']))
            (tgt_eq_bg, _gaps, _nearsh) = _tgt_eq_bg_nondiscriminative(tgt_labs, tgt_shares, bg_lab, tau.get('tgt_eq_bg_de00', 5.0))
            if tgt_eq_bg:
                rec['tgt_eq_bg_nondiscriminative'] = True
        else:
            (med_bg, bg_exp, new_ink) = (None, None, None)
            tgt_eq_bg = False
            rec['tgt_eq_bg_nondiscriminative'] = None
        vanished = bool(bg_exp is not None and (not tgt_eq_bg) and (bg_exp >= tau['vanished_bg_min']) and (new_ink is not None) and (new_ink <= tau['vanished_new_ink_max']))
        refs = {'target': (med_tgt, tau_pass_eff), 'src': (med_src, tau['tau_src'])}
        if med_bg is not None and (not tgt_eq_bg):
            refs['bg'] = (med_bg, tau['tau_bg'])
        explaining = [k for (k, (med, tol)) in refs.items() if med <= tol]
        if vanished:
            state = 'vanished'
        elif not explaining:
            state = 'wrong_color'
        else:
            closest = min(refs.items(), key=lambda kv: kv[1][0])[0]
            if closest == 'src':
                state = 'source_residue'
            elif closest == 'bg':
                state = 'bg_exposed'
            else:
                state = 'ok' if cov >= tau.get('ok_cov_min', 0.5) else 'partial'
        states_all = [state]
        if peak_missing and state != 'vanished':
            states_all = ['peak_missing', state]
            state = 'peak_missing'
        lo = float(tau.get('cov_lo', 0.0))
        hi = float(tau.get('cov_hi', 1.0))
        if hi <= lo:
            hi = lo + 1e-09
        if tgt_kind == 'multi_hist' and peak_detail is not None:
            wsum = 0.0
            for dd in peak_detail:
                s_r = dd['share_recorded']
                s_hat = dd['share_attained']
                if s_r > 0:
                    wsum += s_r * min(s_hat / s_r, 1.0)
            obj_attain = 100.0 * min(max(wsum, 0.0), 1.0)
        else:
            obj_attain = 100.0 * min(max((cov - lo) / (hi - lo), 0.0), 1.0)
        hard_zero_reason = None
        if state == 'vanished':
            obj_attain = 0.0
            hard_zero_reason = 'vanished'
        rec.update({'state': state, 'states_all': states_all, 'attain': round(obj_attain, 4), 'hard_zero': hard_zero_reason, 'peak_attain': round(peak_attain, 6) if peak_attain is not None else None, 'peak_detail': peak_detail, 'med_dE00_pred_vs_gtTarget': round(med_tgt, 4), 'p90_dE00_pred_vs_gtTarget': round(float(np.percentile(d_tgt, 90)), 4), 'cov_pred_at_target': round(cov, 6), 'src_residue': round(src_res, 6), 'med_dE00_pred_vs_src': round(med_src, 4), 'bg_exposed': round(bg_exp, 6) if bg_exp is not None else None, 'med_dE00_pred_vs_bg': round(med_bg, 4) if med_bg is not None else None, 'new_ink_frac_in_tgt_region': round(new_ink, 6) if new_ink is not None else None, 'vanished': vanished})
        per_object.append(rec)
        attain_vals.append(obj_attain)
    if not attain_vals:
        return _na(cov_status, meta, 'all object cores are empty')
    attain = float(np.mean(attain_vals))
    attain_min = float(np.min(attain_vals))
    detail = {'coverage_status': cov_status, 'n_objects': len(objects), 'n_scored_objects': len(attain_vals), 'attain': round(float(attain), 4), 'attain_min': round(attain_min, 4), 'aggregate': 'mean_over_objects', 'per_object': per_object, 'tau_pass': tau['tau_pass'], 'tau_src': tau['tau_src'], 'tau_bg': tau['tau_bg'], 'cov_lo': tau.get('cov_lo', 0.0), 'cov_hi': tau.get('cov_hi', 1.0), 'weak_sep_mode': tau.get('weak_sep_mode', 'abs'), 'cov_map': 'tau_band_linear' if tau.get('cov_lo', 0.0) > 0 or tau.get('cov_hi', 1.0) < 1.0 else tau.get('cov_map'), 'mask_integrity': {'target_core_union_px': int(core_union.sum()), 'reconstructed_union_px': int(px_recon_union.sum()), 'union_xor_px': int((px_recon_union ^ core_union).sum()), 'per_object_core_px_match': f'{n_core_px_match}/{len(objects)}'}, 'family': meta.get('family'), 'edit_family': meta.get('edit_family'), 'uncertifiable_reason': meta.get('uncertifiable_reason'), 'scorer': SCORER_VERSION, 'target_schema': TARGET_SCHEMA}
    if meta.get('merged_target_ambiguous'):
        detail['merged_target_ambiguous'] = True
    return (round(float(attain), 4), detail)
