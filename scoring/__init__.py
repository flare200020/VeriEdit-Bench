from .core import REGISTRY as CORE_REGISTRY

def get_core_axes():
    return dict(CORE_REGISTRY)

def load_extras(case, pred_path_override=None):
    import json as _json
    import numpy as np
    from PIL import Image
    from .preprocess import load_triple, rgb2lab, deltaE76
    base = case['_dir']
    (orig, gt, pred, mask, det) = load_triple(base / case['source_image'], base / case['gt_image'], pred_path_override or case['pred_path'], base / case['gt_mask'])
    lab_o = rgb2lab(orig)
    lab_g = rgb2lab(gt)
    lab_p = rgb2lab(pred)
    ext = {'orig': orig, 'gt': gt, 'pred': pred, 'mask': mask, 'outside_mask': ~mask, 'lab_orig': lab_o, 'lab_gt': lab_g, 'lab_pred': lab_p, 'dE_pred_gt': deltaE76(lab_p, lab_g), 'dE_pred_orig': deltaE76(lab_p, lab_o), 'dE_gt_orig': deltaE76(lab_g, lab_o), 'fidelity_baseline': case.get('fidelity_baseline', 0.0), 'prep_detail': det}
    notes = {}
    am_rel = case.get('allowed_mask')
    if am_rel and am_rel != case['gt_mask']:
        allowed = np.asarray(Image.open(base / am_rel).convert('L')) > 127
        notes['allowed_mask_from'] = am_rel
    else:
        allowed = mask
        notes['allowed_mask_from'] = 'gt_mask'

    def _opt_mask(rel, name):
        if not rel:
            return None
        try:
            m = np.asarray(Image.open(base / rel).convert('L')) > 127
        except OSError as e:
            notes[f'{name}_dropped'] = f'load failed: {e}'
            return None
        if m.shape != mask.shape:
            notes[f'{name}_dropped'] = f'shape {m.shape} != gt_mask {mask.shape}'
            return None
        return m
    target_meta = None
    if case.get('target_meta'):
        try:
            target_meta = _json.loads((base / case['target_meta']).read_text())
        except Exception as e:
            notes['target_meta_error'] = f'{type(e).__name__}: {e}'
    ext.update({'allowed_mask': allowed, 'target_core': _opt_mask(case.get('target_core_mask'), 'target_core'), 'target_pairs': _opt_mask(case.get('target_pairs_mask'), 'target_pairs'), 'target_meta': target_meta, 'mask_notes': notes})
    return ext
