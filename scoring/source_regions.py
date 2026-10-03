import hashlib, re
import numpy as np
SCHEMA = 'source_region_rle/v1'
RULE = 'explicit_source_region_rle_v1'

def decode_region(record, allowed_mask):
    if not isinstance(record, dict) or record.get('schema') != SCHEMA or record.get('order') != 'C':
        raise ValueError('Invalid source region schema or order')
    shape = record.get('shape')
    allowed = np.asarray(allowed_mask)
    if not isinstance(shape, list) or len(shape) != 2 or any((type(x) is not int or x <= 0 for x in shape)) or (tuple(shape) != allowed.shape):
        raise ValueError('Source region shape does not match allowed mask')
    runs = record.get('runs')
    if not isinstance(runs, list):
        raise ValueError('Source region is missing runs')
    area = record.get('area_px')
    if type(area) is not int or area <= 0 or area > allowed.size:
        raise ValueError('Invalid source region area')
    flat = np.zeros(allowed.size, dtype=np.uint8)
    end = -1
    count = 0
    for r in runs:
        if not isinstance(r, list) or len(r) != 2 or any((type(x) is not int for x in r)):
            raise ValueError('Each source region run must be two integers')
        (s, n) = r
        if s < 0 or n <= 0 or s + n > flat.size or (s <= end):
            raise ValueError('Source region runs are out of range, overlapping, or unordered')
        flat[s:s + n] = 1
        count += n
        end = s + n
    if count != area:
        raise ValueError('Source region area does not match runs')
    digest = record.get('mask_sha256')
    if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest) or hashlib.sha256(flat.tobytes()).hexdigest() != digest:
        raise ValueError('Source region hash mismatch')
    out = flat.reshape(shape).astype(bool)
    if np.any(out & ~allowed.astype(bool)):
        raise ValueError('Source region extends outside allowed mask')
    proof = record.get('provenance')
    required = ['source_image_sha256', 'full_gt_image_sha256', 'source_html_sha256', 'isolated_edit_html_sha256', 'isolated_edit_image_sha256', 'isolated_repeat_image_sha256']
    if not isinstance(proof, dict) or any((not isinstance(proof.get(k), str) or not re.fullmatch('[0-9a-f]{64}', proof[k]) for k in required)):
        raise ValueError('Source region is missing provenance hashes')
    if proof['isolated_edit_image_sha256'] != proof['isolated_repeat_image_sha256']:
        raise ValueError('Source region renders are inconsistent')
    if not isinstance(proof.get('mapping'), dict) or set(proof['mapping']) != {'from', 'to'} or (not all((isinstance(proof['mapping'][k], str) and proof['mapping'][k] for k in ['from', 'to']))):
        raise ValueError('Source region is missing color mapping')
    return out

def regions_from_explicit(objects, lab_orig, allowed_mask):
    if not objects or any(('source_region_rle' not in o for o in objects)):
        raise ValueError('Every object requires an explicit source region')
    if tuple(np.asarray(lab_orig).shape[:2]) != tuple(np.asarray(allowed_mask).shape):
        raise ValueError('Source image and allowed mask sizes differ')
    regions = []
    for (i, o) in enumerate(objects):
        if o.get('src_find_rule') != RULE or o.get('pair_id') != i:
            raise ValueError('Source region rule or object order mismatch')
        r = decode_region(o['source_region_rle'], allowed_mask)
        m = o['source_region_rle']['provenance']['mapping']
        if m != {'from': o['param_from'], 'to': o['param_to']}:
            raise ValueError('Source region color mapping mismatch')
        if int(r.sum()) != o.get('src_region_px'):
            raise ValueError('Source region area does not match metadata')
        regions.append(r)
    return (regions, RULE)
