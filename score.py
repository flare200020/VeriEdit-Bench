import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import cv2
import numpy as np
import scoring
from scoring.preprocess import load_triple

ROOT = Path(__file__).resolve().parent
AXES = ('edit_fidelity', 'preservation', 'localization', 'magnitude')


def evaluate(case, prediction, root=ROOT, with_color=False):
    cv2.setNumThreads(1)
    c = dict(case)
    c['_dir'] = Path(root) / c['directory']
    prediction = Path(prediction).resolve()
    selected = None
    if c.get('reference_set'):
        d = (Path(root) / c['reference_set']['manifest']).parent
        with np.load(d / 'selection_features.npz', allow_pickle=False) as pack:
            _, _, image, _, _ = load_triple(
                c['_dir'] / c['source_image'], c['_dir'] / c['gt_image'],
                prediction, c['_dir'] / c['gt_mask'])
            pixels = cv2.GaussianBlur(image, (0, 0), 1.).reshape(-1, 3)[pack['index']].astype('float32')
            errors = [float(np.mean((feature.astype('float32') - pixels) ** 2)) for feature in pack['features']]
            selected = int(np.argmin(errors))
        c['gt_image'] = str(d / f'phase_{selected:02}.png')
        c['gt_mask'] = c['allowed_mask'] = str(d / f'mask_{selected:02}.png')
    extras = scoring.load_extras(c, prediction)
    row = {'case_id': c['case_id'], 'modality': c['modality'], 'edit_type': c['edit_type']}
    for name, fn in scoring.get_core_axes().items():
        value, _ = fn(prediction, c['_dir'] / c['gt_image'], c['_dir'] / c['source_image'], c['edit_meta'], extras)
        row[name] = round(float(value), 4)
    values = [row[name] for name in AXES]
    if not np.isfinite(values).all():
        raise ValueError('Non-finite score: ' + c['case_id'])
    row['H'] = 0.0 if min(values) <= 1e-9 else round(4 / sum(1 / v for v in values), 4)
    row['selected_dash_phase'] = selected
    if with_color:
        from scoring.targets import target_attainment
        if not extras.get('target_meta'):
            raise ValueError('Missing color metadata: ' + c['case_id'])
        value, detail = target_attainment(prediction, c['_dir'] / c['gt_image'], c['_dir'] / c['source_image'], c['edit_meta'], extras, c)
        if detail.get('error'):
            raise ValueError(detail)
        row['color_attainment'] = round(float(value), 4) if value is not None else None
    return row


def _worker(args):
    return evaluate(*args)


def summarize(rows, full_count):
    cells = defaultdict(list)
    for row in rows:
        cells[(row['modality'], row['edit_type'].split('@')[0])].append(row['H'])
    return {
        'cases_scored': len(rows), 'benchmark_cases': full_count,
        'scope': 'full_benchmark' if len(rows) == full_count else 'selected_cases',
        'asset_operation_cells': len(cells),
        'cell_mean_H': float(np.mean([np.mean(v) for v in cells.values()])),
        'case_mean_H': float(np.mean([r['H'] for r in rows])),
        'case_mean_axes': {name: float(np.mean([r[name] for r in rows])) for name in AXES},
    }


def main():
    parser = argparse.ArgumentParser(description='Score edited images on VeriEdit-Bench.')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'cases.json')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--predictions', type=Path, help='Directory containing <case_id>.png')
    group.add_argument('--reference', choices=['input', 'target'])
    parser.add_argument('--case', action='append', help='Optional case ID; repeat to select multiple cases')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--with-color', action='store_true', help='Include separate color attainment on eligible cases')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    root = args.manifest.resolve().parent
    manifest = json.loads(args.manifest.read_text())
    cases = manifest['cases']
    all_ids = {c['case_id'] for c in cases}
    if len(all_ids) != len(cases):
        raise ValueError('Duplicate case IDs in manifest')
    if args.case:
        selected = set(args.case)
        if selected - all_ids:
            parser.error('Unknown case IDs: ' + ', '.join(sorted(selected - all_ids)))
        cases = [c for c in cases if c['case_id'] in selected]
    work = []
    for c in cases:
        prediction = root / c['directory'] / c['gt_image' if args.reference == 'target' else 'source_image'] if args.reference else args.predictions / (c['case_id'] + '.png')
        if not prediction.is_file():
            raise FileNotFoundError(prediction)
        work.append((c, prediction, root, args.with_color))
    if args.jobs == 1:
        rows = list(map(_worker, work))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(_worker, work))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, manifest['case_count'])
    summary_path = args.output.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Scored {len(rows)} cases; cell-mean H = {summary["cell_mean_H"]:.4f}')
    print(f'Wrote {args.output} and {summary_path}')


if __name__ == '__main__':
    main()
