# VeriEdit-Bench

1,740 image-editing tasks from 153 SVG, chart, web, and slide assets, covering 22 asset–operation combinations.

## Contents

```text
cases.json       Case IDs, asset IDs, instructions, and relative file paths
cases/           Input images, targets, edit masks, and color metadata
references/      Shared starting-phase references for 75 dashed-line tasks
score.py         CPU evaluation and aggregation
scoring/         Fixed scoring functions
LICENSE          Code license (MIT)
LICENSE-DATA     Dataset license and third-party notice
```

Editors receive only `original.png` and `instruction`. Targets, masks, and metadata are evaluation references; do not provide them to the editor. Case IDs use neutral names such as `svg_0001`. `asset_id` groups tasks sharing a source asset; `edit_type` identifies the editing operation.

## Evaluate

Use Python 3.10 or newer. Install dependencies and place outputs at `predictions/<case_id>.png`:

```sh
python -m pip install -r requirements.txt
python score.py --predictions predictions --output outputs/scores.csv --jobs 4
```

The CSV contains Fidelity, Preservation, Localization, Magnitude, and their harmonic mean H, all in [0,100]. H is zero when any axis is zero. The summary JSON reports the main score: average H within each asset–operation cell, then weight the 22 cells equally. It also reports case-mean H and axes. Missing predictions raise an error.

Add `--with-color` for the separate color-attainment diagnostic on 621 eligible tasks; it does not enter H. Dashed-line tasks select one of 64 shared starting-phase references by smoothed image error, using that same reference for all four axes.

For a selected task or reference controls:

```sh
python score.py --reference target --case svg_0001 --output outputs/check.csv
python score.py --reference input --output outputs/input_scores.csv --jobs 4
```

No GPU or external API is required. The package contains the dataset and evaluator; model predictions and paper-analysis scripts are not included.

## License

The evaluation code is released under the [MIT License](LICENSE). Our annotations, instructions, masks, and reference images are released under [CC BY-NC 4.0](LICENSE-DATA) for non-commercial research use. Some web and slide renderings include third-party material that we do not own and do not license; see [LICENSE-DATA](LICENSE-DATA) and [SOURCES.md](SOURCES.md). Rights holders may request removal by opening an issue.
