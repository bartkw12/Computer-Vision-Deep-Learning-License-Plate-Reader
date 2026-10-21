# data/

Datasets, videos and annotations live here and are not tracked by git: they
are large, and they are not ours to redistribute.

```
data/
├── plates/          YOLO-format plate detection dataset (train/ valid/ test/ data.yaml)
├── openalpr/        OpenALPR end-to-end benchmark (endtoend/us, eu, br)
├── videos/          input videos
└── annotations/     hand-typed ground truth for videos (label sheets)
```

| Folder | Source | How to get it |
|---|---|---|
| `plates/` | [License Plate Recognition](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e), Roboflow Universe | `python scripts/download_data.py plates` |
| `openalpr/` | [openalpr/benchmarks](https://github.com/openalpr/benchmarks) | `python scripts/download_data.py openalpr` |
| `videos/` | your own footage | copy it in |
| `annotations/` | you | `alpr label-sheet`, then fill in the `plate` column |

Annotations of real footage contain real plate numbers. Keep them out of the
repository unless you are sure they may be published.
