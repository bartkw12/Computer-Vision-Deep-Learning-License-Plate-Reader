# weights/

Model weights live here and are not tracked by git.

| File | What it is | How to get it |
|---|---|---|
| `yolo11n.pt` | COCO-pretrained YOLO11n, used to detect vehicles | Downloaded automatically by Ultralytics on first use |
| `plate_yolo11n.pt` | YOLO11n fine-tuned to detect license plates; the pipeline's default | Train it with `alpr train`, or download it from the project's releases page |
| `plate_yolo11n_original.pt` | The plate detector trained for the first version of this project, kept to compare original and reproduced training | The project's releases page |

Until a reproduced `plate_yolo11n.pt` exists, either copy the original to that
name or pass it explicitly, e.g. `alpr run VIDEO --set plate.weights=weights/plate_yolo11n_original.pt`.
