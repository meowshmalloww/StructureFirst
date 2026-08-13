# Local detector model

`yolo26n.onnx` is the official static 640 px YOLO26 Nano end-to-end ONNX
release asset:

- Source: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.onnx
- SHA-256: `2e947b787d9e787b93a16772a5f55b1d4d8c4d86f53146149c5d6a642442d6f7`
- Embedded exporter version: Ultralytics 8.4.38
- Output: `(1, 300, 6)` rows of `x1, y1, x2, y2, confidence, class_id`
- Model license: AGPL-3.0, or Ultralytics Enterprise License
- License details: https://www.ultralytics.com/license

The detector runs locally and displays only unverified 2D view-space
observations. It does not by itself identify collapse risk or create persistent
3D hazard geometry. Using the ONNX model without the Ultralytics Python runtime
does not remove the model's stated license obligations.
