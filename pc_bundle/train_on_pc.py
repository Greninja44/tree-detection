"""Train the tree-trunk detector on a (Windows) GPU PC.

Usage (from this folder, after unzipping the datasets next to this script):
    python train_on_pc.py                 # both models
    python train_on_pc.py --only trunk    # or: --only path
Output: results/tree_trunk.pt  -> copy it to the Jetson's models/ folder.
(The path segmenter that used to be trained here was replaced by the pretrained RUGD + DeepScene models.)
"""
import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write_yaml(name: str, root: Path, nc_name: str) -> Path:
    y = HERE / f"{name}.yaml"
    y.write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: 1\nnames:\n  0: {nc_name}\n",
        encoding="utf-8",
    )
    return y


def train(model: str, data: Path, name: str, out_name: str, epochs: int, imgsz: int, batch: int, patience: int = 30, **extra) -> None:
    from ultralytics import YOLO

    m = YOLO(model)
    m.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, device=0, workers=4, patience=patience,
            project=str(HERE / "runs"), name=name, exist_ok=True, **extra)
    best = HERE / "runs" / name / "weights" / "best.pt"
    (HERE / "results").mkdir(exist_ok=True)
    shutil.copy(best, HERE / "results" / out_name)
    print(f"\n==> {HERE / 'results' / out_name}")
    r = YOLO(str(best)).val(data=str(data), split="test", imgsz=imgsz, device=0)
    print(f"TEST  mAP50={r.box.map50:.3f}  mAP50-95={r.box.map:.3f}  P={r.box.mp:.3f}  R={r.box.mr:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["trunk"], help="kept for compatibility; only the trunk model is trained")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="Lower (8/4) if you hit CUDA out-of-memory.")
    ap.add_argument("--trunk-model", default="yolov8s.pt", help="yolov8n/s/m/l.pt (bigger = more accurate, slower on the Jetson).")
    ap.add_argument("--tag", default="", help="Suffix for run/output names, e.g. _m960, so earlier results are not overwritten.")
    ap.add_argument("--patience", type=int, default=30)
    a = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available to PyTorch. Install the CUDA build: https://pytorch.org/get-started/locally/")
    print("GPU:", torch.cuda.get_device_name(0))

    if a.only in (None, "trunk"):
        train(a.trunk_model, write_yaml("trunk", HERE / "data", "tree_trunk"), f"tree_trunk{a.tag}", f"tree_trunk{a.tag}.pt",
              a.epochs, a.imgsz, a.batch, a.patience, hsv_v=0.6, hsv_s=0.7, cos_lr=True, close_mosaic=15)


if __name__ == "__main__":  # required on Windows (dataloader worker processes)
    main()
