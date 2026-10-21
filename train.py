"""Train PCDNet from random initialization using the repository's RGB/AoLP/DoLP loader."""

import argparse
import math
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.yolo import Model
from utils.dataset import LoadImagesAndLabels, img_formats


DEFAULT_HYP = {
    "lr0": 0.01, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005,
    "warmup_epochs": 3.0, "warmup_momentum": 0.8, "warmup_bias_lr": 0.1,
    "box": 0.05, "obj": 1.0, "cls": 0.5,
    "anchor_t": 4.0, "fl_gamma": 0.0,
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0,
    "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
    "mosaic": 1.0, "copy_paste": 0.0, "paste_in": 0.0,
}


def resolve_split(value, data_dir):
    path = Path(value)
    return (path if path.is_absolute() else data_dir / path).resolve()


def read_data(path):
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "train" not in data or "val" not in data:
        raise ValueError(f"{path} must define train and val")
    names = data.get("names")
    nc = int(data.get("nc", len(names) if isinstance(names, (list, dict)) else 0))
    if nc < 1:
        raise ValueError("Dataset YAML must define a positive nc or names list")
    data_dir = Path(data.get("path", path.parent))
    if not data_dir.is_absolute():
        data_dir = (path.parent / data_dir).resolve()
    data["nc"] = nc
    data["names"] = names or [str(i) for i in range(nc)]
    for split in ("train", "val"):
        data[split] = resolve_split(data[split], data_dir)
    return data


def split_images(split):
    images = split / "images"
    if not images.is_dir():
        raise FileNotFoundError(f"Expected RGB image directory: {images}")
    files = sorted(p for p in images.rglob("*") if p.is_file() and p.suffix[1:].lower() in img_formats)
    if not files:
        raise FileNotFoundError(f"No supported images found in {images}")
    for image in files:
        relative = image.relative_to(images)
        for modality, suffix in (("aolp", "_aolp"), ("dolp", "_dolp")):
            stem = image.stem
            if stem.endswith(suffix):
                raise ValueError(f"Expected plain RGB filename under images/: {image.name}")
            paired = split / modality / relative.with_name(f"{stem}{suffix}{image.suffix}")
            if not paired.is_file():
                raise FileNotFoundError(f"Missing {modality} counterpart for {image}: {paired}")
        label = split / "labels" / relative.with_suffix(".txt")
        if not label.is_file():
            raise FileNotFoundError(f"Missing YOLO label for {image}: {label}")
    return files


def validate_labels(files, nc):
    count = 0
    for file in files:
        for line_no, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{file}:{line_no}: expected 'class x y w h'")
            values = np.asarray(fields, dtype=np.float32)
            cls = int(values[0])
            if values[0] != cls or cls < 0 or cls >= nc:
                raise ValueError(f"{file}:{line_no}: class id {values[0]} outside [0, {nc - 1}]")
            if not np.isfinite(values).all() or (values[1:] < 0).any() or (values[1:] > 1).any():
                raise ValueError(f"{file}:{line_no}: coordinates must be finite and normalized to [0, 1]")
            count += 1
    if count == 0:
        raise ValueError("No objects found in split labels")
    return count


class ComputeLoss:
    """YOLOv7-style anchor assignment and box/object/class losses for IDetect."""

    def __init__(self, model, hyp):
        self.device = next(model.parameters()).device
        self.hyp = hyp
        self.det = model.model[-1]
        self.nc, self.nl, self.na = self.det.nc, self.det.nl, self.det.na
        self.anchors = self.det.anchors
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=self.device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=self.device))
        self.balance = [4.0, 1.0, 0.4][:self.nl]
        self.cp, self.cn = 1.0, 0.0

    def __call__(self, predictions, targets):
        device = predictions[0].device
        lcls, lbox, lobj = (torch.zeros((), device=device) for _ in range(3))
        tcls, tbox, indices, anch = self.build_targets(predictions, targets)
        for i, pred in enumerate(predictions):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pred[..., 4], device=device)
            n = b.shape[0]
            if n:
                ps = pred[b, a, gj, gi]
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2.0) ** 2 * anch[i]
                iou = self.box_iou(torch.cat((pxy, pwh), 1), tbox[i])
                lbox += (1.0 - iou).mean()
                tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)
                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)
                    t[torch.arange(n, device=device), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)
            obj_loss = self.BCEobj(pred[..., 4], tobj)
            lobj += obj_loss * self.balance[i]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        return (lbox + lobj + lcls), torch.stack((lbox.detach(), lobj.detach(), lcls.detach()))

    @staticmethod
    def box_iou(box1, box2, eps=1e-7):
        b1_x1y1, b1_x2y2 = box1[:, :2] - box1[:, 2:] / 2, box1[:, :2] + box1[:, 2:] / 2
        b2_x1y1, b2_x2y2 = box2[:, :2] - box2[:, 2:] / 2, box2[:, :2] + box2[:, 2:] / 2
        inter = (torch.minimum(b1_x2y2, b2_x2y2) - torch.maximum(b1_x1y1, b2_x1y1)).clamp(0).prod(1)
        union = box1[:, 2:].prod(1) + box2[:, 2:].prod(1) - inter + eps
        return inter / union

    def build_targets(self, predictions, targets):
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)
        bias = 0.5
        offset_template = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float() * bias
        for i, pred in enumerate(predictions):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor([pred.shape[3], pred.shape[2], pred.shape[3], pred.shape[2]], device=targets.device)
            scaled = targets * gain
            if nt:
                ratio = scaled[:, :, 4:6] / anchors[:, None]
                matches = torch.max(ratio, 1.0 / ratio).max(2)[0] < self.hyp["anchor_t"]
                scaled = scaled[matches]
                gxy = scaled[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < bias) & (gxy > 1)).T
                l, m = ((gxi % 1 < bias) & (gxi > 1)).T
                mask = torch.stack((torch.ones_like(j), j, k, l, m))
                scaled = scaled.repeat((5, 1, 1))[mask]
                offsets = (torch.zeros_like(gxy)[None] + offset_template[:, None])[mask]
            else:
                scaled = targets[0]
                offsets = 0
            bc, gxy, gwh, a = scaled.split((2, 2, 2, 1), 1)
            a = a.long().view(-1)
            b, c = bc.long().T
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            indices.append((b, a, gj.clamp_(0, pred.shape[2] - 1), gi.clamp_(0, pred.shape[3] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)
        return tcls, tbox, indices, anch


def load_architecture(path, nc):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint.get("ema") or checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
    config = getattr(source, "yaml", None)
    if config is None:
        raise ValueError(f"{path} does not contain a model with an embedded YAML architecture")
    config = deepcopy(config)
    config["nc"] = nc
    return config


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(4, device=device)
    context = torch.enable_grad if training else torch.no_grad
    with context():
        bar = tqdm(loader, leave=False)
        for rgb, aolp, dolp, targets, _, _ in bar:
            inputs = [x.to(device, non_blocking=True).float().div_(255.0) for x in (rgb, aolp, dolp)]
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(inputs)
                if not training:
                    outputs = outputs[1]
                loss, items = criterion(outputs, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            totals[:3] += items
            totals[3] += loss.detach()
            bar.set_postfix(loss=f"{loss.item():.4f}")
    return (totals / max(len(loader), 1)).cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description="Train the original RGB-Polarization PCDNet from scratch")
    parser.add_argument("--data", type=Path, default=ROOT / "data/rgbpc_coco.yaml")
    parser.add_argument("--architecture", type=Path, default=ROOT.parent / "PCDNet.pt",
                        help="Checkpoint used only to read its embedded PCDNet YAML architecture")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA device such as 0")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/train")
    parser.add_argument("--name", default="pcdnet_scratch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.imgsz < 32 or args.imgsz % 32:
        parser.error("epochs/batch-size must be positive and imgsz must be a positive multiple of 32")
    data = read_data(args.data.resolve())
    train_files, val_files = split_images(data["train"]), split_images(data["val"])
    print(f"Dataset: {args.data.resolve()} | nc={data['nc']} | train={len(train_files)} val={len(val_files)}")
    print(f"Objects: train={validate_labels(train_files, data['nc'])} val={validate_labels(val_files, data['nc'])}")
    if not args.architecture.is_file():
        raise FileNotFoundError(f"PCDNet architecture checkpoint not found: {args.architecture.resolve()}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + args.device if args.device.isdigit() else args.device) if args.device != "auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyp = DEFAULT_HYP.copy()
    cfg = load_architecture(args.architecture.resolve(), data["nc"])
    model = Model(cfg, input_channel=3, nc=data["nc"]).to(device)
    model.names = data["names"]
    print(f"Device: {device}; architecture: embedded PCDNet YAML; pretrained weights: NOT loaded")

    stride = max(int(model.stride.max()), 32)
    train_ds = LoadImagesAndLabels(data["train"], args.imgsz, args.batch_size, augment=True,
                                   hyp=hyp, rect=False, stride=stride, prefix="train: ")
    val_ds = LoadImagesAndLabels(data["val"], args.imgsz, args.batch_size, augment=False,
                                 hyp=hyp, rect=True, stride=stride, pad=0.5, prefix="val: ")
    train_loader = DataLoader(train_ds, batch_size=min(args.batch_size, len(train_ds)), shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda",
                              collate_fn=LoadImagesAndLabels.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=min(args.batch_size, len(val_ds)), shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda",
                            collate_fn=LoadImagesAndLabels.collate_fn)

    pg0, pg1, pg2 = [], [], []
    norm_parameters = {id(module.weight) for module in model.modules()
                       if isinstance(module, nn.BatchNorm2d) and module.weight is not None}
    for name, parameter in model.named_parameters():
        if name.endswith(".bias"):
            pg2.append(parameter)
        elif id(parameter) in norm_parameters:
            pg0.append(parameter)
        else:
            pg1.append(parameter)
    optimizer = torch.optim.SGD(pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True)
    optimizer.add_param_group({"params": pg1, "weight_decay": hyp["weight_decay"]})
    optimizer.add_param_group({"params": pg2})
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (1 - epoch / args.epochs) * (1 - hyp["lrf"]) + hyp["lrf"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not args.no_amp)
    use_amp = device.type == "cuda" and not args.no_amp
    criterion = ComputeLoss(model, hyp)

    output_dir = (args.project / args.name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best = math.inf
    for epoch in range(args.epochs):
        warmup = epoch < hyp["warmup_epochs"]
        if warmup:
            warm = min((epoch + 1) / hyp["warmup_epochs"], 1.0)
            for group in optimizer.param_groups:
                group["lr"] = hyp["warmup_bias_lr"] * (1 - warm) + group["initial_lr"] * warm if "initial_lr" in group else hyp["lr0"] * warm
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp)
        val_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"epoch {epoch + 1}/{args.epochs} train(box/obj/cls/total)={train_loss} val={val_loss}")
        state = {"epoch": epoch + 1, "model": deepcopy(model).float().cpu(), "optimizer": optimizer.state_dict(),
                 "train_loss": train_loss, "val_loss": val_loss, "names": data["names"],
                 "architecture_source": str(args.architecture.resolve()), "pretrained_weights_loaded": False}
        torch.save(state, output_dir / "last.pt")
        if val_loss[3] < best:
            best = val_loss[3]
            torch.save(state, output_dir / "best.pt")
        model.to(device)


if __name__ == "__main__":
    main()
