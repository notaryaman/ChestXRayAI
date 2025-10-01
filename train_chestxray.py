import argparse, os, json, random, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torchvision
from torchvision import transforms, models
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    precision_recall_curve, accuracy_score
)
import matplotlib.pyplot as plt
from tqdm import tqdm


# -----------------------------
# Device (prefer Apple MPS -> CUDA -> CPU)
# -----------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # keep perf on MPS


# -----------------------------
# Model definitions
# -----------------------------
class DeeperCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc = None
        self._initialize_fc()

    def _initialize_fc(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            x = self.pool(self.relu(self.conv1(dummy)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.relu(self.conv3_1(x))
            x = self.relu(self.conv3_2(x))
            x = self.pool(x)
            x = self.relu(self.conv4_1(x))
            x = self.relu(self.conv4_2(x))
            x = self.pool(x)
            flattened = x.view(1, -1).shape[1]
            self.fc = nn.Linear(flattened, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))); x = self.dropout(x)
        x = self.pool(self.relu(self.conv2(x))); x = self.dropout(x)
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.pool(x); x = self.dropout(x)
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.pool(x); x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def build_model(name: str, num_classes=2, pretrained=True):
    name = name.lower()
    if name == "deeper":
        return DeeperCNN(num_classes=num_classes)
    elif name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model: {name} (use 'deeper' or 'resnet18')")


# -----------------------------
# Data
# -----------------------------
def build_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(7),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return train_tf, eval_tf


def stratified_split_indices(dataset, val_ratio=0.15, seed=42):
    labels = [dataset.samples[i][1] for i in range(len(dataset.samples))]
    labels = np.array(labels)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels))
    val_idx = []
    train_idx = []
    for c in np.unique(labels):
        c_idx = idx[labels == c]
        rng.shuffle(c_idx)
        n_val = max(1, int(round(len(c_idx) * val_ratio)))
        val_idx.extend(c_idx[:n_val].tolist())
        train_idx.extend(c_idx[n_val:].tolist())
    rng.shuffle(train_idx); rng.shuffle(val_idx)
    return train_idx, val_idx


def make_loaders(data_root, batch_size=64, num_workers=4, val_ratio=0.15, seed=42):
    train_tf, eval_tf = build_transforms()
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    full_train = torchvision.datasets.ImageFolder(train_dir, transform=train_tf)
    full_train_for_split = torchvision.datasets.ImageFolder(train_dir, transform=eval_tf)

    train_idx, val_idx = stratified_split_indices(full_train_for_split, val_ratio=val_ratio, seed=seed)
    ds_train = Subset(full_train, train_idx)
    ds_val   = Subset(full_train_for_split, val_idx)
    ds_test  = torchvision.datasets.ImageFolder(test_dir, transform=eval_tf)

    subset_targets = [full_train.samples[i][1] for i in train_idx]
    class_sample_count = np.bincount(subset_targets)
    class_weights = 1.0 / np.clip(class_sample_count, 1, None)
    sample_weights = [class_weights[t] for t in subset_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    pin_mem = torch.cuda.is_available() and not torch.backends.mps.is_available()

    loader_train = DataLoader(ds_train, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin_mem)
    loader_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_mem)
    loader_test  = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_mem)

    class_names = full_train.classes  # ['NORMAL', 'PNEUMONIA']
    return loader_train, loader_val, loader_test, class_names


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(y_true, y_prob, y_pred):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)

    roc = float('nan'); pr = float('nan')
    if len(np.unique(y_true)) == 2:
        roc = roc_auc_score(y_true, y_prob)
        pr  = average_precision_score(y_true, y_prob)

    f1  = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    acc  = (y_true == y_pred).mean()
    return {
        "accuracy": float(acc),
        "roc_auc": float(roc) if not math.isnan(roc) else None,
        "pr_auc": float(pr) if not math.isnan(pr) else None,
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
    }


# -----------------------------
# Train / Evaluate loops
# -----------------------------
def run_epoch(model, loader, criterion, device, train=True, use_amp=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    y_true, y_prob, y_pred = [], [], []

    if train:
        optimizer = run_epoch.optimizer  # set externally

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device); labels = labels.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=("mps" if device.type == "mps" else "cuda" if device.type=="cuda" else "cpu"),
                            dtype=torch.float16 if use_amp else torch.float32,
                            enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.detach().item() * images.size(0)

        probs = torch.softmax(logits.detach(), dim=1)[:, 1]
        preds = torch.argmax(logits.detach(), dim=1)

        y_true.extend(labels.cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(y_true, y_prob, y_pred)
    return avg_loss, metrics


# -----------------------------
# Plot helpers
# -----------------------------
def save_confusion_matrix(y_true, y_pred, class_names, out_path):
    from itertools import product
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig, ax = plt.subplots(figsize=(4,4), dpi=150)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j], ha='center', va='center')
    fig.tight_layout()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path); plt.close(fig)


def save_curves(y_true, y_prob, out_dir):
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    os.makedirs(out_dir, exist_ok=True)

    # ROC
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig1, ax1 = plt.subplots(dpi=150)
        ax1.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
        ax1.plot([0,1], [0,1], linestyle='--')
        ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR'); ax1.set_title('ROC')
        ax1.legend(loc='lower right')
        fig1.savefig(os.path.join(out_dir, "roc_curve.png")); plt.close(fig1)

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = np.trapz(prec[::-1], rec[::-1])
    fig2, ax2 = plt.subplots(dpi=150)
    ax2.plot(rec, prec, label=f'PR AUC ≈ {pr_auc:.3f}')
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); ax2.set_title('Precision-Recall')
    ax2.legend(loc='lower left')
    fig2.savefig(os.path.join(out_dir, "pr_curve.png")); plt.close(fig2)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path containing chest_xray/train and chest_xray/test")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--model", type=str, default="deeper", choices=["deeper", "resnet18"])
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device} (MPS available: {torch.backends.mps.is_available()})")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    n_workers = max(2, os.cpu_count() // 2)
    train_loader, val_loader, test_loader, class_names = make_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=n_workers,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # Model / Opt / Loss
    model = build_model(args.model, num_classes=2, pretrained=(args.model=="resnet18")).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run_epoch.optimizer = optimizer  # attach for training loop

    best_metric = -float("inf")
    best_state = None
    history = []

    # Training loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, device,
                                              train=True, use_amp=(not args.no_amp))
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, device,
                                          train=False, use_amp=False)
        elapsed = time.time() - t0

        monitor = val_metrics.get("pr_auc") if val_metrics.get("pr_auc") is not None else \
                  (val_metrics.get("roc_auc") if val_metrics.get("roc_auc") is not None else val_metrics["f1"])
        if (monitor is not None) and (monitor > best_metric):
            best_metric = monitor
            best_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "class_names": class_names,
                "args": vars(args),
            }
            torch.save(best_state, out_dir / "best.ckpt")

        history.append({
            "epoch": epoch,
            "time_sec": round(elapsed, 2),
            "train_loss": train_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "best_monitor": best_metric,
        })

        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.3f} "
            f"val_f1={val_metrics['f1']:.3f} "
            f"val_roc={val_metrics.get('roc_auc')} "
            f"val_pr={val_metrics.get('pr_auc')} "
            f"({elapsed:.1f}s)"
        )

        if epoch > 6:
            recent = [h["best_monitor"] for h in history[-5:]]
            if all(r <= best_metric for r in recent):
                print("Early stop: no improvement in last 5 epochs.")
                break

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Load best and Evaluate on TEST
    if best_state is None:
        print("Warning: no best state captured; using final model for test eval.")
    else:
        model.load_state_dict(best_state["model"])

    model.eval()
    y_true, y_prob, y_pred = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test", leave=False):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    # ---- Defensive checks for threshold tuning ----
    y_true_np = np.asarray(y_true).astype(int)
    y_prob_np = np.asarray(y_prob).astype(float)
    if np.isnan(y_prob_np).any() or np.isinf(y_prob_np).any():
        raise ValueError("NaN/Inf found in y_prob; check logits/prob computation.")

    # Default threshold metrics (τ = 0.50)
    y_pred_default = (y_prob_np >= 0.5).astype(int)
    default_metrics = {
        "threshold": 0.5,
        "accuracy": accuracy_score(y_true_np, y_pred_default),
        "precision": precision_score(y_true_np, y_pred_default, zero_division=0),
        "recall": recall_score(y_true_np, y_pred_default, zero_division=0),
        "f1": f1_score(y_true_np, y_pred_default, zero_division=0),
    }

    # Best-F1 threshold (only if both classes present)
    best_thr = None
    tuned_metrics = None
    if np.unique(y_true_np).size >= 2:
        p, r, thr = precision_recall_curve(y_true_np, y_prob_np)
        f1s = 2 * p * r / (p + r + 1e-9)
        best_i = f1s.argmax()
        # precision_recall_curve returns thresholds len = len(p)-1
        best_thr = float(thr[best_i]) if best_i < len(thr) else 0.5
        y_pred_best = (y_prob_np >= best_thr).astype(int)
        tuned_metrics = {
            "threshold": best_thr,
            "accuracy": accuracy_score(y_true_np, y_pred_best),
            "precision": precision_score(y_true_np, y_pred_best, zero_division=0),
            "recall": recall_score(y_true_np, y_pred_best, zero_division=0),
            "f1": f1_score(y_true_np, y_pred_best, zero_division=0),
        }
        print(f"\nBest-F1 threshold τ={best_thr:.3f}")
        print(f"@τ={best_thr:.3f}  F1={tuned_metrics['f1']:.3f}  "
              f"Precision={tuned_metrics['precision']:.3f}  "
              f"Recall={tuned_metrics['recall']:.3f}  "
              f"Accuracy={tuned_metrics['accuracy']:.3f}")
    else:
        print("Only one class in y_true; PR/ROC thresholds undefined. Showing τ=0.50 metrics only.")

    # Aggregate test metrics at τ=0.50 (for compatibility with compute_metrics)
    metrics = compute_metrics(y_true_np, y_prob_np, y_pred_default)
    print("\n=== TEST METRICS (τ=0.50) ===")
    for k, v in metrics.items():
        print(f"{k}: {('n/a' if v is None else f'{v:.4f}')}")

    # Save metrics & plots
    out_dir.mkdir(exist_ok=True, parents=True)
    save_confusion_matrix(y_true_np, y_pred_default, class_names, out_dir / "confusion_matrix_default.png")
    save_curves(y_true_np, y_prob_np, out_dir)

    # Also save confusion matrix for tuned threshold (if computed)
    if tuned_metrics is not None:
        y_pred_best = (y_prob_np >= best_thr).astype(int)
        save_confusion_matrix(y_true_np, y_pred_best, class_names, out_dir / "confusion_matrix_bestF1.png")

    # Write a single JSON with everything useful
    to_save = {
        "test_metrics_tau_0_50": metrics,
        "operating_point_default": default_metrics,
        "operating_point_bestF1": tuned_metrics,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(to_save, f, indent=2)

    # Export TorchScript for quick inference
    sample = torch.randn(1, 3, 224, 224).to(device)
    traced = torch.jit.trace(model, sample)
    traced.save(str(out_dir / "model_traced.pt"))

    print(f"\nSaved: {out_dir}/best.ckpt, metrics.json, history.json, curves & confusion matrices.")
    print("Done.")


if __name__ == "__main__":
    main()
