import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data_utils import build_cifar10_dataloaders
from model_utils import build_model, load_checkpoint, resolve_device, save_checkpoint


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-26 on CIFAR-10.")
    parser.add_argument("--arch", default="resnet26", help="Model architecture")
    parser.add_argument("--data-dir", default="./data", help="Dataset root")
    parser.add_argument(
        "--ckpt-path",
        default="./ckpt/cifar10/resnet26_best.pth",
        help="Best-checkpoint output path",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Train batch size")
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Evaluation batch size",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Optional label smoothing for training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker count",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR-10 if it is not already present",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the best checkpoint if it exists",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of output classes",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show per-batch tqdm progress bars",
    )
    return parser.parse_args()


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_last_checkpoint_path(best_checkpoint_path):
    best_path = Path(best_checkpoint_path)
    stem = best_path.stem
    if stem.endswith("_best"):
        stem = stem[: -len("_best")]
    return best_path.with_name(f"{stem}_last{best_path.suffix}")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs, show_progress):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc=f"train {epoch}/{epochs}",
        leave=False,
        disable=not show_progress,
    )
    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += batch_size
        progress.set_postfix(
            loss=f"{running_loss / max(total, 1):.4f}",
            acc=f"{100.0 * correct / max(total, 1):.2f}",
        )

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += batch_size

    return running_loss / max(total, 1), correct / max(total, 1)


def maybe_resume(model, optimizer, scheduler, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.info("resume skipped, checkpoint not found: %s", checkpoint_path)
        return 0, 0.0

    checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    start_epoch = int(checkpoint.get("epoch") or 0)
    best_acc = float(checkpoint.get("best_acc") or 0.0)
    logger.info(
        "resumed from %s at epoch %d with best_acc %.4f",
        checkpoint_path,
        start_epoch,
        best_acc,
    )
    return start_epoch, best_acc


def main():
    args = parse_args()
    configure_logging()
    set_seed(args.seed)

    device = resolve_device(args.device)
    logger.info("using device: %s", device)
    logger.info("training architecture: %s", args.arch)
    logger.info("best checkpoint path: %s", args.ckpt_path)

    train_loader, test_loader = build_cifar10_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        download=args.download,
    )

    model = build_model(args.arch, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    start_epoch = 0
    best_acc = 0.0
    last_checkpoint_path = get_last_checkpoint_path(args.ckpt_path)
    if args.resume:
        resume_path = last_checkpoint_path if last_checkpoint_path.exists() else args.ckpt_path
        start_epoch, best_acc = maybe_resume(model, optimizer, scheduler, resume_path)

    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch + 1,
            args.epochs,
            args.progress,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        current_best = max(best_acc, test_acc)
        save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            best_acc=current_best,
            arch=args.arch,
            num_classes=args.num_classes,
        )

        if test_acc >= best_acc:
            best_acc = test_acc
            save_checkpoint(
                args.ckpt_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_acc=best_acc,
                arch=args.arch,
                num_classes=args.num_classes,
            )

        logger.info(
            "epoch %03d/%03d | lr %.5f | train_loss %.4f | train_acc %.2f%% | "
            "test_loss %.4f | test_acc %.2f%% | best_acc %.2f%%",
            epoch + 1,
            args.epochs,
            current_lr,
            train_loss,
            train_acc * 100.0,
            test_loss,
            test_acc * 100.0,
            best_acc * 100.0,
        )
        scheduler.step()


if __name__ == "__main__":
    main()
