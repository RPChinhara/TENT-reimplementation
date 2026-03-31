import logging

import torch
import torch.optim as optim

import norm
import tent
from conf import cfg, load_cfg_fom_args
from data_utils import build_cifar10_dataloaders, build_cifar10c_loader
from model_utils import build_model, load_checkpoint, resolve_device


logger = logging.getLogger(__name__)


def load_base_model(device):
    model = build_model(cfg.MODEL.ARCH, num_classes=cfg.MODEL.NUM_CLASSES)
    checkpoint = load_checkpoint(model, cfg.MODEL.CKPT_PATH, map_location="cpu")
    model.to(device)

    logger.info("loaded checkpoint: %s", cfg.MODEL.CKPT_PATH)
    if checkpoint:
        logger.info(
            "checkpoint metadata: epoch=%s, best_acc=%s",
            checkpoint.get("epoch"),
            checkpoint.get("best_acc"),
        )
    return model


def evaluate_loader(model, loader, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    return correct / max(total, 1)


def maybe_reset(model):
    try:
        model.reset()
        logger.info("resetting model")
    except Exception:
        logger.info("model does not require reset")


def evaluate(description):
    load_cfg_fom_args(description)
    device = resolve_device(cfg.DEVICE)
    logger.info("using device: %s", device)

    base_model = load_base_model(device)
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    elif cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model)
    elif cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    else:
        raise NotImplementedError(f"Unknown adaptation mode: {cfg.MODEL.ADAPTATION}")

    if cfg.TEST.EVAL_CLEAN:
        _, clean_loader = build_cifar10_dataloaders(
            data_dir=cfg.DATA_DIR,
            batch_size=cfg.TEST.BATCH_SIZE,
            eval_batch_size=cfg.TEST.BATCH_SIZE,
            num_workers=cfg.TEST.NUM_WORKERS,
            download=False,
        )
        maybe_reset(model)
        clean_acc = evaluate_loader(model, clean_loader, device)
        logger.info("clean error %% [cifar10]: %.2f%%", (1.0 - clean_acc) * 100.0)

    for severity in cfg.CORRUPTION.SEVERITY:
        for corruption_type in cfg.CORRUPTION.TYPE:
            maybe_reset(model)
            loader = build_cifar10c_loader(
                corruption=corruption_type,
                severity=severity,
                data_dir=cfg.DATA_DIR,
                batch_size=cfg.TEST.BATCH_SIZE,
                num_examples=cfg.CORRUPTION.NUM_EX,
                num_workers=cfg.TEST.NUM_WORKERS,
            )
            acc = evaluate_loader(model, loader, device)
            err = 1.0 - acc
            logger.info("error %% [%s%d]: %.2f%%", corruption_type, severity, err * 100.0)


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info("model for evaluation: %s", model.__class__.__name__)
    return model


def setup_norm(model):
    """Set up test-time normalization adaptation."""
    norm_model = norm.Norm(model, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
    _, stat_names = norm.collect_stats(norm_model.model)
    logger.info("model for adaptation: %s", model.__class__.__name__)
    logger.info("stats for adaptation: %s", stat_names)
    return norm_model


def setup_tent(model):
    """Set up tent adaptation."""
    model = tent.configure_model(model)
    tent.check_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(
        model,
        optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
    )
    logger.info("model for adaptation: %s", model.__class__.__name__)
    logger.info("params for adaptation: %s", param_names)
    logger.info("optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_optimizer(params):
    """Set up optimizer for tent adaptation."""
    if cfg.OPTIM.METHOD == "Adam":
        return optim.Adam(
            params,
            lr=cfg.OPTIM.LR,
            betas=(cfg.OPTIM.BETA, 0.999),
            weight_decay=cfg.OPTIM.WD,
        )
    if cfg.OPTIM.METHOD == "SGD":
        return optim.SGD(
            params,
            lr=cfg.OPTIM.LR,
            momentum=cfg.OPTIM.MOMENTUM,
            dampening=cfg.OPTIM.DAMPENING,
            weight_decay=cfg.OPTIM.WD,
            nesterov=cfg.OPTIM.NESTEROV,
        )
    raise NotImplementedError(f"Unsupported optimizer: {cfg.OPTIM.METHOD}")


if __name__ == "__main__":
    evaluate("CIFAR-10-C evaluation.")
