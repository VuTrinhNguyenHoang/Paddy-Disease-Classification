import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from ..utils.metrics import evaluate, fps

def train_one_epoch(model, loader, criterion, optimizer, scaler, gpu_aug=None, MEAN=None, STD=None):
    device = next(model.parameters()).device
    model.train()

    total, correct, running = 0, 0, 0.0
    pbar = tqdm(loader, leave=False)

    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if gpu_aug != None:
            x = gpu_aug(x)
        if (MEAN != None) and (STD != None):
            x = (x - MEAN) / STD

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        pbar.set_postfix(loss=running/total, acc=correct/total)

    return running/total, correct/total

def save_ckpt(model, path, meta):
    if os.path.exists(path):
        os.remove(path)
    torch.save({"model": model.state_dict(), "meta": meta}, path)

def get_param_groups(model, base_lr=1e-4, head_lr=1e-3, weight_decay=1e-2,
                     new_keywords=("fc", "classifier", "bot", "mhsa", "mhla", "attention", "eca", "ca", "cablock")):
    bb_decay, bb_no_decay = [], []
    new_decay, new_no_decay = [], []

    # Identify explicit classifier if provided
    head_module = model.get_classifier() if hasattr(model, "get_classifier") else None
    head_ids = {id(p) for p in head_module.parameters()} if head_module is not None else set()

    norm_layer_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                        nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)

    def is_new_param(name, param):
        if id(param) in head_ids:
            return True
        lower = name.lower()
        return any(kw in lower for kw in new_keywords)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        target_is_new = is_new_param(name, param)
        # Decide decay vs no_decay
        no_decay_flag = False
        if name.endswith('bias'):
            no_decay_flag = True
        else:
            # Try to fetch module to check its type (optional best-effort)
            # We won't traverse modules each time for speed; heuristic on name first.
            if any(nd in name.lower() for nd in ["norm", "bn", "gn", "ln"]):
                no_decay_flag = True

        if target_is_new:
            if no_decay_flag:
                new_no_decay.append(param)
            else:
                new_decay.append(param)
        else:
            if no_decay_flag:
                bb_no_decay.append(param)
            else:
                bb_decay.append(param)

    param_groups = []
    if bb_decay:
        param_groups.append({"params": bb_decay, "lr": base_lr, "weight_decay": weight_decay})
    if bb_no_decay:
        param_groups.append({"params": bb_no_decay, "lr": base_lr, "weight_decay": 0.0})
    if new_decay:
        param_groups.append({"params": new_decay, "lr": head_lr, "weight_decay": weight_decay})
    if new_no_decay:
        param_groups.append({"params": new_no_decay, "lr": head_lr, "weight_decay": 0.0})

    return param_groups

def train_model(model_name, model, train_loader, valid_loader, criterion, optimizer, scaler, scheduler,
                gpu_aug=None, MEAN=None, STD=None, epochs=5, patience=None, fps_image_size=256):
    best_f1, best_epoch = -1.0, -1
    best_path = f"{model_name}_best.pt"

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": []
    }

    no_improve_epochs = 0

    pbar = tqdm(range(1, epochs+1), desc=model_name, unit="epoch")
    for epoch in pbar:
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, gpu_aug, MEAN, STD)
        valid_loss, valid_acc, valid_f1 = evaluate(model, valid_loader, criterion, MEAN, STD)

        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)

        scheduler.step()

        pbar.set_postfix(train_loss=f"{train_loss:.4f}", valid_loss=f"{valid_loss:.4f}",
                             valid_acc=f"{valid_acc*100:.2f}%", valid_f1=f"{valid_f1:.4f}")

        print(f"[{epoch}/{epochs}]: train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f} | valid_acc={valid_acc*100:.2f}% | valid_f1={valid_f1:.4f}")
        
        if patience != None:
            if valid_f1 > best_f1:
                best_f1, best_epoch = valid_f1, epoch
                save_ckpt(model, best_path, {"model_name": model_name, "epoch": epoch})
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs == patience:
                    print(f"EarlyStopping at epoch {epoch} (best f1={best_f1:.4f} at epoch {best_epoch})")
                    break

    device = next(model.parameters()).device
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    _, valid_acc, valid_f1 = evaluate(model, valid_loader, criterion, MEAN, STD)
    fps_value = fps(model, 32, fps_image_size)

    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel()*p.element_size() for p in model.parameters())/(1024**2)

    return history, {
        "model_name": model_name,
        "size_mb": model_size_mb,
        "valid_acc": valid_acc,
        "valid_f1": valid_f1,
        "fps": fps_value,
        "num_params": num_params,
        "ckpt_path": best_path
    }