import torch
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
    torch.save({"model": model.state_dict(), "meta": meta}, path)

def get_param_groups(model, base_lr=1e-4, head_lr=1e-3, weight_decay=1e-2):
    decay, no_decay, head_params = [], [], []

    head = model.get_classifier() if hasattr(model, "get_classifier") else None
    head_ids = {id(p) for p in head.parameters()} if head is not None else set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # tách head theo id()
        if id(param) in head_ids:
            head_params.append(param)
            continue

        # backbone: không decay cho bias và norm
        if name.endswith("bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay, "lr": base_lr, "weight_decay": weight_decay},
        {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
    ]
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    return param_groups

def train_model(model_name, model, train_loader, valid_loader, criterion, optimizer, scaler, scheduler,
          gpu_aug=None, MEAN=None, STD=None, epochs=5, patience=None):
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

        print(f"[{epoch}/{epochs}]: train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f} | \
              valid_acc={valid_acc*100:.2f}% | valid_f1={valid_f1:.4f}")
        
        if patience != None:
            if valid_f1 > best_f1:
                best_f1, best_epoch = valid_f1, epoch
                save_ckpt(model, best_path, {"model_name": model_name})
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
    fps_value = fps(model, 64, 224)

    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel()*p.element_size() for p in model.parameters())/(1024**2)

    return {
        "model_name": model_name,
        "size_mb": model_size_mb,
        "valid_acc": valid_acc,
        "valid_f1": valid_f1,
        "fps": fps_value,
        "num_params": num_params,
        "ckpt_path": best_path
    }