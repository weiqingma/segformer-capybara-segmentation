import torch
from tqdm import tqdm

from loss import CE_Loss, Dice_loss


def compute_loss(outputs, masks, seg_labels, cls_weights, num_classes, use_dice = True):
    """
    统一计算 loss

    outputs: 模型输出，shape = [B, num_classes, H, W]
    masks: 普通 mask，shape = [B, H, W]，给 CE_Loss 用
    seg_labels: one-hot mask，shape = [B, H, W, num_classes + 1]，给 Dice_loss 用
    """
    ce_loss = CE_Loss(inputs = outputs, target = masks, cls_weights = cls_weights, num_classes = num_classes)
    if use_dice:
        dice_loss = Dice_loss(inputs = outputs, target = seg_labels)
        total_loss = ce_loss + dice_loss
    else:
        total_loss = ce_loss
    return total_loss


def train_one_epoch(model_train, train_loader, optimizer, device, epoch, total_epochs, cls_weights, num_classes, use_dice=True, local_rank=0):
    model_train.train()

    total_loss = 0.0

    if local_rank == 0:
        progress_bar = tqdm(train_loader, desc=f"Train Epoch [{epoch}/{total_epochs}]")
    else:
        progress_bar = train_loader

    for iteration, batch in enumerate(progress_bar):
        images, masks, seg_labels = batch

        images = images.to(device)
        masks = masks.to(device)
        seg_labels = seg_labels.to(device)

        outputs = model_train(images)
        loss = compute_loss(outputs=outputs, masks=masks, seg_labels=seg_labels, cls_weights=cls_weights,
                            num_classes=num_classes, use_dice=use_dice)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (iteration + 1)

        if local_rank == 0:
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

    epoch_avg_loss = total_loss / len(train_loader)
    return epoch_avg_loss


def validate_one_epoch(model_val, val_loader, device, epoch, total_epochs, cls_weights, num_classes, use_dice=True, local_rank=0):
    model_val.eval()

    total_loss = 0.0

    if local_rank == 0:
        progress_bar = tqdm(val_loader, desc=f"Val Epoch [{epoch}/{total_epochs}]")
    else:
        progress_bar = val_loader

    with torch.no_grad():
        for iteration, batch in enumerate(progress_bar):
            images, masks, seg_labels = batch

            images = images.to(device)
            masks = masks.to(device)
            seg_labels = seg_labels.to(device)

            outputs = model_val(images)
            loss = compute_loss(outputs=outputs, masks=masks, seg_labels=seg_labels,
                                cls_weights=cls_weights, num_classes=num_classes, use_dice=use_dice)

            total_loss += loss.item()
            avg_loss = total_loss / (iteration + 1)

            if local_rank == 0:
                progress_bar.set_postfix({"val_loss": f"{avg_loss:.4f}"})

    return total_loss / len(val_loader)


def fit_one_epoch(model_train, train_loader, val_loader, optimizer, device, epoch, total_epochs, cls_weights, num_classes, use_dice=True, local_rank=0):
    train_loss = train_one_epoch(
        model_train=model_train,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epoch=epoch,
        total_epochs=total_epochs,
        cls_weights=cls_weights,
        num_classes=num_classes,
        use_dice=use_dice,
        local_rank=local_rank
    )

    val_loss = validate_one_epoch(
        model_val=model_train,
        val_loader=val_loader,
        device=device,
        epoch=epoch,
        total_epochs=total_epochs,
        cls_weights=cls_weights,
        num_classes=num_classes,
        use_dice=use_dice,
        local_rank=local_rank
    )

    if local_rank == 0:
        print(
            f"Epoch [{epoch}/{total_epochs}] "
            f"train_loss: {train_loss:.4f} "
            f"val_loss: {val_loss:.4f}"
        )

    return train_loss, val_loss
