import os
import random
from functools import partial

import yaml
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.model import SegFormer, load_pretrained_weights
from src.dataloader import SegmentationDataset, seg_dataset_collate
from src.trainer import fit_one_epoch
from src.lr_scheduler import get_lr_scheduler, set_optimizer_lr


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    tr_cfg = cfg["train"]
    ds_cfg = cfg["dataset"]
    md_cfg = cfg["model"]

    distributed = tr_cfg.get("distributed", False)
    sync_bn = tr_cfg.get("sync_bn", False)
    ngpus_per_node = torch.cuda.device_count() #当前机器可见GPU数量

    seed_everything(tr_cfg["seed"])

    # ---- DDP 初始化 ----
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"]) #当前机器上第几张GPU
        rank = int(os.environ["RANK"])             #全局第几个训练进程
        device = torch.device("cuda", local_rank)
        if local_rank == 0: #只有主进程打印日志
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device("cuda" if tr_cfg["cuda"] and torch.cuda.is_available() else "cpu")
        local_rank = 0
        rank = 0

    # ---- 数据集 ----
    voc_root = ds_cfg["voc_root"]
    num_classes = md_cfg["num_classes"]
    input_shape = tuple(ds_cfg["input_shape"])

    train_txt = os.path.join(voc_root, ds_cfg["train_txt"])
    val_txt = os.path.join(voc_root, ds_cfg["val_txt"])

    #读取训练集图片ID
    with open(train_txt, "r") as f:
        train_lines = [line.strip() for line in f.readlines() if line.strip()]
    #读取验证集图片ID
    with open(val_txt, "r") as f:
        val_lines = [line.strip() for line in f.readlines() if line.strip()]

    #创建训练数据集
    train_dataset = SegmentationDataset(
        annotation_lines=train_lines, input_shape=input_shape,
        num_classes=num_classes, dataset_path=voc_root,
        image_dir=ds_cfg["image_dir"], mask_dir=ds_cfg["mask_dir"],
        mask_255_to_1=ds_cfg.get("mask_255_to_1", False), train=True)
    #创建验证数据集
    val_dataset = SegmentationDataset(
        annotation_lines=val_lines, input_shape=input_shape,
        num_classes=num_classes, dataset_path=voc_root,
        image_dir=ds_cfg["image_dir"], mask_dir=ds_cfg["mask_dir"],
        mask_255_to_1=ds_cfg.get("mask_255_to_1", False), train=False)

    # ---- DataLoader ----
    batch_size = tr_cfg["batch_size"]
    num_workers = tr_cfg["num_workers"]

    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=tr_cfg["seed"])
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        batch_size = batch_size // ngpus_per_node
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        #这里shuffle = True 后面同时用于训练和验证，验证集其实不需要shuffle
        shuffle = True

    train_loader = DataLoader(
        train_dataset, shuffle=shuffle, batch_size=batch_size,
        num_workers=num_workers, drop_last=True,
        collate_fn=seg_dataset_collate, sampler=train_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=tr_cfg["seed"]))

    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=batch_size,
        num_workers=num_workers, drop_last=False,
        collate_fn=seg_dataset_collate, sampler=val_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=tr_cfg["seed"]))

    #检查数据是否太少
    if len(train_loader) == 0:
        raise ValueError("train_loader is empty.")
    if len(val_loader) == 0:
        raise ValueError("val_loader is empty.")

    # ---- 模型 ----
    #输出shape = [B, num_classes, H, W]
    model = SegFormer(num_classes=num_classes, pretrained=md_cfg["pretrained"])
    model_path = md_cfg.get("model_path", "")
    model = load_pretrained_weights(model=model, model_path=model_path, device=device)

    model_train = model.train()

    # 多卡同步 BN
    # 如果开启SyncBN，而且是多卡DDP，就把普通BatchNorm转成SyncBatchNorm
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        if local_rank == 0:
            print("Sync_bn is not support in one gpu or not distributed.")

    if tr_cfg["cuda"]:
        if distributed:
            #先把模型移动到当前GPU
            model_train = model_train.cuda(local_rank)
            #用DistributesDataParallel包起来；find_unused_parameters = True允许有些参数在某次forward中没有参加计算
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model = model.to(device)
            model_train = model

    # ---- 优化器 ----
    #类别权重，例如[1.0, 1.0]表示背景和前景loss权重一样
    epochs = tr_cfg["epochs"]
    cls_weights = torch.ones(num_classes, dtype=torch.float32).to(device)
    lr = tr_cfg["lr"]
    min_lr = lr * tr_cfg.get("min_lr_ratio", 0.01)
    lr_decay_type = tr_cfg.get("lr_decay_type", "cos")
    lr_scheduler_func = get_lr_scheduler(lr_decay_type = lr_decay_type, lr = lr, min_lr = min_lr, total_iters = epochs)
    optimizer = optim.AdamW(model_train.parameters(), lr=lr, weight_decay=tr_cfg["weight_decay"]) #权重衰退， L2正则化，防止过拟合

    # ---- 保存目录 ----
    #权重保存目录
    save_dir = tr_cfg["save_dir"]
    #每隔多少epoch保存一次周期权重
    save_period = tr_cfg.get("save_period", 5)
    #当前最佳验证loss，初始为无穷大
    best_val_loss = float("inf")
    #只让主进程创建保存目录
    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=True) #exist_ok表示目录已经存在也不报错

    # ---- 训练循环 ----
    for epoch in range(1, epochs + 1):
        current_lr = set_optimizer_lr(optimizer, lr_scheduler_func, epoch - 1)

        if local_rank == 0:
            print(f"Current lr:{current_lr:.8f}")
        if distributed:
            train_sampler.set_epoch(epoch)

        train_loss, val_loss = fit_one_epoch(
            model_train=model_train,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            cls_weights=cls_weights,
            num_classes=num_classes,
            use_dice=True,
            local_rank=local_rank,
        )

        #只有主进程保存权重
        if local_rank == 0:
            state_dict = model.state_dict()

            last_path = os.path.join(save_dir, "last_epoch_weights.pth")
            torch.save(state_dict, last_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(save_dir, "best_epoch_weights.pth")
                torch.save(state_dict, best_path)
                print(f"Saved best model: {best_path}")

            #周期性保存模型
            if epoch % save_period == 0 or epoch == epochs:
                ckpt_path = os.path.join(save_dir, f"epoch_{epoch:03d}_loss_{val_loss:.4f}.pth")
                torch.save(state_dict, ckpt_path)
                print(f"Saved periodic checkpoint: {ckpt_path}")

        #DDP同步所有进程,确保所有GPU都完成当前epoch，再进入下一轮
        if distributed:
            dist.barrier()

    #训练结束后销毁DDP进程组
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
