import os
import random
import yaml

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SegFormer
from dataloader import SegmentationDataset, seg_dataset_collate
from trainer import fit_one_epoch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_pretrained_weights(model, model_path, device):
    if not model_path:
        return model
    if not os.path.exists(model_path):
        print(f"Warning: model_path'{model_path}'does not exist, training from scratch.")
        return model
    print(f"Loading model weights from {model_path}")

    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location = device)

    load_key, no_load_key, temp_dict = [], [], {}

    for k, v in pretrained_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)

    print(f"Successful load keys: {len(load_key)}")
    print(f"Failed load keys: {len(no_load_key)}")
    print("分类头没有载入是正常的；backbone 大量没载入才是不正常的。")

    return model




def main():
    '''
    读取配置文件
    '''
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    '''
    固定随机种子
    '''
    seed_everything(cfg["train"]["seed"])

    '''
    选择训练设备
    '''
    device = torch.device("cuda" if cfg["train"]["cuda"] and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- dataset ---
    ds_cfg = cfg["dataset"]
    voc_root = ds_cfg["voc_root"]
    num_classes = cfg["model"]["num_classes"]
    input_shape = tuple(ds_cfg["input_shape"])  # (H, W)

    '''
    读取训练集和测试集的样本列表
    '''
    train_txt = os.path.join(voc_root, ds_cfg["train_txt"])
    val_txt = os.path.join(voc_root, ds_cfg["val_txt"])
    
    with open(train_txt, "r") as f:
        train_lines = [line.strip() for line in f.readlines() if line.strip()]
    with open(val_txt, "r") as f:
        val_lines = [line.strip() for line in f.readlines() if line.strip()]

    '''
    创建dataset
    '''
    train_dataset = SegmentationDataset(annotation_lines = train_lines, input_shape = input_shape, num_classes = num_classes, 
                                        dataset_path = voc_root, train = True)
    val_dataset = SegmentationDataset(annotation_lines = val_lines, input_shape = input_shape, num_classes = num_classes,
                                       dataset_path = voc_root, train = False)
    '''
    读取train配置
    '''
    tr_cfg = cfg["train"]

    '''
    创建训练和验证dataloader,把dataset包装成可迭代的数据加载器
    '''
    train_loader = DataLoader(
        train_dataset,
        batch_size=tr_cfg["batch_size"],
        shuffle=True,
        num_workers=tr_cfg["num_workers"],
        drop_last=True,
        collate_fn=seg_dataset_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=tr_cfg["batch_size"],
        shuffle=False,
        num_workers=tr_cfg["num_workers"],
        drop_last=False,
        collate_fn=seg_dataset_collate,
    )

    if len(train_loader) == 0:
        raise ValueError("train_loader is empty. Check train.txt, batch_size, and drop_last.")

    if len(val_loader) == 0:
        raise ValueError("val_loader is empty. Check val.txt, batch_size, and drop_last.")
    '''
    创建模型
    '''
    md_cfg = cfg["model"]
    model = SegFormer(num_classes=num_classes, pretrained=md_cfg["pretrained"])
    '''
    读取模型权重路径
    '''
    model_path = md_cfg.get("model_path", "")
    model = load_pretrained_weights(model = model, model_path = model_path, device = device)
    model = model.to(device)

    '''
    设置类别权重    # --- optimizer & loss weights ---
    '''

    cls_weights = torch.ones(num_classes, dtype=torch.float32).to(device)

    '''
    创建优化器
    '''

    optimizer = optim.AdamW(model.parameters(), lr=tr_cfg["lr"], weight_decay=tr_cfg["weight_decay"])

    '''
    创建保存目录
    '''
    save_dir = tr_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    save_period = tr_cfg.get("save_period", 5)
    best_val_loss =float("inf")

    '''
    训练循环
    '''
    epochs = tr_cfg["epochs"]
    for epoch in range(1, epochs + 1):
        train_loss, val_loss = fit_one_epoch(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            cls_weights=cls_weights,
            num_classes=num_classes,
            use_dice=True,
        )
        '''
        保存模型权重：best、last、定期checkpoint
        '''
        # 永远保存最后一轮，文件名固定，会覆盖旧文件
        last_path = os.path.join(save_dir,"last_epoch_weights.pth")
        torch.save(model.state_dict(), last_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(save_dir, "best_epoch_weights.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model:{best_path}")

        if epoch % save_period == 0 or epoch == epochs:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch:03d}_loss_{val_loss:.4f}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved periodic checkpoint:{ckpt_path}")


if __name__ == "__main__":
    main()
