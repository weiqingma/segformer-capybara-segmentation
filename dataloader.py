import os
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset


def preprocess_input(image):
    """
    对输入图片做标准化
    image: numpy array, shape = [H, W, 3], RGB, float32
    """
    image -= np.array([123.675, 116.28, 103.53], dtype=np.float32)
    image /= np.array([58.395, 57.12, 57.375], dtype=np.float32)
    return image


class SegmentationDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, dataset_path, train=True):
        """
        annotation_lines: train.txt / val.txt 读出来的每一行，比如 ["xxx", "yyy"]
        input_shape: [H, W]，比如 [512, 512]
        num_classes: 类别数，包含背景
        dataset_path: VOCdevkit 路径
        train: 保留这个参数只是为了兼容 train.py，当前最小版不做增强
        """
        super().__init__()

        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dataset_path = dataset_path
        self.train = train

        self.image_dir = os.path.join(dataset_path, "VOC2007", "JPEGImages")
        self.mask_dir = os.path.join(dataset_path, "VOC2007", "SegmentationClass")

    def __len__(self):
        '''
        数据有多少张
        '''
        return len(self.annotation_lines)

    def __getitem__(self, index):
        '''
        每次取一张图片时怎么处理
        图片和标签mask从哪里读
        '''
        name = self.annotation_lines[index].strip().split()[0]

        image_path = os.path.join(self.image_dir, name + ".jpg")
        mask_path = os.path.join(self.mask_dir, name + ".png")

        image = Image.open(image_path).convert("RGB")

        # 注意：mask 不要 convert("RGB")。
        # VOC 的 mask 是类别编号图，np.array(mask) 应该得到 0,1,2,... 或 255。
        mask = Image.open(mask_path)
        mask = Image.fromarray(np.array(mask, dtype=np.uint8))

        '''
        等比例缩放
        '''
        image, mask = self.resize_with_letterbox(image, mask)


        image = np.array(image, dtype=np.float32)
        image = preprocess_input(image)
        image = np.transpose(image, (2, 0, 1))  # [H,W,3] -> [3,H,W]

        '''
        把mask转成整数类别图
        '''

        mask = np.array(mask, dtype=np.int64)
        # 大于等于 num_classes 的像素作为 ignore 类。
        # 原项目 CE_Loss 里 ignore_index=num_classes，所以这里这样处理是对齐的。
        mask[mask >= self.num_classes] = self.num_classes

        '''
        Dice loss / f_score 用 one-hot 标签。
        #seg_labels最后的形状 shape: [H, W, num_classes + 1]
        '''
        seg_labels = np.eye(self.num_classes + 1, dtype=np.float32)[mask.reshape(-1)]
        seg_labels = seg_labels.reshape(
            self.input_shape[0],
            self.input_shape[1],
            self.num_classes + 1
        )

        return image, mask, seg_labels

    def resize_with_letterbox(self, image, mask):
        """
        等比例缩放，不拉伸。
        空白区域：image 填灰色，mask 填 0。
        """
        h, w = self.input_shape
        iw, ih = image.size

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        mask = mask.resize((nw, nh), Image.NEAREST)

        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_mask = Image.new("L", (w, h), 0)

        left = (w - nw) // 2
        top = (h - nh) // 2

        new_image.paste(image, (left, top))
        new_mask.paste(mask, (left, top))

        return new_image, new_mask


def seg_dataset_collate(batch):
    images = []
    masks = []
    seg_labels = []

    for image, mask, label in batch:
        images.append(image)
        masks.append(mask)
        seg_labels.append(label)

    images = torch.from_numpy(np.array(images)).float()
    masks = torch.from_numpy(np.array(masks)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).float()

    '''
    最后形状, B是batch_size
    images:     [B, 3, H, W]
    masks:      [B, H, W]
    seg_labels: [B, H, W, num_classes + 1]
    '''

    return images, masks, seg_labels