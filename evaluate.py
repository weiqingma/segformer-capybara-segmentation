import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from src.model import load_inference_model
from predict import predict

'''
a: 真实标签，通常已经拉平成一维，形状为[HxW]
b: 模型预测标签，一维，形状为[HxW]
n:类别数量
返回值：形状为[n, n]的混淆矩阵 hist[真实类别，预测类别]
'''
def _fast_hist(a, b, n):
    #得到布尔数组，表示哪些像素的真实标签是合格的
    k = (a >= 0) & (a < n)
    #bincount统计每个组合出现的次数
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


class SegmentationMetrics:
    """
    语义分割评估器：累积混淆矩阵，从中计算各类指标。

    用法:
        metrics = SegmentationMetrics(num_classes=2, name_classes=["bg", "capybara"])
        for pred, label in zip(preds, labels):
            metrics.update(pred, label)

        metrics.print_report()
        print(metrics.miou, metrics.mpa, metrics.accuracy)
    """
    def __init__(self, num_classes, name_classes=None):
        self.num_classes = num_classes
        self.name_classes = name_classes if name_classes is not None else [str(i) for i in range(num_classes)]
        self.reset()

    #重置混淆矩阵
    def reset(self):
        self._hist = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred, label):
        self._hist += _fast_hist(label.flatten(), pred.flatten(), self.num_classes)

    @property
    def hist(self):
        return self._hist.copy()

    # ---------- 对于每个类别的性质 per-class properties (1-D array, length = num_classes) ----------
    @property
    def tp(self):
        return np.diag(self._hist)
    
    @property
    #FP = 预测为该类的总数 - 预测正确的数量
    def fp(self):
        return self._hist.sum(axis = 0) - self.tp
    
    @property
    #FN = 真实为该类的总数 - 预测正确的数量
    def fn(self):
        return self._hist.sum(axis = 1) - self.tp
    
    #IoU = TP/ (TP + FP + FN)
    @property
    def iou(self):
        denominator = self.tp + self.fp + self.fn
        return self.tp / np.maximum(denominator, 1)

    #Precision = TP / (TP + FP)
    @property
    def precision(self):
        denominator = self.tp + self.fp
        return self.tp / np.maximum(denominator, 1)

    #Recall = TP / (TP + FN)
    @property
    def recall(self):
        denominator = self.tp + self.fn
        return self.tp / np.maximum(denominator, 1)

    #Dice = 2TP / (2TP + FP + FN)
    @property
    def dice(self):
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / np.maximum(denominator, 1)

    # ---------- 整体像素准确率 global properties (scalar) ----------
    @property
    def accuracy(self):
        return np.sum(np.diag(self._hist)) / np.maximum(np.sum(self._hist), 1)

    @property
    #各类别mIoU的平均值
    def miou(self):
        return np.nanmean(self.iou)

    @property
    #Mean Pixel Accuracy
    def mpa(self):
        return np.nanmean(self.recall)

    @property
    def mprecision(self):
        return np.nanmean(self.precision)

    @property
    def mdice(self):
        return np.nanmean(self.dice)

    # ---------- output ----------
    def per_class_report(self):
        """逐类别指标 dict，key 为类别名"""
        return {
            name: {"IoU": round(self.iou[i] * 100, 2),
                   "Recall": round(self.recall[i] * 100, 2),
                   "Precision": round(self.precision[i] * 100, 2),
                   "Dice": round(self.dice[i] * 100, 2)}
            for i, name in enumerate(self.name_classes)
        }

    def summary(self):
        """返回汇总 dict"""
        return {
            "mIoU": round(self.miou * 100, 2),
            "mPA": round(self.mpa * 100, 2),
            "mPrecision": round(self.mprecision * 100, 2),
            "mDice": round(self.mdice * 100, 2),
            "Accuracy": round(self.accuracy * 100, 2),
        }

    def report(self):
        """返回完整结构化评估结果。"""
        return {
            "per_class": self.per_class_report(),
            "summary": self.summary(),
        }

    def print_report(self):
        report = self.report()
        for name, metrics in report["per_class"].items():
            print(f"===>{name}:\tIoU-{metrics['IoU']:.2f}"
                  f"; Recall (equal to the PA)-{metrics['Recall']:.2f}"
                  f"; Precision-{metrics['Precision']:.2f}"
                  f"; Dice-{metrics['Dice']:.2f}")
        s = report["summary"]
        print(f"===> mIoU: {s['mIoU']:.2f}; mPA: {s['mPA']:.2f}; "
              f"mPrecision: {s['mPrecision']:.2f}; mDice: {s['mDice']:.2f}; "
              f"Accuracy: {s['Accuracy']:.2f}")

    #每处理10张照片，打印一次中间指标
    def _mid_progress(self, current, total):
        if self.name_classes is not None and current > 0 and current % 10 == 0:
            print(f"{current} / {total}: mIou-{self.miou * 100:.2f}%; mPA-{self.mpa * 100:.2f}%; Accuracy-{self.accuracy * 100:.2f}%")

    @classmethod
    #gt_dir:真实mask目录； pred_dir: 预测mask目录; png_name_list：图片ID列表; num_classes:类别数; verbose:是否打印过程信息
    def from_images(cls, gt_dir, pred_dir, png_name_list, num_classes, name_classes=None, verbose=True):
        """从两组 PNG 文件夹直接构建评估器。"""
        metrics = cls(num_classes=num_classes, name_classes=name_classes)
        #生成真实mask和预测mask路径列表
        gt_imgs = [os.path.join(gt_dir, x + ".png") for x in png_name_list]
        pred_imgs = [os.path.join(pred_dir, x + ".png") for x in png_name_list]

        for ind in range(len(gt_imgs)):
            pred = np.array(Image.open(pred_imgs[ind]))
            label = np.array(Image.open(gt_imgs[ind]))

            #如果预测图和真实图尺寸不同，就跳过
            if len(label.flatten()) != len(pred.flatten()):
                if verbose:
                    print(f"Skipping: len(gt) = {len(label.flatten())}, len(pred) = {len(pred.flatten())}, {gt_imgs[ind]}, {pred_imgs[ind]}")
                continue

            #更新混淆矩阵
            metrics.update(pred, label)
            if verbose:
                metrics._mid_progress(ind + 1, len(gt_imgs))

        return metrics


def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None, mask_255_to_1=False):
    """
    兼容旧 API 的封装。推荐直接使用 SegmentationMetrics 类。
    """
    metrics = SegmentationMetrics(num_classes=num_classes, name_classes=name_classes)
    print("Num classes", num_classes)

    gt_imgs = [os.path.join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [os.path.join(pred_dir, x + ".png") for x in png_name_list]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        if mask_255_to_1:
            label[label == 255] = 1

        if len(label.flatten()) != len(pred.flatten()):
            print(f"Skipping: len(gt) = {len(label.flatten())}, len(pred) = {len(pred.flatten())}, {gt_imgs[ind]}, {pred_imgs[ind]}")
            continue

        metrics.update(pred, label)
        metrics._mid_progress(ind + 1, len(gt_imgs))

    metrics.print_report()
    #返回混淆矩阵， 每类IoU， 每类Recall， 每类Precision， 每类Dice
    return metrics.hist, metrics.iou, metrics.recall, metrics.precision, metrics.dice


if __name__ == "__main__":
    # 读取 config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    eval_cfg = cfg.get("eval", {})
    # miou_mode用于指定该文件运行时计算的内容
    # miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    # miou_mode为1代表仅仅获得预测结果。
    # miou_mode为2代表仅仅计算miou。
    miou_mode = eval_cfg.get("miou_mode", 0)

    num_classes = cfg["model"]["num_classes"]
    name_classes = cfg["model"].get("class_names")
    if name_classes is None:
        raise ValueError("config.yaml must define model.class_names for evaluation.")
    if len(name_classes) != num_classes:
        raise ValueError("model.class_names length must equal model.num_classes.")

    input_shape = tuple(cfg["dataset"]["input_shape"])
    VOCdevkit_path = cfg["dataset"]["voc_root"]
    mask_255_to_1 = cfg["dataset"].get("mask_255_to_1", False)

    #读取验证图片ID
    image_ids = open(os.path.join(VOCdevkit_path, cfg["dataset"]["val_txt"]), 'r').read().splitlines()
    #真实mask目录
    gt_dir = os.path.join(VOCdevkit_path, cfg["dataset"]["mask_dir"])
    #评估输出目录
    miou_out_path = eval_cfg.get("output_dir", "outputs/eval")
    pred_dir = os.path.join(miou_out_path, eval_cfg.get("pred_dir", "detection-results"))

    #模式0或1，先生成预测结果
    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        device = torch.device("cuda" if cfg["train"]["cuda"] and torch.cuda.is_available() else "cpu")
        #读取权重路径
        model_path = sys.argv[1] if len(sys.argv) > 1 else eval_cfg.get("model_path", "outputs/checkpoints/best_epoch_weights.pth")
        model = load_inference_model(num_classes=num_classes, model_path=model_path, device=device)
        print("Load model done.")

        print("Get predict result.")
        #遍历验证集中的每张照片
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, cfg["dataset"]["image_dir"], image_id + ".jpg")
            image = Image.open(image_path)
            #pred_mask.shape = [H, W]
            pred_mask = predict(model, image, input_shape, device)
            Image.fromarray(pred_mask).save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes, mask_255_to_1=mask_255_to_1)
        print("Get miou done.")
