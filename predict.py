import colorsys
import os
import sys
#sys用于读取命令行参数

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from src.model import load_inference_model
from src.dataloader import preprocess_input


def cvtColor(image):
    """
    将图像转换成RGB图像，防止灰度图在预测时报错
    """
    #判断image是否是一个至少三维，并且第三个维度大小为3的数组
    if len(np.shape(image)) >= 3 and np.shape(image)[2] == 3:
        return image
    image = image.convert('RGB')
    return image


def resize_image(image, size):
    """
    等比例缩放，空白区域填灰色(128,128,128)
    size: (W, H)
    返回: (new_image, nw, nh)
    """
    #原图宽和高
    iw, ih = image.size
    #目标宽和高
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    #创建一张目标尺寸的灰色背景图，把缩放后的图片贴到中间
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    #返回new_image 补灰边后的图片；nw缩放后真实图像宽度；nh缩放后真实图像高度
    return new_image, nw, nh


def gen_colors(num_classes):
    """生成每个类别的可视颜色"""
    if num_classes <= 21:
        colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
                  (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
                  (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
                  (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                  (0, 64, 128), (128, 64, 12)]
    #如果类别很多，使用HSC均匀生成颜色
    else:
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    return colors


# --------------------------------------------------------#
#   核心预测 —— 返回原始类别索引 mask
# --------------------------------------------------------#
def predict(model, image, input_shape, device):
    """
    预测单张图片，返回原始分割mask（类别索引图）

    model: SegFormer 模型
    image: PIL.Image
    input_shape: (H, W)
    device: torch.device

    返回: np.ndarray, shape=(H, W), dtype=np.uint8, 值域[0, num_classes-1]
    """
    #确保图片是RGB
    image = cvtColor(image)
    #记录原图尺寸,NumPy图像shape是[H, W, C]
    ori_h, ori_w = np.array(image).shape[:2]
    
    #等比例resize，并补灰边
    #input_shape是(H,W)，resize_image要的是(W, H)
    image_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))
    #1.把PIL图像转换成NumPy数组
    image_array = np.array(image_data)
    #2.把数据类型转换成float32
    image_float = image_array.astype(np.float32)
    #3.根据模型要求预处理像素值
    image_preprocessed = preprocess_input(image_float)
    #4.调整维度顺序，PyTorch卷积模型要求通道在前
    image_chw = np.transpose(image_preprocessed, (2, 0, 1))
    #5. 增加batch维度
    image_data = np.expand_dims(image_chw, axis = 0)
    #image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data).astype(np.float32)), (2, 0, 1)), 0)
   
    #关闭梯度，进入预测模式
    with torch.no_grad():
        #NumPy转PyTorch Tensor,并移动到GPU/CPU
        images = torch.from_numpy(image_data).float().to(device)
        #模型前向预测，假设模型输出[1, num_classes, H, W],这里取出第一张图[num_classes, H, W]
        #pr是logits,不是概率
        pr = model(images)[0]
        #softmax得到每个类别的概率
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

        #计算灰边起始位置
        h_start = int((input_shape[0] - nh) // 2)
        w_start = int((input_shape[1] - nw) // 2)
        #裁掉灰边，只保留真实图片区域的预测概率
        pr = pr[h_start:h_start + nh, w_start:w_start + nw]

        # 恢复到原图大小,恢复的是概率图，不是类别图，所以用线性插值合理
        pr = cv2.resize(pr, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
        # 取概率最大的类别,shape : [ori_h, ori_w, num_classes] -> [ori_h, ori_w]
        pr = pr.argmax(axis=-1)

    return pr.astype(np.uint8)


# --------------------------------------------------------#
#   可视化 —— 基于 mask 生成彩色图和 overlay
# --------------------------------------------------------#
def mask_to_color(mask, colors):
    """
    将类别索引 mask 转为彩色分割图

    mask: np.ndarray, shape=(H, W), dtype=uint8
    colors: 颜色列表

    返回: PIL.Image (RGB)
    """
    H, W = mask.shape
    #第一步 把颜色表转换成uint8类型的NumPy数组
    color_table = np.array(colors, dtype = np.uint8)
    #第二步 把二维mask拉平成一维
    flat_mask = np.reshape(mask, [-1])
    #第三步 根据类别索引查询颜色
    flat_color_image = color_table[flat_mask]
    #第四步 恢复为彩色图像形状
    seg_img = np.reshape(flat_color_image, [H, W, -1])
    #seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(mask, [-1])], [H, W, -1])
    return Image.fromarray(seg_img)


def mask_to_overlay(image, mask, colors, alpha=0.7):
    """
    原图与彩色分割图混合

    image: PIL.Image 原始图片
    mask: np.ndarray 类别索引图
    colors: 颜色列表
    alpha: 彩色分割图的混合比例

    返回: PIL.Image (RGB)
    """
    image = cvtColor(image)
    color_img = mask_to_color(mask, colors)
    return Image.blend(image, color_img, alpha)


def main():
    if len(sys.argv) < 2:
        print("用法: python predict.py 图片路径")
        print("例如: python predict.py VOCdevkit/VOC2007/JPEGImages/xxx.jpg")
        sys.exit(1)
    #读取命令行中的图片路径
    img_path = sys.argv[1]

    # 读取配置
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    pred_cfg = cfg.get("predict", {})
    num_classes = cfg["model"]["num_classes"]
    input_shape = tuple(cfg["dataset"]["input_shape"])
    model_path = pred_cfg.get("model_path", "outputs/checkpoints/best_epoch_weights.pth")
    output_dir = pred_cfg.get("output_dir", "outputs/predict")

    # 选择设备
    device = torch.device("cuda" if cfg["train"]["cuda"] and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print(f"加载权重: {model_path}")
    model = load_inference_model(num_classes=num_classes, model_path=model_path, device=device)

    colors = gen_colors(num_classes)

    # 加载图片
    image = Image.open(img_path)
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 核心预测：获取原始类别索引 mask
    mask = predict(model, image, input_shape, device)

    # 保存 mask（类别索引 PNG，供 evaluate.py 使用）
    # mode = 'L'表示单通道灰度图
    mask_img = Image.fromarray(mask, mode='L')
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    mask_img.save(mask_path)
    print(f"Mask 已保存: {mask_path}")

    # 保存 overlay（原图+分割混合，供可视化）
    overlay = mask_to_overlay(image, mask, colors)
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    overlay.save(overlay_path)
    print(f"Overlay 已保存: {overlay_path}")


if __name__ == "__main__":
    main()
