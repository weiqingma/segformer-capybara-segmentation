import math
from functools import partial

#固定学习率策略
def fixed_lr(lr, iters):
    return lr

#step下降学习率，希望经过step_num-1次下降后，初始学习率lr降到最小学习率min_lr
def step_lr(lr, min_lr, total_iters, step_num, iters):
    decay_rate = (min_lr / lr) ** (1 / (step_num -1))
    step_size = total_iters / step_num
    n = iters // step_size
    return lr * decay_rate ** n

#余弦策略，二次warmup-cosine decay-保持min_lr
'''
lr:                     目标/最大学习率
min_lr:                 最小学习率
total_iters:            总训练轮数或总迭代数
warmup_total_iters:     warmup持续多久
warmup_lr_start         warmup起始学习率
no_aug_iter             最后固定最小学习率的长度
iters                   当前epoch或iteration
'''
def warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters,
                 warmup_lr_start, no_aug_iter, iters):
    if iters <= warmup_total_iters:
        return (lr - warmup_lr_start) * ((iters / warmup_total_iters) ** 2) + warmup_lr_start
    elif iters >= total_iters - no_aug_iter:
        return min_lr
    else:
        #计算在cosine decay阶段的进度,progress的取值在0和1之间
        progress = (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter)
        #cos(pi*progress)取值在-1和1之间, 1/2（1+cos）的取值在0和1之间
        return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, 
                     warmup_iters_ratio = 0.1, 
                     warmup_lr_ratio = 0.1, 
                     no_aug_iter_ratio = 0.3, 
                     step_num = 10):
    if lr_decay_type == "fixed":
        return partial(fixed_lr, lr)
    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        return partial(warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    if lr_decay_type == "step":
        return partial(step_lr, lr, min_lr, total_iters, step_num)
    raise ValueError(f"Unsupported lr_decay_type:{lr_decay_type}")

'''
lr_scheduler_func是根据当前epoch/iteration计算学习率的函数
'''
def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr
    
def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]