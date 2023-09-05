import random
import os
import time
import logging
from logging import handlers
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import faiss


################################################################################
#                                  General-purpose                             #
################################################################################

class Logger(object):
    def __init__(self, logpth, fmt='%(message)s'):
        # '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
        logfile = 'MediFusion-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
        if os.path.exists(logpth) == True:  # 文件存在
            logfile = os.path.join(logpth, logfile)
        else:
            os.makedirs(logpth)  # 创建文件夹
            logfile = os.path.join(logpth, logfile)
        self.logger = logging.getLogger(logfile)   # 指定日志文件名
        format_str = logging.Formatter(fmt) # 设置日志格式
        self.logger.setLevel(logging.INFO)   # 设置日志级别
        sh = logging.StreamHandler()    # 往屏幕上输出
        sh.setFormatter(format_str)     # 设置屏幕上显示的格式
        th = logging.FileHandler(logfile)
        th.setFormatter(format_str) # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里 屏幕输出
        self.logger.addHandler(th)  # 把对象加到logger里 文件写入
    def getLog(self):
        return self.logger

# def setup_logger(logpth):
#     logfile = 'MediFusion-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
#     logfile = os.path.join(logpth, logfile)
#     FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
#     log_level = logging.INFO
#     # if dist.is_initialized() and not dist.get_rank()==0:
#     #     log_level = logging.ERROR
#     logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
#     logging.root.addHandler(logging.StreamHandler())
#     logger = logging.getLogger()
#     return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_datetime(time_delta):
    days_delta = time_delta // (24*3600)
    time_delta = time_delta % (24*3600)
    hour_delta = time_delta // 3600
    time_delta = time_delta % 3600
    mins_delta = time_delta // 60
    time_delta = time_delta % 60
    secs_delta = time_delta

    return '{}:{}:{}:{}'.format(days_delta, hour_delta, mins_delta, secs_delta)

################################################################################
#                                Metric-related ops                            #
################################################################################
def get_metric_as_conv(centroids):
    N, C = centroids.size()
    centroids_weight = centroids.unsqueeze(-1).unsqueeze(-1)
    metric_function = nn.Conv2d(C, N, 1, padding=0, stride=1, bias=False)
    metric_function.weight.data = centroids_weight
    metric_function = nn.DataParallel(metric_function)
    metric_function = metric_function.cuda()
    return metric_function

def compute_negative_euclidean(featmap, centroids, metric_function):
    centroids = centroids.unsqueeze(-1).unsqueeze(-1)
    return - (1 - 2*metric_function(featmap) + (centroids*centroids).sum(dim=1).unsqueeze(0)) # negative l2 squared

################################################################################
#                                General torch ops                             #
################################################################################
def initialize_classifier(args):
    classifier = get_linear(args.seg_in_dim, args.seg_K_train)
    classifier = nn.DataParallel(classifier)
    classifier = classifier.cuda()
    return classifier


def get_linear(indim, outdim):
    classifier = nn.Conv2d(indim, outdim, kernel_size=1, stride=1, padding=0, bias=True)
    classifier.weight.data.normal_(0, 0.01)
    classifier.bias.data.zero_()
    return classifier


def feature_flatten(feats):
    if len(feats.size()) == 2:
        # feature already flattened.
        return feats
    feats = feats.view(feats.size(0), feats.size(1), -1).transpose(2, 1) \
        .contiguous().view(-1, feats.size(1))
    return feats

def freeze_all(model):
    for param in model.module.parameters():
        param.requires_grad = False


################################################################################
#                                   Faiss related                              #
################################################################################

def get_faiss_module(args):
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = 0  # NOTE: Single GPU only.
    idx = faiss.GpuIndexFlatL2(res, args.seg_in_dim, cfg)
    return idx

def get_init_centroids(args, K, featlist, index):
    clus = faiss.Clustering(args.seg_in_dim, K)
    clus.seed  = np.random.randint(2023)
    clus.niter = args.seg_kmeans_n_iter
    clus.max_points_per_centroid = 10000000
    clus.train(featlist, index)
    return faiss.vector_float_to_array(clus.centroids).reshape(K, args.seg_in_dim)

def module_update_centroids(index, centroids):
    index.reset()
    index.add(centroids)
    return index

def fix_seed_for_reproducability(seed):
    """
    Unfortunately, backward() of [interpolate] functional seems to be never deterministic.
    Below are related threads:
    https://github.com/pytorch/pytorch/issues/7068
    https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842?u=sbelharbi
    """
    # Use random seed.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def worker_init_fn(seed):
    return lambda x: np.random.seed(seed + x)

################################################################################
#                               Training Pipelines                             #
################################################################################
def get_transform_params(args):
    inv_list = []
    eqv_list = []
    if args.seg_augment:
        if args.seg_blur:
            inv_list.append('blur')
        if args.seg_grey:
            inv_list.append('grey')
        if args.seg_jitter:
            inv_list.extend(['brightness', 'contrast', 'saturation', 'hue'])
        if args.seg_equiv:
            if args.seg_h_flip:
                eqv_list.append('h_flip')
            if args.seg_v_flip:
                eqv_list.append('v_flip')
            if args.seg_random_crop:
                eqv_list.append('random_crop')
    return inv_list, eqv_list


def collate_sd_train_seg(batch):
    if batch[0][-1] is not None:
        indice = [b[0] for b in batch]
        image1 = torch.stack([b[1] for b in batch])
        image2 = torch.stack([b[2] for b in batch])
        label1 = torch.stack([b[3] for b in batch])
        label2 = torch.stack([b[4] for b in batch])
        return indice, image1, image2, label1, label2
    indice = [b[0] for b in batch]
    image1 = torch.stack([b[1] for b in batch])
    return indice, image1


def eqv_transform_if_needed(args, dataloader, indice, input):
    if args.seg_equiv:
        input = dataloader.dataset.transform_eqv(indice, input)
    return input


def postprocess_label(args, K, idx, idx_img, scores, n_dual):
    # out = scores[idx].topk(1, dim=0)[1].flatten().detach().cpu().numpy()
    out = scores[idx].topk(1, dim=0)[1].detach().cpu().numpy()
    # Save labels.
    if not os.path.exists(os.path.join(args.label_dir, 'label_' + str(n_dual))):
        os.makedirs(os.path.join(args.label_dir, 'label_' + str(n_dual)))
    torch.save(out, os.path.join(args.label_dir, 'label_' + str(n_dual), '{}.pkl'.format(idx_img)))
    # Count for re-weighting.
    counts = torch.tensor(np.bincount(out.flatten(), minlength=K)).float()
    return counts