import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
# import matplotlib.pyplot as plt

from torchvision import transforms
from modules import fpn
from PIL import Image
from modules.arg_parser import parse_arguments
args = parse_arguments()
args.in_dim = 128
args.K = 3


class Args:
    def __init__(self):
        pass


def collate_eval(batch):
    indice = [b[0] for b in batch]
    image = torch.stack([b[1] for b in batch])
    if batch[0][-1] is not None:
        label = torch.stack([b[2] for b in batch])

        return indice, image, label
    else:
        return indice, image


def compute_dist(featmap, metric_function, euclidean_train=True):
    centroids = metric_function.module.weight.data
    if euclidean_train:
        return - (1 - 2 * metric_function(featmap)+ (centroids * centroids).sum(dim=1).unsqueeze(0))  # negative l2 squared
    else:
        return metric_function(featmap)

def lb_map(x):
    rst = np.zeros([64,64,3],dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            curr = x[i,j]
            rst[i,j] = [0,0,240] if curr == 1 else [240,240,0] if curr == 0 else [240,0,0]
    return rst

preprocess = transforms.Compose([transforms.ToTensor()])

picie_load = torch.load('./models/seg_model.pth')

model = fpn.PanopticFPN(args)
model = nn.DataParallel(model)
model.load_state_dict(picie_load['state_dict'])
model.cuda()

classifier = nn.Conv2d(args.in_dim, args.K, kernel_size=1, stride=1, padding=0, bias=True)
classifier = nn.DataParallel(classifier)
classifier.load_state_dict(picie_load['classifier1_state_dict'])
classifier.cuda()

model.eval()
classifier.eval()

fused_dir = './results/fused_imgs/'
label_dir = './results/labels/'

for each in os.listdir(fused_dir):
    img_raw = Image.open(os.path.join(fused_dir,each))
    img = preprocess(img_raw)[None, :]
    out = model(img)
    out = F.normalize(out, dim=1, p=2)
    prb = compute_dist(out, classifier)
    lbl = prb.topk(1, dim=1)[1]
    lbl = lbl.squeeze(0).squeeze(0)
    lbm = lb_map(lbl)
    Image.fromarray(lbm).save(os.path.join(label_dir,each))
    print('Label of {0} Saved!'.format(each))