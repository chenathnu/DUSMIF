import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
from sklearn.utils.linear_assignment_ import linear_assignment
from PIL import Image
import torchvision.transforms.functional as torchvision_transforms
from modules import fpn
from utils import *
import warnings

warnings.filterwarnings('ignore')


def get_model_and_optimizer(args, logger,num):
    # Init model
    model = fpn.PanopticFPN(args)
    model = nn.DataParallel(model)
    model = model.cuda()
    # Init classifier (for eval only.)
    classifier = initialize_classifier(args)
    # Init optimizer
    optimizer = None
    if args.seg_optim_type == 'SGD':
        # logger.info('SGD optimizer is used.')
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.module.parameters()), lr=args.seg_lr,
                                    momentum=args.seg_momentum, weight_decay=args.seg_weight_decay)
    elif args.seg_optim_type == 'Adam':
        # logger.info('Adam optimizer is used.')
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.module.parameters()), lr=args.seg_lr)
    args.seg_start_epoch = 0
    if num>0:
        load_path = args.seg_model_pth
        if os.path.isfile(load_path):
            checkpoint = torch.load(load_path)
            args.seg_start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            try:
                classifier.load_state_dict(checkpoint['classifier1_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                pass
            logger.info('Loaded model. [epoch {}]'.format(args.seg_start_epoch))
        else:
            logger.info('No model file found at [{}].\nStart from beginning...'.format(load_path))
    return model, optimizer, classifier


def run_mini_batch_kmeans(args, logger, dataloader, model, view):
    """
    num_init_batches: (int) The number of batches/iterations to accumulate before the initial k-means clustering.
    num_batches     : (int) The number of batches/iterations to accumulate before the next update.
    """
    kmeans_loss = AverageMeter()
    faiss_module = get_faiss_module(args)
    data_count = np.zeros(args.seg_K_train)
    featslist = []
    num_batches = 0
    first_batch = True
    # Choose which view it is now.
    dataloader.dataset.view = view
    model.train()
    with torch.no_grad():
        for i_batch, (indice, image) in enumerate(dataloader):
            # 1. Compute initial centroids from the first few batches.
            if view == 1:
                # image = eqv_transform_if_needed(args, dataloader, indice, image.cuda(non_blocking=True))
                image = eqv_transform_if_needed(args, dataloader, indice, image)
                feats = model(image)
            elif view == 2:
                # image = image.cuda(non_blocking=True)
                image = image
                feats = eqv_transform_if_needed(args, dataloader, indice, model(image))
            else:
                # For evaluation.
                # image = image.cuda(non_blocking=True)
                image = image
                feats = model(image)
            # Normalize.
            if args.seg_metric == 'cosine':
                feats = F.normalize(feats, dim=1, p=2)
            # if i_batch == 0:
            #     logger.info('Batch input size : {}'.format(list(image.shape)))
            #     logger.info('Batch feature : {}'.format(list(feats.shape)))
            feats = feature_flatten(feats).detach().cpu()
            if num_batches < args.seg_num_init_batches:
                featslist.append(feats)
                num_batches += 1
                if num_batches == args.seg_num_init_batches or num_batches == len(dataloader):
                    if first_batch:
                        # Compute initial centroids.
                        # By doing so, we avoid empty cluster problem from mini-batch K-Means.
                        featslist = torch.cat(featslist).cpu().numpy().astype('float32')
                        centroids = get_init_centroids(args, args.seg_K_train, featslist, faiss_module).astype('float32')
                        D, I = faiss_module.search(featslist, 1)
                        kmeans_loss.update(D.mean())
                        # logger.info('Initial k-means loss: {:.4f} '.format(kmeans_loss.avg))
                        # Compute counts for each cluster.
                        for k in np.unique(I):
                            data_count[k] += len(np.where(I == k)[0])
                        first_batch = False
                    else:
                        b_feat = torch.cat(featslist)
                        faiss_module = module_update_centroids(faiss_module, centroids)
                        D, I = faiss_module.search(b_feat.numpy().astype('float32'), 1)
                        kmeans_loss.update(D.mean())
                        # Update centroids.
                        for k in np.unique(I):
                            idx_k = np.where(I == k)[0]
                            data_count[k] += len(idx_k)
                            centroid_lr = len(idx_k) / (data_count[k] + 1e-6)
                            centroids[k] = (1 - centroid_lr) * centroids[k] + centroid_lr * b_feat[idx_k].mean(
                                0).numpy().astype('float32')
                    # Empty.
                    featslist = []
                    num_batches = args.seg_num_init_batches - args.seg_num_batches
            # if (i_batch % 100) == 0:
            #     logger.info('[Saving features]: {} / {} | [K-Means Loss]: {:.4f}'.format(i_batch, len(dataloader),
            #                                                                              kmeans_loss.avg))
    centroids = torch.tensor(centroids, requires_grad=False).cuda()
    return centroids, kmeans_loss.avg


def get_view_img_feat(it, indice, args, view, image, model, transform_tools):
    image_detach = image.detach().cpu().numpy().transpose((0, 2, 3, 1))
    image_detach = (image_detach - np.min(image_detach)) / (np.max(image_detach) - np.min(image_detach))
    image_detach = np.uint8(255.0 * image_detach)
    image_re = torch.Tensor(image.shape).cuda()
    if view == 1:
        for l in range(image.shape[0]):
            image_detach[l] = transform_tools.transform_inv(it, Image.fromarray(image_detach[l]), 0)
            image_re[l] = transform_tools.transform_tensor(image_detach[l])
        image_re = transform_tools.transform_eqv(indice, image_re.cuda(non_blocking=True))
        feats = model(image_re)
    elif view == 2:
        for l in range(image.shape[0]):
            image_detach[l] = transform_tools.transform_inv(it, Image.fromarray(image_detach[l]), 1)
            image_re[l] = transform_tools.transform_tensor(image_detach[l])
        # image_re = torchvision_transforms.resize(image_re, args.seg_res1, Image.BILINEAR)
        image_re = image_re.cuda(non_blocking=True)
        feats = transform_tools.transform_eqv(indice, model(image_re))
    if args.seg_metric == 'cosine':
        feats = F.normalize(feats, dim=1, p=2)
    return feats


def run_batch_kmeans_for_single(args, feats, faiss_module):
    kmeans_loss = AverageMeter()

    data_count = np.zeros(args.seg_K_train)
    featslist = []
    feats = feature_flatten(feats).detach().cpu()
    featslist.append(feats)
    featslist = torch.cat(featslist).cpu().numpy().astype('float32')
    centroids = get_init_centroids(args, args.seg_K_train, featslist, faiss_module).astype('float32')
    D, I = faiss_module.search(featslist, 1)
    kmeans_loss.update(D.mean())
    for k in np.unique(I):
        data_count[k] += len(np.where(I == k)[0])
    centroids = torch.tensor(centroids, requires_grad=False).cuda()
    return centroids, kmeans_loss.avg


def compute_labels(args, logger, dataloader, model, centroids, view):
    """
    Label all images for each view with the obtained cluster centroids.
    The distance is efficiently computed by setting centroids as convolution layer.
    """
    K = centroids.size(0)
    # Choose which view it is now.
    dataloader.dataset.view = view
    # Define metric function with conv layer.
    metric_function = get_metric_as_conv(centroids)
    counts = torch.zeros(K, requires_grad=False).cpu()
    model.eval()
    with torch.no_grad():
        for i, (indice, image) in enumerate(dataloader):
            if view == 1:
                image = eqv_transform_if_needed(args, dataloader, indice, image.cuda(non_blocking=True))
                feats = model(image)
            elif view == 2:
                image = image.cuda(non_blocking=True)
                feats = eqv_transform_if_needed(args, dataloader, indice, model(image))
            # Normalize.
            if args.seg_metric == 'cosine':
                feats = F.normalize(feats, dim=1, p=2)
            B, C, H, W = feats.shape
            # if i == 0:
            #     logger.info('Centroid size      : {}'.format(list(centroids.shape)))
            #     logger.info('Batch input size   : {}'.format(list(image.shape)))
            #     logger.info('Batch feature size : {}'.format(list(feats.shape)))
            # Compute distance and assign label.
            scores  = compute_negative_euclidean(feats, centroids, metric_function)
            # Save labels and count.
            for idx, idx_img in enumerate(indice):
                counts += postprocess_label(args, K, idx, idx_img, scores, n_dual=view)
            # if (i % 200) == 0:
            #     logger.info('[Assigning labels] {} / {}'.format(i, len(dataloader)))
    weight = counts / counts.sum()
    return weight

def compute_labels_for_single(centroids, feats):
    K = centroids.size(0)
    metric_function = get_metric_as_conv(centroids)
    counts = torch.zeros(K, requires_grad=False).cpu()
    B, C, H, W = feats.shape
    scores = compute_negative_euclidean(feats, centroids, metric_function)
    label = torch.LongTensor(B,H,W)
    for ind in range(B):
        label[ind] = scores[ind].topk(1, dim=0)[1][0]
        counts += torch.tensor(np.bincount(scores[ind].topk(1, dim=0)[1].flatten().detach().cpu().numpy(), minlength=K)).float()
    weight = counts / counts.sum()
    return torch.LongTensor(label).cuda(), weight
