import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from PIL import Image
import numpy as np
from torch.autograd import Variable
from dataset import FusionDataset, SegDataset
from modules.arg_parser import parse_arguments
import datetime
import time
import logging
import os.path as osp
import os
from FusionNet import FusionNet
from loss import OhemCELoss, Fusionloss, Segloss
from optimizer import Optimizer
import torch
from torch.utils.data import DataLoader
from commons import *
from utils import *
from pytorch_msssim import SSIM
from modules import custom_transforms
import warnings

warnings.filterwarnings('ignore')

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def train_fusion(num=0, logger=None):
    # num: control the segmodel
    lr_start = 0.001
    fusionmodel = FusionNet()
    fusionmodel.cuda()
    fusionmodel.train()
    if num>0:
        fusionmodel.load_state_dict(torch.load(args.fusion_model_pth))
    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    train_dataset = FusionDataset(args)
    logger.info("the training dataset is length:{}".format(train_dataset.length))
    if num > 0:
        seg_model, _, _ = get_model_and_optimizer(args, logger, num)
        seg_model.eval()
        inv_list, eqv_list = get_transform_params(args)
        transform_tools = SegDataset(args, inv_list=inv_list, eqv_list=eqv_list)
        transform_tools.N = train_dataset.length
        transform_tools.init_transforms()
        faiss_module = get_faiss_module(args)
        logger.info('Load Segmentation Model Sucessfully~')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn(args.seed)
    )
    train_loader.n_iter = len(train_loader)
    if num > 0:
        criteria_seg = Segloss()
    criteria_ssim = SSIM(win_size=5, win_sigma=1.5, data_range=1, size_average=True, channel=1)
    criteria_fusion = Fusionloss()
    best_fusion_loss = np.inf
    epoch = 20
    st = glob_st = time.time()
    logger.info('Training Fusion Model start~')
    for epo in range(0, epoch):
        # logger.info('fusion train | epo #%s begin...' % epo)
        lr_start = 0.001
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo
        for it, (image_mr, image_ct, name, indice) in enumerate(train_loader):
            fusionmodel.train()
            image_ct = Variable(image_ct).cuda()
            if not args.one_in_rgb:
                image_mr = Variable(image_mr).cuda()
                image_fused = fusionmodel(image_mr, image_ct)
                image_fused = image_fused.expand(-1, 3, -1, -1)
            else:
                image_mr = Variable(image_mr).cuda()
                image_mr_ycrcb = RGB2YCrCb(image_mr)
                image_fused = fusionmodel(image_mr[:,:1], image_ct)
                image_fused = torch.cat(
                    (image_fused, image_mr_ycrcb[:, 1:2, :, :],
                     image_mr_ycrcb[:, 2:, :, :]),
                    dim=1,
                )
                image_fused = YCrCb2RGB(image_fused)
            ones = torch.ones_like(image_fused)
            zeros = torch.zeros_like(image_fused)
            image_fused = torch.where(image_fused > ones, ones, image_fused)
            image_fused = torch.where(image_fused < zeros, zeros, image_fused)
            optimizer.zero_grad()
            # seg loss
            seg_loss = 0
            ssim_loss_1 = 1 - criteria_ssim(image_ct, image_fused[:, 0:1, :, :])
            ssim_loss_2 = 1 - criteria_ssim(image_mr if not args.one_in_rgb else image_mr[:, :1, :, :],
                                            image_fused[:, 0:1, :, :])
            ssim_loss = (ssim_loss_1 + ssim_loss_2) / 2
            if num > 0:
                seg_loss = criteria_seg(it, indice, args, image_fused, seg_model,transform_tools, faiss_module)
            # fusion loss
            loss_fusion, loss_in, loss_grad = criteria_fusion(image_mr if not args.one_in_rgb else image_mr[:,:1,:,:], image_ct, image_fused)
            loss_total = loss_fusion
            loss_total = loss_total + ssim_loss
            if num > 0:
                loss_total = loss_total + num * seg_loss
                # loss_total = loss_total + num * seg_loss
            loss_total.backward()
            optimizer.step()
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            if now_it % 10 == 0:
                loss_seg = 0
                loss_ssim = ssim_loss
                if num > 0:
                    loss_seg = seg_loss
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss_total: {loss_total:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'loss_seg: {loss_seg:.4f}',
                        'loss_ssim: {loss_ssim:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss_total.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    loss_seg=loss_seg,
                    loss_ssim=loss_ssim,
                    time=t_intv,
                    eta=eta,
                )
                logger.info(msg)
                st = ed
        if loss_total.item() < best_fusion_loss:
            torch.save(fusionmodel.state_dict(), args.fusion_model_pth)
            best_fusion_loss = loss_total
    # torch.save(fusionmodel.state_dict(), args.fusion_model_pth)
    logger.info("Fusion Model Save to: {}".format(args.fusion_model_pth))
    logger.info('\n')


def run_fusion():
    fusionmodel = eval('FusionNet')()
    fusionmodel.eval()
    fusionmodel.cuda()
    fusionmodel.load_state_dict(torch.load(args.fusion_model_pth))
    test_dataset = FusionDataset(args)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn(args.seed)
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_mr, images_ct, name, indice) in enumerate(test_loader):
            images_ct = Variable(images_ct).cuda()
            if not args.one_in_rgb:
                images_mr = Variable(images_mr).cuda()
                images_fused = fusionmodel(images_mr, images_ct)
                images_fused = images_fused.expand(-1, 3, -1, -1)
            else:
                images_mr = Variable(images_mr).cuda()
                images_mr_ycrcb = RGB2YCrCb(images_mr)
                images_fused = fusionmodel(images_mr[:, :1], images_ct)
                images_fused = torch.cat(
                    (images_fused, images_mr_ycrcb[:, 1:2, :, :],
                     images_mr_ycrcb[:, 2:, :, :]),
                    dim=1,
                )
                images_fused = YCrCb2RGB(images_fused)
            ones = torch.ones_like(images_fused)
            zeros = torch.zeros_like(images_fused)
            images_fused = torch.where(images_fused > ones, ones, images_fused)
            images_fused = torch.where(images_fused < zeros, zeros, images_fused)
            fused_image = images_fused.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                    np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = image.squeeze()
                image = Image.fromarray(image)
                save_path = os.path.join(args.fused_dir, name[k])
                image.save(save_path)
                # logger.info('Fuse {0} Sucessfully!'.format(save_path))


def train_seg(num=0, logger=None):
    # Start time.
    t_start = time.time()
    # Get model and optimizer.
    model, optimizer, classifier1 = get_model_and_optimizer(args, logger, num)
    # New trainset inside for-loop.
    inv_list, eqv_list = get_transform_params(args)
    seg_trainset = SegDataset(args, inv_list=inv_list, eqv_list=eqv_list)
    seg_trainloader = torch.utils.data.DataLoader(seg_trainset,
                                              batch_size=args.seg_batch_size_cluster,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              collate_fn=collate_sd_train_seg,
                                              worker_init_fn=worker_init_fn(args.seed))
    # Train start.
    best_seg_loss = np.inf
    args.seg_start_epoch = args.seg_num_epoch  # patch_0301
    args.seg_num_epoch += args.seg_per_train_epoch
    for epoch in range(args.seg_start_epoch, args.seg_num_epoch):
        # Assign probs.
        seg_trainloader.dataset.mode = 'compute'
        seg_trainloader.dataset.reshuffle()
        # logger.info('seg train | epo #%s begin...' % epoch)
        # logger.info('Start computing centroids.')
        t1 = time.time()
        centroids1, kmloss1 = run_mini_batch_kmeans(args, logger, seg_trainloader, model, view=1)
        centroids2, kmloss2 = run_mini_batch_kmeans(args, logger, seg_trainloader, model, view=2)
        # logger.info('-Centroids ready. [Loss: {:.5f}| {:.5f}/ Time: {}]'.format(kmloss1, kmloss2, get_datetime(
        #     int(time.time()) - int(t1))))
        # Compute cluster assignment.
        t2 = time.time()
        weight1 = compute_labels(args, logger, seg_trainloader, model, centroids1, view=1)
        weight2 = compute_labels(args, logger, seg_trainloader, model, centroids2, view=2)
        # logger.info('-Cluster labels ready. [{}]'.format(get_datetime(int(time.time()) - int(t2))))
        # Criterion.
        criterion1 = torch.nn.CrossEntropyLoss(weight=weight1).cuda()
        criterion2 = torch.nn.CrossEntropyLoss(weight=weight2).cuda()
        # Setup nonparametric classifier.
        classifier1 = initialize_classifier(args)
        classifier2 = initialize_classifier(args)
        classifier1.module.weight.data = centroids1.unsqueeze(-1).unsqueeze(-1)
        classifier2.module.weight.data = centroids2.unsqueeze(-1).unsqueeze(-1)
        freeze_all(classifier1)
        freeze_all(classifier2)
        # Delete since no longer needed.
        del centroids1
        del centroids2
        # Set-up train loader.
        seg_trainset.mode = 'train'
        seg_trainloader_loop = torch.utils.data.DataLoader(seg_trainset,
                                                       batch_size=args.seg_batch_size_train,
                                                       shuffle=True,
                                                       num_workers=args.num_workers,
                                                       pin_memory=True,
                                                       collate_fn=collate_sd_train_seg,
                                                       worker_init_fn=worker_init_fn(args.seed))
        # logger.info('Start training ...')
        train_loss, train_cet, cet_within, cet_across, train_mse = train_seg_sub(args, logger, seg_trainloader_loop, model,
                                                                         classifier1, classifier2, criterion1,
                                                                         criterion2, optimizer, epoch)
        msg = ', '.join(
            [
                'step: {it}/{max_it}',
                'loss_total: {loss_total:.4f}',
                'loss_kmeans_1: {loss_kmeans_1:.4f}',
                'loss_kmeans_2: {loss_kmeans_2:.4f}',
                'loss_within: {loss_within:.4f}',
                'loss_cross: {loss_cross:.4f}',
                'eta: {eta}',
                'time: {time:.4f}',
            ]
        ).format(
            it=epoch+1,
            max_it=args.seg_num_epoch,
            loss_total=train_loss,
            loss_kmeans_1=kmloss1,
            loss_kmeans_2=kmloss2,
            loss_within=cet_within,
            loss_cross=cet_across,
            time=int(time.time()) - int(t1),
            eta=str(datetime.timedelta(seconds=((int(time.time()) - int(t1)))*(args.seg_num_epoch-1-epoch))),
        )
        logger.info(msg)
        if train_loss < best_seg_loss:
            torch.save({'epoch': epoch + 1,
                        'args': args,
                        'state_dict': model.state_dict(),
                        'classifier1_state_dict': classifier1.state_dict(),
                        'classifier2_state_dict': classifier2.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        },
                       os.path.join(args.seg_model_pth))
            best_seg_loss = train_loss

def train_seg_sub(args, logger, dataloader, model, classifier1, classifier2, criterion1, criterion2, optimizer, epoch):
    losses = AverageMeter()
    losses_mse = AverageMeter()
    losses_cet = AverageMeter()
    losses_cet_across = AverageMeter()
    losses_cet_within = AverageMeter()
    # switch to train mode
    model.train()
    classifier1.eval()
    classifier2.eval()
    for i, (indice, input1, input2, label1, label2) in enumerate(dataloader):
        input1 = eqv_transform_if_needed(args, dataloader, indice, input1.cuda(non_blocking=True))
        label1 = label1.cuda(non_blocking=True)
        featmap1 = model(input1)
        input2 = input2.cuda(non_blocking=True)
        label2 = label2.cuda(non_blocking=True)
        featmap2 = eqv_transform_if_needed(args, dataloader, indice, model(input2))
        B, C, _ = featmap1.size()[:3]
        # if i == 0:
        #     logger.info('Batch input size   : {}'.format(list(input1.shape)))
        #     logger.info('Batch label size   : {}'.format(list(label1.shape)))
        #     logger.info('Batch feature size : {}'.format(list(featmap1.shape)))
        if args.seg_metric == 'cosine':
            featmap1 = F.normalize(featmap1, dim=1, p=2)
            featmap2 = F.normalize(featmap2, dim=1, p=2)
        featmap12_processed, label12_processed = featmap1, label2.flatten()
        featmap21_processed, label21_processed = featmap2, label1.flatten()
        # Cross-view loss
        output12 = feature_flatten(classifier2(featmap12_processed))  # NOTE: classifier2 is coupled with label2
        output21 = feature_flatten(classifier1(featmap21_processed))  # NOTE: classifier1 is coupled with label1
        loss12 = criterion2(output12, label12_processed)
        loss21 = criterion1(output21, label21_processed)
        loss_across = (loss12 + loss21) / 2.
        losses_cet_across.update(loss_across.item(), B)
        featmap11_processed, label11_processed = featmap1, label1.flatten()
        featmap22_processed, label22_processed = featmap2, label2.flatten()
        # Within-view loss
        output11 = feature_flatten(classifier1(featmap11_processed))  # NOTE: classifier1 is coupled with label1
        output22 = feature_flatten(classifier2(featmap22_processed))  # NOTE: classifier2 is coupled with label2
        loss11 = criterion1(output11, label11_processed)
        loss22 = criterion2(output22, label22_processed)
        loss_within = (loss11 + loss22) / 2.
        losses_cet_within.update(loss_within.item(), B)
        loss = (loss_across + loss_within) / 2.
        losses_cet.update(loss.item(), B)
        losses.update(loss.item(), B)
        # compute gradient and do step
        optimizer.zero_grad()
        loss.backward()
        # loss_across.backward()
        optimizer.step()
    return losses.avg, losses_cet.avg, losses_cet_within.avg, losses_cet_across.avg, losses_mse.avg


if __name__ == "__main__":
    args = parse_arguments()
    args.seg_num_epoch = 0
    logger = Logger(args.log_dir).getLog()
    fix_seed_for_reproducability(args.seed)
    # train_fusion(1, logger)
    # run_fusion()
    # train_seg(0, logger)
    for i in range(5):
        train_fusion(i, logger)
        logger.info("train fusion|{0} Finish train Fusion Model".format(i + 1))
        run_fusion()
        logger.info("run fusion|{0} Fuse Image Sucessfully.".format(i + 1))
        if i != 4:
            train_seg(i, logger)
            logger.info("|{0} Finish rain Segmentation Model.".format(i + 1))
    logger.info("Training Done.")
