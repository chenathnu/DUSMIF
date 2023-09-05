import torch
import torch.nn as nn
import torch.nn.functional as F
from commons import *


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1. - scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


class NormalLoss(nn.Module):
    def __init__(self, ignore_lb=255, *args, **kwargs):
        super(NormalLoss, self).__init__()
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels)
        return torch.mean(loss)


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_a, image_b, image_gen):
        image_gen = image_gen[:, 0, :, :].unsqueeze(1)
        x_in_max = torch.max(image_a, image_b)
        loss_in = F.l1_loss(x_in_max, image_gen)
        a_grad = self.sobelconv(image_a)
        b_grad = self.sobelconv(image_b)
        gen_grad = self.sobelconv(image_gen)
        x_grad_joint = torch.max(a_grad, b_grad)
        loss_grad = F.l1_loss(x_grad_joint, gen_grad)
        loss_total = loss_in + 5*loss_grad
        return loss_total, loss_in, loss_grad


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)

class Segloss(nn.Module):
    def __init__(self):
        super(Segloss, self).__init__()

    def forward(self, it, indice, args, image, model,transform_tools, faiss_module):
        # # image = transform_tools.transform_base(0, image)
        feats_view1 = get_view_img_feat(it, indice, args, 1, image, model, transform_tools)
        feats_view2 = get_view_img_feat(it, indice, args, 2, image, model, transform_tools)
        centroids1, kmloss1 = run_batch_kmeans_for_single(args, feats_view1, faiss_module)
        centroids2, kmloss2 = run_batch_kmeans_for_single(args, feats_view2, faiss_module)
        label1, weight1 = compute_labels_for_single(centroids1, feats_view1)
        label2, weight2 = compute_labels_for_single(centroids2, feats_view2)
        criterion1 = torch.nn.CrossEntropyLoss(weight=weight1).cuda()
        criterion2 = torch.nn.CrossEntropyLoss(weight=weight2).cuda()
        classifier1 = initialize_classifier(args)
        classifier2 = initialize_classifier(args)
        classifier1.module.weight.data = centroids1.unsqueeze(-1).unsqueeze(-1)
        classifier2.module.weight.data = centroids2.unsqueeze(-1).unsqueeze(-1)
        freeze_all(classifier1)
        freeze_all(classifier2)
        del centroids1
        del centroids2
        # criterion_mse = torch.nn.MSELoss().cuda()
        model.eval()
        classifier1.eval()
        classifier2.eval()
        featmap = model(image)
        featmap = torch.nn.functional.normalize(featmap, dim=1, p=2)
        B, C, _ = featmap.size()[:3]
        label1 = label1.flatten()
        label2 = label2.flatten()
        output1 = feature_flatten(classifier1(featmap))
        output2 = feature_flatten(classifier2(featmap))
        loss1 = criterion1(output1, label1)
        loss2 = criterion2(output2, label2)
        loss_cet = (loss1 + loss2) / 2.
        # loss_mse = criterion_mse(torch.Tensor(label1.cpu().numpy()).cuda(), torch.Tensor(label2.cpu().numpy()).cuda())
        # loss = 0.1*(loss_cet + loss_mse) / 2.
        loss = loss_cet
        return 0.5*loss


class NTXentLoss(nn.Module):
    """
    Need to provide the `anchor, positive` patch pairs within the same image.
    Negative sample can be directly inferred by using
    the positive sample from a different anchor in the same image.
    """

    def __init__(self, temperature: float = 0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.epsilon = 1e-7

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor):
        """
        Assuming `anchors` and `positives` to have dimension [B, S, L]
            B: batch size
            S: number of sampled patches per image
            L: latent vector dimension
        """
        assert len(anchors.shape) == 3
        assert anchors.shape == positives.shape

        B, S, _ = anchors.shape

        loss = 0
        # We would like to learn contrastively across patches in the same image.
        # So we will use all sampled patches within the same batch idx to compute the loss.
        for batch_idx in range(B):
            Z_anchors = anchors[batch_idx, ...]
            Z_pos = positives[batch_idx, ...]

            # Create a matrix that represent the [i,j] entries of positive pairs.
            pos_pair_ij = torch.diag(torch.ones(S)).bool()

            Z_anchor = F.normalize(input=Z_anchors, p=2, dim=-1)
            Z_pos = F.normalize(input=Z_pos, p=2, dim=-1)
            sim_matrix = torch.matmul(Z_anchor, Z_pos.T)

            # Entries noted by 1's in `pos_pair_ij` are similarities of positive pairs.
            numerator = torch.sum(
                torch.exp(sim_matrix[pos_pair_ij] / self.temperature))

            # Entries elsewhere are similarities of negative pairs.
            denominator = torch.sum(
                torch.exp(sim_matrix[~pos_pair_ij] / self.temperature))

            loss += -torch.log(numerator /
                               (denominator + self.epsilon) + self.epsilon)

        return loss / B


if __name__ == '__main__':
    pass
