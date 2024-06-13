import torch
import torch.nn as nn
from torch.nn import functional as F
from skimage.feature import graycomatrix, graycoprops

def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

def kl_loss(inputs, targets, ep=1e-8):
    kl_loss=nn.KLDivLoss(reduction='mean')
    consist_loss = kl_loss(torch.log(inputs+ep), targets)
    return consist_loss

def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs+ep)
    return  torch.mean(-(target[:,0,...]*logprobs[:,0,...]+target[:,1,...]*logprobs[:,1,...]))

def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def dice_loss2(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.

    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.

    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1).to(true.device)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes).to(true.device)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


class TextureFeatureLoss(nn.Module):
    def __init__(self, gray_levels, distance, angles, lambda_weight):
        super(TextureFeatureLoss, self).__init__()
        self.gray_levels = gray_levels
        self.distance = distance
        self.angles = angles
        self.lambda_weight = lambda_weight

    def forward(self, x, y):
        # 将输入图像转换为灰度图像
        x_gray = x
        y_gray = y

        # 计算输入图像和生成图像的纹理特征
        x_features = self.compute_texture_features(x_gray)
        y_features = self.compute_texture_features(y_gray)

        # 计算纹理特征损失
        loss = self.lambda_weight * torch.mean(torch.abs(x_features - y_features))

        return loss

    def compute_texture_features(self, image):
        # 计算图像的纹理特征
        features = []
        for i in range(image.size(0)):
            image_i = image[i, 0, :, :]

            feature_i = []
            for angle in self.angles:
                # 计算GLCM矩阵
                image_i_uint = (image_i * (self.gray_levels - 1)).type(torch.uint8)
                glcm = graycomatrix(image_i_uint.cpu().numpy(), distances=[self.distance], angles=[angle],
                                    levels=self.gray_levels, symmetric=True, normed=True)
                # 提取纹理特征
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]
                entropy = graycoprops(glcm, 'homogeneity')[0, 0]
                feature_i.append([contrast, energy, entropy])

            features.append(feature_i)

        # 将纹理特征转化为张量
        features = torch.tensor(features, dtype=torch.float32)


        return features