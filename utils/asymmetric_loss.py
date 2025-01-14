import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralizedCrossEntropy(nn.Module):
    """Computes the generalized cross-entropy loss, from `
    "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels"
    <https://arxiv.org/abs/1805.07836>`_
    Args:
        q: Box-Cox transformation parameter, :math:`\in (0,1]`
    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        self.q = q
        self.epsilon = 1e-6
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]
        # Avoid undefined gradient for p == 0 by adding epsilon
        p += self.epsilon
        loss = (1 - p ** self.q) / self.q
        return torch.mean(loss)


class PLL_loss(nn.Module):
    def __init__(self, type=None, PartialY=None, device='cuda',
                 gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6, disable_torch_grad_focal_loss=True):
        super(PLL_loss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.losstype = type
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        #PLL items: 
        if type == 'rc':
            self.confidence = self.init_confidence(PartialY)
        if type == 'gce':
            self.q = 0.7

    def init_confidence(self, PartialY):
        tempY = PartialY.sum(dim=1, keepdim=True).repeat(1, PartialY.shape[1])   #repeat train_givenY.shape[1] times in dim 1
        confidence = PartialY.float()/tempY
        confidence = confidence.to(self.device)
        return confidence
    
    def forward(self, *args):
        """"
        x: outputs logits
        y: targets (multi-label binarized vector)
        """
        if self.losstype == 'cc':
            loss = self.forward_cc(*args)
        elif self.losstype == 'ce':
            loss = self.forward_ce(*args)
        elif self.losstype == 'gce':
            loss = self.forward_gce(*args)
        elif self.losstype == 'rc':
            loss = self.forward_rc(*args)
        else:
            raise ValueError
        return loss

    def forward_gce(self, x, y, index):
        """y is shape of (batch_size, num_classes)"""
        p = F.softmax(x, dim=1)      #outputs are logits
        # Create a tensor filled with a very small number to represent 'masked' positions
        masked_p = p.new_full(p.size(), float('-inf'))
        # Apply the mask
        masked_p[y.bool()] = p[y.bool()]
        # Adjust masked positions to avoid undefined gradients by adding epsilon
        masked_p[masked_p == float('-inf')] = self.eps
        loss = (1 - masked_p ** self.q) / self.q
        loss = (loss.sum(dim=1) / y.sum(dim=1)).mean()
        return loss
    
    def forward_cc(self, x, y, index):
        sm_outputs = F.softmax(x, dim=1)      #outputs are logits
        final_outputs = sm_outputs * y
        average_loss = - torch.log(final_outputs.sum(dim=1) / y.sum(dim=1)).mean()     #NOTE: add y.sum(dim=1)
        return average_loss
    
    def forward_ce(self, x, y, index):
        sm_outputs = F.log_softmax(x, dim=1)
        final_outputs = sm_outputs * y
        average_loss = - (final_outputs.sum(dim=1) / y.sum(dim=1)).mean()  #NOTE: add y.sum(dim=1)
        return average_loss
    
    def forward_rc(self, x, y, index):
        logsm_outputs = F.log_softmax(x, dim=1)         #x is the model ouputs
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()    #final_outputs=torch.Size([256, 10]) -> Q: why use negative? A:  
        self.confidence_update(self.confidence, x, y, index)
        return average_loss     

    def confidence_update(self, confidence, batch_outputs, batchY, batch_index):
        with torch.no_grad():
            temp_un_conf = F.softmax(batch_outputs, dim=1)
            confidence[batch_index, :] = temp_un_conf * batchY # un_confidence stores the weight of each example
            #weight[batch_index] = 1.0/confidence[batch_index, :].sum(dim=1)
            base_value = confidence.sum(dim=1).unsqueeze(1).repeat(1, confidence.shape[1])
            self.confidence = confidence/base_value  # use maticx for element-wise division
    
    def forward_old(self, x, y):
        # Calculating Probabilities
        x_softmax = self.softmax(x)     #x.shape=torch.Size([32, 2, 20])
        xs_pos = x_softmax[:, 1, :]
        xs_neg = x_softmax[:, 0, :]
        # y = y.reshape(-1)               #in y, -1 represent what? A: -1 represent the masked(unknown) data
        # xs_pos = xs_pos.reshape(-1)     #shape=torch.Size([640]) (32*20)
        # xs_neg = xs_neg.reshape(-1)

        # xs_pos = xs_pos[y!=-1]  #324``
        # xs_neg = xs_neg[y!=-1]
        # y = y[y!=-1]

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)      #self.clip = c = 0.05

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        # import pdb
        # pdb.set_trace()

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)     #one_sided_gamma=torch.Size([324])
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)    #one_sided_w is used to weight the negative component of the loss.
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()      #loss.shape = torch.Size([324])

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_softmax = self.softmax(x)     #x.shape=torch.Size([32, 2, 20])
        xs_pos = x_softmax[:, 1, :]
        xs_neg = x_softmax[:, 0, :]
        y = y.reshape(-1)               #in y, -1 represent what? A: -1 represent the masked(unknown) data
        xs_pos = xs_pos.reshape(-1)     #shape=torch.Size([640]) (32*20)
        xs_neg = xs_neg.reshape(-1)

        xs_pos = xs_pos[y!=-1]  #324``
        xs_neg = xs_neg[y!=-1]
        y = y[y!=-1]

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)      #self.clip = c = 0.05

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        # import pdb
        # pdb.set_trace()

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)     #one_sided_gamma=torch.Size([324])
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)    #one_sided_w is used to weight the negative component of the loss.
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()      #loss.shape = torch.Size([324])



class AsymmetricLoss2(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss2, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLoss3(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss3, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_neg = x_sigmoid
        xs_pos = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()




class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()
