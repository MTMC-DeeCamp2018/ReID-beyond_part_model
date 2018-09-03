import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .resnet import resnet50


class PCBModel(nn.Module):
  def __init__(
      self,
      last_conv_stride=1,
      last_conv_dilation=1,
      num_stripes=6,
      local_conv_out_channels=256,
      num_classes=0
  ):
    super(PCBModel, self).__init__()

    self.base = resnet50(
      pretrained=True,
      last_conv_stride=last_conv_stride,
      last_conv_dilation=last_conv_dilation)
    self.num_stripes = num_stripes

    self.local_conv_list = nn.ModuleList()
    for _ in range(num_stripes):
      self.local_conv_list.append(nn.Sequential(
        nn.Conv2d(3072, local_conv_out_channels, 1),
        nn.BatchNorm2d(local_conv_out_channels),
        nn.ReLU(inplace=True)
      ))

    if num_classes > 0:
      self.fc_list = nn.ModuleList()
      for _ in range(num_stripes):
        fc = nn.Linear(local_conv_out_channels, num_classes)
        init.normal(fc.weight, std=0.001)
        init.constant(fc.bias, 0)
        self.fc_list.append(fc)


  '''def forward(self, x):
    """
    Returns:
      local_feat_list: each member with shape [N, c]
      logits_list: each member with shape [N, num_classes]
    """
    # shape [N, C, H, W]
    feat, _ = self.base(x)
    assert feat.size(2) % self.num_stripes == 0
    stripe_h = int(feat.size(2) / self.num_stripes)
    local_feat_list = []
    output_feat_list = []
    logits_list = []
    for i in range(self.num_stripes):
      # shape [N, C, 1, 1]
      local_feat = F.avg_pool2d(
        feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],
        (stripe_h, feat.size(-1)))
      output_feat_list.append(local_feat[:,:,0,0])
      # shape [N, c, 1, 1]
      local_feat = self.local_conv_list[i](local_feat)
      # shape [N, c]
      local_feat = local_feat.view(local_feat.size(0), -1)
      local_feat_list.append(local_feat)
      if hasattr(self, 'fc_list'):
        logits_list.append(self.fc_list[i](local_feat))

    if hasattr(self, 'fc_list'):
      return output_feat_list, logits_list

    return output_feat_list'''


  def forward(self, x):
    """
    Returns:
      local_feat_list: each member with shape [N, c]
      logits_list: each member with shape [N, num_classes]
    """
#    print('Please recover forward function in PCBModel!')
    # shape [N, C, H, W]
    feat1, multiscale_feats = self.base(x)
    feat2 = multiscale_feats[3]
#    print(len(multiscale_feats), multiscale_feats[0].shape, multiscale_feats[1].shape, multiscale_feats[2].shape, multiscale_feats[3].shape)
    assert feat1.size(2) % self.num_stripes == 0 and feat2.size(2) % self.num_stripes == 0
    stripe_h1 = int(feat1.size(2) / self.num_stripes)
    stripe_h2 = int(feat2.size(2) / self.num_stripes)
    local_feat_list = []
    output_feat_list = []
    logits_list = []
    for i in range(self.num_stripes):
      # shape [N, C, 1, 1]
      local_feat1 = F.avg_pool2d(
        feat1[:, :, i * stripe_h1: (i + 1) * stripe_h1, :],
        (stripe_h1, feat1.size(-1)))
      local_feat2 = F.avg_pool2d(
        feat2[:,:,i*stripe_h2:(i+1)*stripe_h2, :],
        (stripe_h2, feat2.size(-1))
      )
      local_feat = torch.cat([local_feat1, local_feat2], dim=1)
      output_feat_list.append(local_feat[:,:,0,0])
      # shape [N, c, 1, 1]
      local_feat = self.local_conv_list[i](local_feat)
      # shape [N, c]
      local_feat = local_feat.view(local_feat.size(0), -1)
      local_feat_list.append(local_feat)
      if hasattr(self, 'fc_list'):
        logits_list.append(self.fc_list[i](local_feat))

    if hasattr(self, 'fc_list'):
      return local_feat_list, logits_list

    '''for multiscale_feat in multiscale_feats[3:]:
        shape = multiscale_feat.shape
#        print(shape)
        output_feat_list.append(torch.mean(multiscale_feat.view(shape[0], shape[1], -1), dim=2))
#        print(output_feat_list[-1].shape)
    #input()'''
    return local_feat_list
