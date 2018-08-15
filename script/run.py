# the script to extract features
# Input: cropped image path
# Output: features
import sys

sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
from torch.nn.parallel import DataParallel

import time
import numpy as np
from PIL import Image
import cv2
import os

from bpm.model.PCBModel import PCBModel as Model
from bpm.dataset.PreProcessImage import PreProcessIm

from bpm.utils.utils import load_state_dict
from bpm.utils.utils import set_devices

#############
# Arguments #
#############

sys_device_ids = (2,)
TVT, TMO = set_devices(sys_device_ids)

# image input
scale_im = True
im_mean = [0.486, 0.459, 0.408]
im_std = [0.229, 0.224, 0.225]
resize_h_w = (384, 128)

# model
last_conv_stride = 1
last_conv_dilation = 1
num_stripes = 6
local_conv_out_channels = 256

# weight file
model_weight_file = "/mnt/md1/lztao/deecamp/ReID-beyond_part_model/baseline_log/ckpt.pth"



class ExtractFeature(object):
  """A function to be called in the val/test set, to extract features.
  Args:
    TVT: A callable to transfer images to specific device.
  """

  def __init__(self, model, TVT):
    self.model = model
    self.TVT = TVT

  def __call__(self, ims):
    old_train_eval_model = self.model.training
    # Set eval mode.
    # Force all BN layers to use global mean and variance, also disable
    # dropout.
    self.model.eval()

    ims = Variable(self.TVT(torch.from_numpy(ims).float()))
    try:
      local_feat_list, logits_list = self.model(ims)
    except:
      local_feat_list = self.model(ims)
    feat = [lf.data.cpu().numpy() for lf in local_feat_list]
    feat = np.concatenate(feat, axis=1)

    # Restore the model to its old train/eval mode.
    self.model.train(old_train_eval_model)
    return feat


def start():

    ###############
    # preparation #
    ###############

    model = Model(
        last_conv_stride=last_conv_stride,
        num_stripes=num_stripes,
        local_conv_out_channels=local_conv_out_channels
    )
    model_w = DataParallel(model)
    TMO([model])

    # preprocessing
    preprocessor = PreProcessIm(resize_h_w=resize_h_w, scale=scale_im,
                                im_mean=im_mean, im_std=im_std)

    # load model
    map_location = (lambda storage, loc: storage)
    sd = torch.load(model_weight_file, map_location=map_location)
    load_state_dict(model, sd['state_dicts'][0])
    print('Loaded model weight from {}'.format(model_weight_file))

    extractor = ExtractFeature(model_w, TVT)
    return preprocessor, extractor

def run(preprocessor, extractor, im_path):
    im = np.asarray(Image.open(im_path))
    # preprocessing
    im, _ = preprocessor(im)
    im = np.stack([im], axis=0)
    return extractor(im)

def run_video(preprocessor, extractor, video_path, bbox_path):
    # read video
    cap = cv2.VideoCapture(video_path)
    fid, frame = 0, None

    records,count = [], 0

    with open(bbox_path, 'r') as f:
        line = f.readline()
        while line:
            id, x, y, w, h, conf = line.strip().split()
            id, x, y, w, h, conf = int(id), float(x), float(y), float(w), float(h), float(conf)
            x1, y1 = int(np.round(max(0, x))), int(np.round(max(0, y)))
            x2, y2 = int(np.round(max(0, x + w))), int(np.round(max(0, y + h)))

            while fid < id:
                succ, frame = cap.read()
                fid += 1

            im = frame[y1:y2, x1:x2]
            im, _ = preprocessor(im)
            im = np.stack([im], axis=0)

            feat = extractor(im)

            rec = np.zeros([6 + feat.shape[1]])
            rec[0], rec[1], rec[2], rec[3], rec[4], rec[5] = id, x, y, w, h, conf
            rec[6:] = feat[0]

            records.append(rec)

            count += 1
            line = f.readline()
            print(str(count), end='\r')

    records = np.stack(records, axis=0)
    return records

if __name__ == '__main__':
    '''im_dir = '/mnt/md1/lztao/deecamp/ReID-beyond_part_model/eval/people/s1c0/8'
    feat_dir = '/mnt/md1/lztao/deecamp/ReID-beyond_part_model/eval/feats/s1c0/8'

    preprocessor, extractor = start()
    for im_name in os.listdir(im_dir):
        feat = run(preprocessor, extractor, os.path.join(im_dir, im_name))
        np.save(os.path.join(feat_dir, im_name.split('.')[0]), feat)
        print(im_name, end='\r')'''

    #bbox_path = "/mnt/md1/lztao/deecamp/Dataset/epfl/bboxes_fstrcnn/bbox_1_c0.txt"
    bbox_path = "/mnt/md1/lztao/deecamp/Dataset/epfl/bboxes_fstrcnn/terrace1-c0.txt"
    video_path = "/mnt/md1/lztao/deecamp/Dataset/epfl/terrace_seq/terrace1-c0.avi"
    bbox_feat_path = "/mnt/md1/lztao/deecamp/Dataset/epfl/bbox_feats_fstrcnn/bbox_feat_1_c0.npy"

    preprocessor, extractor = start()
    bbox_feat = run_video(preprocessor, extractor, video_path, bbox_path)
    np.save(bbox_feat_path, bbox_feat)
