import argparse
import os
import time
from tqdm import tqdm

import numpy as np
import scipy.io

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

from config import config
from config import update_config
from core.evaluate import sort_index
import models
from utils.utils import unpack


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str,
                        required=True)
    args = parser.parse_args()
    update_config(config, args)
    return args

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(config, model, dataloaders):
    features = torch.FloatTensor()
    for i, data in enumerate(tqdm(dataloaders)):
        img, label = data
        n, c, h, w = img.size()
        ff = torch.FloatTensor(n, 2048).zero_()
        for i in range(2):
            if i == 1:
                img = fliplr(img)
            input_img = Variable(img.cuda())
            if config.MODEL.SR_NAME:
                outputs, _ = model(input_img)
            else:
                outputs = model(input_img)
            ff += outputs.data.cpu()
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features,ff), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

def create_features_file(config, model_state_file, features_path):
    gpus = list(config.GPUS)
    torch.cuda.set_device(gpus[0])

    model_func = 'models.{0}.get_{0}'.format(config.MODEL.NAME)
    model = eval(model_func)(config)

    print('=> loading model from {}'.format(model_state_file))
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_state_file)
    pretrained_up_dict = {k.replace('cbam', 'attn'): v for k, v in pretrained_dict.items()}
    pretrained_up_dict = {k.replace('vdsr', 'sr'): v for k, v in pretrained_up_dict.items()}
    model_dict.update(pretrained_up_dict)
    model.load_state_dict(model_dict)

    model.classifier = nn.Sequential()

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    h, w = config.MODEL.IMAGE_SIZE
    data_transforms = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dir = os.path.join('..', 'data', 'mlr_market1501', 'pytorch')

    image_datasets = {x: datasets.ImageFolder(
        os.path.join(test_dir, x),
        data_transforms
    ) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False, pin_memory=True
    ) for x in ['gallery','query']}

    # Extract feature
    gallery_feature = extract_feature(config, model, dataloaders['gallery'])
    query_feature = extract_feature(config, model, dataloaders['query'])

    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

    # Save to Matlab for check
    result = {'gallery_f': gallery_feature.numpy(),
              'gallery_label': gallery_label,
              'gallery_cam': gallery_cam,
              'query_f': query_feature.numpy(),
              'query_label': query_label,
              'query_cam': query_cam}
    scipy.io.savemat(features_path, result)
    print('=> features file successfully created at {}'.format(features_path))

def compute_mAP(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score)[::-1]  # predict index from large to small

    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:   # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1 / ngood
        precision = (i+1) / (rows_good[i]+1)
        if rows_good[i] != 0:
            old_precision = i / rows_good[i]
        else:
            old_precision = 1
        ap += d_recall * (old_precision+precision)/2

    return ap, cmc

def evaluate(features_path):
    data = unpack(features_path)
    CMC = torch.IntTensor(len(data['gallery_label'])).zero_()
    ap = 0.0
    for i in range(len(data['query_label'])):
        ap_tmp, CMC_tmp = compute_mAP(data['query_feature'][i],
                                      data['query_label'][i],
                                      data['query_cam'][i],
                                      data['gallery_feature'],
                                      data['gallery_label'],
                                      data['gallery_cam'])
        if CMC_tmp[0] == -1:
            continue
        ap, CMC = ap + ap_tmp, CMC + CMC_tmp
    CMC = CMC.float()
    CMC = CMC / len(data['query_label']) #average CMC
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(data['query_label'])))


def main():
    args = parse_args()

    final_output_dir = os.path.join(
        'output', 'market1501', os.path.basename(args.cfg).split('.')[0]
    )

    if config.TEST.MODEL_FILE:
        model_state_file = os.path.join(final_output_dir,
                                        config.TEST.MODEL_FILE)
        filename_w_ext = config.TEST.MODEL_FILE.split('.')[0]
        features_path = os.path.join(final_output_dir, filename_w_ext + '.mat')
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
        features_path = os.path.join(final_output_dir, 'final_state.mat')

    if not os.path.isfile(features_path):
        print('=> creating features file {}'.format(features_path))
        create_features_file(config, model_state_file, features_path)
    else:
        print('=> using features file from {}'.format(features_path))

    evaluate(features_path)

if __name__ == '__main__':
    main()
