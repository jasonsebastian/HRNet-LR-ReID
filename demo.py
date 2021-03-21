import argparse
import random
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.io

import torch
from torchvision import datasets, transforms

from core.evaluate import sort_index
import models
from utils.utils import unpack


def imshow(ax, img, title=None):
    ax.imshow(img)
    ax.axis('off')
    if title is not None:
        ax.set_title(title)

def get_image_path(dataset, i):
    return dataset.imgs[i][0]

def unnormalize(tensor, mean, std):
    for i in range(tensor.size(0)):
        tensor[i] = tensor[i] * std[i] + mean[i]
    return tensor

def parse_args():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--cfg',
                        type=str,
                        required=True)
    parser.add_argument('--n',
                        type=int,
                        help='No of queries',
                        default=3)
    parser.add_argument('--k',
                        type=int,
                        help='Top k results',
                        default=10)
    return parser.parse_args()

def main():
    args = parse_args()

    cfg_name = os.path.basename(args.cfg).split('.')[0]
    final_output_dir = os.path.join('output', 'market1501', cfg_name)

    features_path = os.path.join(final_output_dir, 'final_state.mat')
    print('Using features file from {}'.format(features_path))
    data = unpack(features_path)

    data_dir = os.path.join('..', 'data', 'mlr_market1501', 'pytorch')
    gallery_dir = os.path.join(data_dir, 'gallery')
    query_dir = os.path.join(data_dir, 'query')
    gallery_dataset = datasets.ImageFolder(gallery_dir)
    query_dataset = datasets.ImageFolder(query_dir)

    model = eval('models.vdsr.get_vdsr')()
    model.eval()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(os.path.join(final_output_dir, 'final_state.pth.tar'),
                                 map_location='cuda:0')
    pretrained_dict = {k[len('vdsr.'):]: v for k, v in pretrained_dict.items()
                       if 'vdsr' in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    while True:
        fig, ax = plt.subplots(args.n, args.k+2, figsize=(36, 24))
        for i in range(args.n):
            q = random.randint(0, len(data['query_label'])-1)
            query_path = get_image_path(query_dataset, q)
            print('Query path: {}'.format(query_path))

            img = Image.open(query_path).convert('RGB')
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            t = transforms.Compose([
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            img_transformed = torch.unsqueeze(t(img), 0) # Simulate batch of size 1
            sr = model(img_transformed)
            sr = torch.squeeze(sr, 0).detach()

            sr = unnormalize(sr, mean, std)

            imshow(ax[i][0], img, 'Query')
            # imshow(ax[i][1], img.resize((128, 256)), 'Bicubic')
            imshow(ax[i][1], sr.numpy().transpose((1, 2, 0)), 'VDSR')

            index = sort_index(data['query_feature'][q],
                               data['query_label'][q],
                               data['query_cam'][q],
                               data['gallery_feature'],
                               data['gallery_label'],
                               data['gallery_cam'])

            print('Top {} images are as follow:'.format(args.k))
            for j in range(args.k):
                img_path = get_image_path(gallery_dataset, index[j])
                print(img_path)
                imshow(ax[i][j+2], plt.imread(img_path))
                label = data['gallery_label'][index[j]]
                if label == data['query_label'][q]:
                    ax[i][j+2].set_title(str(j+1), color='green')
                else:
                    ax[i][j+2].set_title(str(j+1), color='red')
        plt.show()

if __name__ == '__main__':
    main()