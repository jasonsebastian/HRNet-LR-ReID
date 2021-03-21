import logging
import os
import re

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from core.transforms import *


class ImagePersonDataset(Dataset):
    """Image Person ReID dataset."""

    def __init__(self, config, mode='train'):
        assert mode in ['train', 'val']
        self.config = config
        self.mode = mode
        self.images_person = self.get_images_person()

        h, w = self.config.MODEL.IMAGE_SIZE
        transform_list = [Resize((h, w))]
        if mode == 'train':
            transform_list.extend([RandomHorizontalFlip(),
                                   RandomCrop((h, w), padding=10)])
        transform_list.extend([ToTensor(),
                               Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
        if mode == 'train' and config.TRAIN.RANDOM_ERASING:
            transform_list.append(RandomErasing(probability=0.5,
                                                mean=[0, 0, 0]))
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.images_person)

    def __getitem__(self, idx):
        img_name, label = self.images_person[idx]
        original_img_path = os.path.join(self.config.DATASET.ORIGINAL_ROOT,
                                         'bounding_box_train',
                                         img_name)
        downsampled_img_path = os.path.join(self.config.DATASET.DOWNSAMPLED_ROOT,
                                            'bounding_box_train',
                                            img_name)
        original_image = Image.open(original_img_path).convert('RGB')
        downsampled_image = Image.open(downsampled_img_path).convert('RGB')
        sample = {'original_image': original_image,
                  'downsampled_image': downsampled_image,
                  'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_images_person(self):
        print('=> loading {} dataset image names to memory'.format(self.mode))
        if self.mode == 'train':
            img_dir = os.path.join(self.config.DATASET.DOWNSAMPLED_ROOT,
                                   self.config.DATASET.TRAIN_SET)
        else:
            img_dir = os.path.join(self.config.DATASET.DOWNSAMPLED_ROOT,
                                   self.config.DATASET.TEST_SET)

        pid_container = os.listdir(img_dir)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        images_person = []
        for pid in pid_container:
            label = pid2label[pid]
            img_filenames = os.listdir(os.path.join(img_dir, pid))
            for img_name in img_filenames:
                images_person.append((img_name, label))
        print('=> loaded {} images'.format(len(images_person)))
        return images_person
