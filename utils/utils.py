import datetime
import logging
import math
import os
from pathlib import Path
import time

import scipy.io

import torch
import torch.optim as optim


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.ORIGINAL_DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
                          (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        if cfg.MODEL.SR_NAME:
            base_params = filter(lambda p: id(p) not in list(map(id, model.sr.parameters())),
                                 model.parameters())
            optimizer = optim.SGD(
                [{'params': base_params},
                 {'params': model.sr.parameters(),
                  'lr': cfg.TRAIN.LR * cfg.TRAIN.SR_MUL}],
                lr=cfg.TRAIN.LR,
                weight_decay=cfg.TRAIN.WD,
                momentum=cfg.TRAIN.MOMENTUM,
                nesterov=cfg.TRAIN.NESTEROV)
        else:
            optimizer = optim.SGD(
                model.parameters(),
                lr=cfg.TRAIN.LR,
                weight_decay=cfg.TRAIN.WD,
                momentum=cfg.TRAIN.MOMENTUM,
                nesterov=cfg.TRAIN.NESTEROV)
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        if cfg.MODEL.SR_NAME:
            base_params = filter(lambda p: id(p) not in list(map(id, model.sr.parameters())),
                                 model.parameters())
            optimizer = optim.Adam(
                [{'params': base_params},
                 {'params': model.sr.parameters(),
                  'lr': cfg.TRAIN.LR * cfg.TRAIN.SR_MUL}],
                lr=cfg.TRAIN.LR
            )
        else:
            optimizer = optim.Adam(
                model.sr.parameters(),
                lr=cfg.TRAIN.LR
            )
    return optimizer


def set_learning_rate(cfg, optimizer, epoch):
    begin_lr = [cfg.TRAIN.LR, cfg.TRAIN.LR * cfg.TRAIN.SR_MUL]
    for i, group in enumerate(optimizer.param_groups):
        if epoch < 5:
            start = begin_lr[i] * cfg.TRAIN.WARM_FACTOR
            end = begin_lr[i]
            if cfg.TRAIN.WARM_MODE == 'linear':
                group['lr'] = start + epoch / 5 * (end - start)
            elif cfg.TRAIN.WARM_MODE == 'constant':
                group['lr'] = start
        else:
            group['lr'] = begin_lr[i] * pow(cfg.TRAIN.LR_FACTOR, epoch // cfg.TRAIN.LR_STEP)


def get_hms_from_sec(seconds):
    seconds = round(seconds)
    h, m, s = map(int, str(datetime.timedelta(seconds=seconds)).split(':'))
    return h, m, s


def round_decimal(number):
    return round(number, abs(math.floor(math.log10(number))))


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))


def unpack(features_path):
    result = scipy.io.loadmat(features_path)
    data = {
        'query_feature': torch.FloatTensor(result['query_f']),
        'query_cam': result['query_cam'][0],
        'query_label': result['query_label'][0],
        'gallery_feature': torch.FloatTensor(result['gallery_f']),
        'gallery_cam': result['gallery_cam'][0],
        'gallery_label': result['gallery_label'][0]
    }
    return data
