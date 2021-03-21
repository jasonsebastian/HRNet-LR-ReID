import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from apex.fp16_utils import *
from apex import amp, optimizers

from config import config
from config import update_config
from core.function import train
from core.function import validate
from core.losses import MSEFilterLoss
from dataset import ImagePersonDataset
import models
from samplers import RandomIdentitySampler
from utils.modelsummary import get_model_summary
from utils.utils import create_logger
from utils.utils import get_hms_from_sec
from utils.utils import get_optimizer
from utils.utils import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str,
                        required=True)
    args = parser.parse_args()
    update_config(config, args)
    return args

def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    model_func = 'models.{0}.get_{0}'.format(config.MODEL.NAME)
    model = eval(model_func)(config, pretrained=True)

    dump_input = torch.rand((1, 3, config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]))
    print(get_model_summary(model, dump_input))

    gpus = list(config.GPUS)
    device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = get_optimizer(config, model)
    if config.FP16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    criterion = {'reid': nn.CrossEntropyLoss(), 'sr': nn.MSELoss()}
    if config.TRAIN.SR_FILTER:
        criterion['sr'] = MSEFilterLoss()

    best_perf = 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if config.FP16:
                amp.load_state_dict(checkpoint['amp'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
            best_model = True

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    train_dataset = ImagePersonDataset(config, mode='train')
    train_sampler = RandomIdentitySampler(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        num_instances=4
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        sampler=train_sampler,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        ImagePersonDataset(config, mode='val'),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        num_workers=config.WORKERS,
        pin_memory=True
    )

    since = time.time()

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        print('Epoch {}/{}'.format(epoch + 1, config.TRAIN.END_EPOCH))

        train(config, train_loader, model, criterion, optimizer, epoch, device,
              final_output_dir, tb_log_dir, writer_dict)

        perf_indicator = validate(config, valid_loader, model, criterion, device,
                                  final_output_dir, tb_log_dir, writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        checkpoint = {
            'epoch': epoch + 1,
            'model': config.MODEL.NAME,
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict()
        }
        if config.FP16:
            checkpoint['amp'] = amp.state_dict()
        save_checkpoint(checkpoint, best_model, final_output_dir,
                        filename='checkpoint.pth.tar')

    h, m, s = get_hms_from_sec(time.time() - since)
    print('=> total training time: {}h {}m {}s'.format(h, m, s))

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.state_dict(), final_model_state_file)
    writer_dict['writer'].close()

if __name__ == '__main__':
    main()
