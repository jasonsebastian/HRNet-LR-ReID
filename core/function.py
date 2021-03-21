import logging
import math
import time
from tqdm import tqdm

import torch
import torch.nn as nn

from apex.fp16_utils import *
from apex import amp, optimizers

from core.evaluate import accuracy
from utils.utils import get_hms_from_sec, round_decimal, set_learning_rate


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch, device,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if config.MODEL.SR_NAME:
        losses_sr = AverageMeter()
    losses_reid = AverageMeter()
    top1 = AverageMeter()

    set_learning_rate(config, optimizer, epoch)

    learning_rates = [round_decimal(group['lr']) for group in optimizer.param_groups]
    logger.info('=> lr: {}'.format(learning_rates))

    model.train()

    end = time.time()
    with tqdm(train_loader) as t:
        for i, data in enumerate(t):
            data_time.update(time.time() - end)

            inputs_res = data['downsampled_image'].to(device)
            inputs = data['original_image'].to(device)
            labels = data['label'].to(device)

            n = inputs.size(0)

            optimizer.zero_grad()
            if config.MODEL.SR_NAME:
                y, sr = model(inputs_res)
            else:
                y = model(inputs_res)
            _, preds = torch.max(y.data, 1)

            if config.MODEL.SR_NAME:
                if config.TRAIN.SR_FILTER:
                    loss_sr = criterion['sr'](inputs, sr,
                                              config.TRAIN.FILTER_VAL,
                                              device)
                else:
                    loss_sr = criterion['sr'](inputs, sr)
            loss_reid = criterion['reid'](y, labels)
            if config.MODEL.SR_NAME:
                loss = config.TRAIN.ALPHA * loss_sr + loss_reid
            else:
                loss = loss_reid

            if config.FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Gradient clipping enables higher lr for VDSR
            if config.MODEL.SR_NAME:
                nn.utils.clip_grad_norm_(model.sr.parameters(), max_norm=0.4)

            optimizer.step()

            if config.MODEL.SR_NAME:
                losses_sr.update(loss_sr.item(), n)
            losses_reid.update(loss_reid.item(), n)
            losses.update(loss.item(), n)
            prec1 = accuracy(y, labels, (1,))
            top1.update(prec1[0].item(), n)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                if writer_dict:
                    writer = writer_dict['writer']
                    global_steps = writer_dict['train_global_steps']
                    writer.add_scalar('train_loss', losses.val, global_steps)
                    writer.add_scalar('train_top1', top1.val, global_steps)
                    writer_dict['train_global_steps'] = global_steps + 1

            if config.MODEL.SR_NAME:
                t.set_postfix(loss_sr=losses_sr.avg, loss_reid=losses_reid.avg,
                              loss=losses.avg, accuracy=top1.avg)
            else:
                t.set_postfix(loss=losses.avg, accuracy=top1.avg)


def validate(config, val_loader, model, criterion, device,
             output_dir, tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    if config.MODEL.SR_NAME:
        losses_sr = AverageMeter()
    losses_reid = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            inputs_res = data['downsampled_image'].to(device)
            inputs = data['original_image'].to(device)
            labels = data['label'].to(device)

            n = inputs.size(0)

            if config.MODEL.SR_NAME:
                y, sr = model(inputs_res)
            else:
                y = model(inputs_res)
            _, preds = torch.max(y.data, 1)

            if config.MODEL.SR_NAME:
                if config.TRAIN.SR_FILTER:
                    loss_sr = criterion['sr'](inputs, sr,
                                              config.TRAIN.FILTER_VAL,
                                              device)
                else:
                    loss_sr = criterion['sr'](inputs, sr)
            loss_reid = criterion['reid'](y, labels)
            if config.MODEL.SR_NAME:
                loss = config.TRAIN.ALPHA * loss_sr + loss_reid
            else:
                loss = loss_reid

            if config.MODEL.SR_NAME:
                losses_sr.update(loss_sr.item(), n)
            losses_reid.update(loss_reid.item(), n)
            losses.update(loss.item(), n)
            prec1 = accuracy(y, labels, (1,))
            top1.update(prec1[0].item(), n)

            batch_time.update(time.time() - end)
            end = time.time()

        if config.MODEL.SR_NAME:
            msg = 'Test: Time {batch_time.avg:.3f}\t' \
                  'Loss {loss.avg:.4f}\t' \
                  'SR loss {loss_sr.avg:.4f}\t' \
                  'Reid loss {loss_reid.avg:.4f}\t' \
                  'Error@1 {error1:.3f}\t' \
                  'Accuracy@1 {top1.avg:.3f}\t'.format(
                      batch_time=batch_time, loss=losses,
                      loss_sr=losses_sr, loss_reid=losses_reid,
                      top1=top1, error1=100-top1.avg)
        else:
            msg = 'Test: Time {batch_time.avg:.3f}\t' \
                  'Loss {loss.avg:.4f}\t' \
                  'Error@1 {error1:.3f}\t' \
                  'Accuracy@1 {top1.avg:.3f}\t'.format(
                      batch_time=batch_time, loss=losses,
                      top1=top1, error1=100-top1.avg)

        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_top1', top1.avg, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
