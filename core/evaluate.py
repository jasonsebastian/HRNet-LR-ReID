import numpy as np

import torch


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def sort_index(qf, ql, qc, gf, gl, gc):
    score = torch.mm(gf, qf.view(-1, 1))
    score = score.squeeze(1).cpu().numpy()
    index = np.argsort(score)[::-1]
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)
    # Only part of body is detected.
    junk_index1 = np.argwhere(gl==-1)
    # The images of the same identity in same cameras
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)
    return index[np.in1d(index, junk_index, invert=True)]
