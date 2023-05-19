"""utils"""
import os
import torch
import shutil

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """utils"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """utils"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, epoch, is_best):
    """utils"""
    # filename = '_'.join([cfg.model.filename, str(epoch)])
    # if not os.path.exists(cfg.model.save_model_path):
    #     os.makedirs(cfg.model.save_model_path)
    #filename = "{output}/{filename}".format(output=cfg.model.save_model_path, filename=filename)
    #torch.save(state, filename)
    if is_best:
        print("save best model, acc:", state['best_prec1'])
        best_name = '_'.join(['best_checkpoint.pth.tar', str(epoch)])
        #best_name = "{output}/{filename}".format(output=cfg.model.save_model_path, filename=best_name)
        #shutil.copyfile(filename, best_name)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


