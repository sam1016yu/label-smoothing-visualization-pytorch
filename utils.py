'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from pathlib import Path

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


class SimLabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(SimLabelSmoothingCrossEntropy, self).__init__()   
        self.sim_mat = [
                [1.00, 0.15, 0.25, 0.01, 0.02, 0.03, 0.01, 0.12, 0.29, 0.11],
                [0.15, 1.00, 0.01, 0.10, 0.30, 0.10, 0.10, 0.10, 0.07, 0.90],
                [0.25, 0.01, 1.00, 0.30, 0.45, 0.25, 0.30, 0.18, 0.01, 0.00],
                [0.01, 0.10, 0.30, 1.00, 0.10, 0.80, 0.10, 0.08, 0.00, 0.05],
                [0.02, 0.30, 0.45, 0.10, 1.00, 0.52, 0.05, 0.90, 0.01, 0.15],
                [0.03, 0.10, 0.25, 0.80, 0.52, 1.00, 0.20, 0.36, 0.05, 0.05],
                [0.01, 0.10, 0.30, 0.10, 0.05, 0.20, 1.00, 0.10, 0.10, 0.05],
                [0.12, 0.10, 0.18, 0.08, 0.90, 0.36, 0.10, 1.00, 0.08, 0.25],
                [0.29, 0.07, 0.01, 0.00, 0.01, 0.05, 0.10, 0.08, 1.00, 0.20],
                [0.11, 0.90, 0.00, 0.05, 0.15, 0.05, 0.05, 0.25, 0.20, 1.00]
            ]
    
    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        b_size = x.shape[0]
        d =  torch.tensor([[0]*10 for _ in range(b_size)])
        for b in range(b_size):
            for k in range(10): 
                d[b][k] = self.sim_mat[k][target[b]]
        
        d = F.normalize(d,1)
        d = d.to("cuda:0")
        nll_loss =  -torch.sum(torch.mul(logprobs,d),dim=1,keepdim=True)
        nll_loss = nll_loss.squeeze(1)
        return nll_loss.mean()

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def save_model(model, model_path):
    if isinstance(model_path, Path):
        model_path = str(model_path)
    if isinstance(model, nn.DataParallel):
        model = model.module
    state_dict = model.state_dict()
    for key in state_dict:
        state_dict[key] = state_dict[key].cpu()
    torch.save(state_dict, model_path)
