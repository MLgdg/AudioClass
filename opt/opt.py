import torch 
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR



def opt(params):
	opt = torch.optim.AdamW(params, lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.1)
	scheduler = CosineAnnealingLR(optimizer=opt, T_max=2000, eta_min=1e-6)
	#lr_history = scheduler_lr(opt, scheduler)
	return opt, scheduler

      # optimizer.step() # 更新参数
      # lr_history.append(optimizer.param_groups[0]['lr'])
      # scheduler.step() # 调整学习率
