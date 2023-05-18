import data
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType
# import pdb
import numpy as np
import os
import sys 
import time
import audiomodel
from opt.opt import opt
from .utils import save_checkpoint, accuracy, AverageMeter
train, test= sys.argv[1],sys.argv[2]


traindata = data.AudioData(train)
trainloader = torch.utils.data.DataLoader(traindata, batch_size=2, shuffle=True, num_workers=10, collate_fn=data.collate_fn)
testdata = data.AudioData(test)
testloader = torch.utils.data.DataLoader(testdata, batch_size=2, shuffle=False, num_workers=10, collate_fn=data.collate_fn)
device = "cuda:0".

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
audio_pre = model.modality_preprocessors["audio"]
audio_trunks = model.modality_trunks['audio']
audio_head = model.modality_heads['audio']
audio_post = model.modality_postprocessors['audio']
model = torch.nn.DataParallel(audiomodel.AudioModel(audio_pre, audio_trunks, audio_head, audio_post).to(device))
optim, scheduler = opt(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()



for e in range(10):
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    s1 = time.time()
    for b, nids, audio_data, labels in enumerate(trainloader):
        s2 = time.time()
        out = model(audio_data.to(device))
        s3 = time.time()
        loss = loos_fc(out, labels.to(device))
        losses.updata(loss, len(nids))
        acc = accuracy(out, labels)
        top1.updata(acc, len(nids))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if print_fn%20==0:
            print()
print(len(dataload))
for nids, audio_datas in dataload:
    if num%1000==0:
        print(num)
    num+=1
    s2 = time.time()
    for i in range(len(nids)):
        audio_data = audio_datas[i].to(device)#data.load_and_transform_audio_data(audio_paths, device)
        nid = nids[i]
        with torch.no_grad():
        
            B, S = audio_data.shape[:2]
            audio_data = audio_data.reshape(
                            B * S, *audio_data.shape[2:]
                    )
            
            s3 = time.time()
            modality_value = audio_pre(**{'audio': audio_data})
            trunk_inputs = modality_value["trunk"]
            head_inputs = modality_value["head"]
            modality_value = audio_trunks(**trunk_inputs)
            s4 =time.time()
            #np.save(os.path.join(save_path + nid + '_all_token'), modality_value.cpu().numpy())
            s5 =time.time()
            #print("data:",s3-s2, "model:",s4-s3,"save:", s5-s4)
            #print(modality_value.shape)
            modality_value = audio_head(modality_value, **head_inputs)
            np.save(os.path.join(save_path + nid + '_head_token' + '.npy'), modality_value.cpu().numpy())
            modality_value = audio_post(modality_value)
            np.save(os.path.join(save_path + nid + '_head_post_token' + '.npy'), modality_value.cpu().numpy())
            #os.system('mv ./audio/{}*')
