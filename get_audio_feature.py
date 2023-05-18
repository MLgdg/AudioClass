import data
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType
# import pdb
import numpy as np
import os
import sys 
import time
name, cuda = sys.argv[1],sys.argv[2]
#paths = {}
#save_path = './audio/'
save_path = '/tmp/v2_mnt/FSC/tags/gaoqingdong/org_data/2023_Q1_train_data_100w_audio_feature/'
# with open(name)as ff:
#     for ll in ff:
#         nid_name = ll.strip()
#         path = '/tmp/v2_mnt/FSC/tags/gaoqingdong/org_data/2023_Q1_train_data_100w_audio/'+nid_name
#         if os.path.exists(path):
#             #print(123)
#             paths[nid_name.split('.')[0]] = [path]
# #print(123)
dataset = data.AudioData(name)
dataload = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=10, collate_fn=data.collate_fn)
device = "cuda:{}".format(cuda)

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)
audio_pre = model.modality_preprocessors["audio"]
audio_trunks = model.modality_trunks['audio']
audio_head = model.modality_heads['audio']
audio_post = model.modality_postprocessors['audio']
num = 0
s1 = time.time()
#print(s1)
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
