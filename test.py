import data
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType
# import pdb
text_list=["A bird", "A bird", "A bird"]
image_paths=["./assets/dog_image.jpg", "./assets/dog_image.jpg", "./assets/dog_image.jpg"]
audio_paths=["./assets/dog_audio.wav", "./assets/car_audio.wav", "/root/paddlejob/workspace/gaoqingdong/train/infer/e2e_infer/feature/audio/9843334641188315117.wav"]
#audio_paths = ["/root/paddlejob/workspace/gaoqingdong/train/infer/e2e_infer/feature/audio/9843334641188315117.wav"]
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)
audio_pre = model.modality_preprocessors["audio"]
audio_trunks = model.modality_trunks['audio']
audio_head = model.modality_heads['audio']
audio_post = model.modality_postprocessors['audio']
# Load data

inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}
#audio_data = data.load_and_transform_audio_data(audio_paths, device)
#print("input data shape", data.load_and_transform_audio_data(audio_paths, device).shape)
with torch.no_grad():
    
    # B, S = audio_data.shape[:2]
    # audio_data = audio_data.reshape(
    #                 B * S, *audio_data.shape[2:]
    #             )
    # modality_value = audio_pre(**{'audio': audio_data})
    # trunk_inputs = modality_value["trunk"]
    # print(trunk_inputs['tokens'].shape)
    # head_inputs = modality_value["head"]
    # modality_value = audio_trunks(**trunk_inputs)
    # print(modality_value.shape)
    # modality_value = audio_head(modality_value, **head_inputs)
    # print(modality_value.shape)
    # modality_value = audio_post(modality_value)
    # print(modality_value.shape)

                # if reduce_list:
                #     modality_value = modality_value.reshape(B, S, -1)
                #     modality_value = modality_value.mean(dim=1)
    embeddings = model(inputs)

#print("output data shape", embeddings[ModalityType.AUDIO].shape)
print(
    "Vision x Text: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
)

print(
    "Audio x Text: ",
    torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Vision x Audio: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
)