import data
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim=1024, hidden_dim=4096):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class AudioModel(torch.nn.Module):
	def __init__(self, pre, trans, head, post):
		super(Audiomodel, self).__init__()

		self.pre = pre
		self.trans = trans
		self.head = head
		self.post = post
		self.fc1 = FeedForward()
		self.fc2 = torch.nn.Linear(1024, 325, False)
	def forward(self, audio_data):
		B, S = audio_data.shape[:2]
		audio_data = audio_data.reshape(
		        B * S, *audio_data.shape[2:]
		)

		modality_value = self.pre(**{'audio': audio_data})
		trunk_inputs = modality_value["trunk"]
		head_inputs = modality_value["head"]
		modality_value = self.trans(**trunk_inputs)

		modality_value = self.head(modality_value, **head_inputs)
		modality_value = modality_value.reshape(B, S, -1)
		modality_value = modality_value.mean(dim=1)
		modality_value = self.fc1(modality_value)
		modality_value = self.fc2(modality_value)
		return modality_value