import torch

device = 'cpu'
if torch.cuda.is_available() :
	device = 'cuda'
elif torch.mps.is_available() :
	device = 'mps'