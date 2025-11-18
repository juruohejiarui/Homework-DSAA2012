import torch
import torch.nn.functional as F
from .device import device

d = torch.tensor([
    [0.0, 2.0, 4.0, 5.0],   # distances from class 0
    [3.0, 0.0, 2.0, 3.0],   # from class 1 (tone 2)
    [5.0, 2.0, 0.0, 3.0],   # from class 2 (tone 4) -> notice 2.0 to class 3
    [5.0, 3.0, 2.0, 0.0],   # from class 3 (tone 3)
], device=device)   # tune these numbers: make 2<->3 distance larger


def rank_loss(logits, targets, tone_values = None, lambda_rank=0.5) -> torch.Tensor :
	if tone_values is None :
		tone_values = torch.tensor([3, 2, 1, 0], device=logits.device, dtype=torch.float32)
	
	probs = torch.softmax(logits, dim=-1)
	pred_value = (probs * tone_values).sum(dim=-1)
	target_value = tone_values[targets]
	
	rank_loss = ((pred_value - target_value) ** 2).mean()

	return lambda_rank * rank_loss


def ordinal_ce_loss(logits, targets, alpha=1.0):
	"""
	- logits: (B, 4)
	- targets: (B,)
	- alpha: scaling factor for distance
	"""
	# Compute tone value distance
	dist = d[targets]			# (B,)

	# soft target distribution
	soft_targets = torch.softmax(-alpha * dist, dim=-1)  # (B,4)

	# log prob
	log_probs = torch.log_softmax(logits, dim=-1)

	loss = -(soft_targets * log_probs).sum(dim=-1).mean()
	return loss

def loss_func(logits, targets) -> torch.Tensor :
	return ordinal_ce_loss(logits, targets)