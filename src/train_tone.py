# training of evaluation models
import data
import models
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tensorboardX import SummaryWriter

learning_rate = 1e-5
epoch_num = 100
split_rate = 0.85
split_item = False # whether to split dataset by items or not
random_seed = data.RANDOM_SEED

def get_valid(preds : torch.Tensor, targets : torch.Tensor, mask : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] :
	# preds : (batch_size, seq_len, tone_vocab_size)
	# targets : (batch_size, seq_len)
	# mask : (batch_size, seq_len)
	# print(preds.shape)
	batch_size, seq_len, tone_size = preds.shape
	mask_flat = mask.view(-1)               # (batch_size * seq_len)
	preds_flat = preds.view(-1, tone_size)  # (batch_size * seq_len)
	targets_flat = targets.view(-1)         # (batch_size * seq_len)

	preds_flat = preds_flat[mask_flat == 1]     # (num_valid, tone_size)
	targets_flat = targets_flat[mask_flat == 1] # (num_valid,)

	return preds_flat, targets_flat

def get_metrics(preds : torch.Tensor, targets : torch.Tensor) -> dict :
	# preds : (num_valid, tone_size)
	# targets : (num_valid,)

	preds = torch.cat(preds, dim=0)
	targets = torch.cat(targets, dim=0)

	pred_labels = preds.argmax(dim=-1).cpu().numpy()
	true_labels = targets.cpu().numpy()

	accuracy = accuracy_score(true_labels, pred_labels)
	precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted', zero_division=0)

	f1_each = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
	prec_each : list[float] = [f1_each[str(i)]['precision'] for i in range(4)]
	recall_each : list[float] = [f1_each[str(i)]['recall'] for i in range(4)]
	f1_each : list[float] = [f1_each[str(i)]['f1-score'] for i in range(4)]

	return {
		'accuracy' : accuracy,
		'precision' : precision,
		'recall' : recall,
		'f1' : f1,
		'prec-each' : prec_each,
		'recall-each': recall_each,
		'f1-each': f1_each
	}

def print_metrics(metrics : dict) -> str :
	output = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if not isinstance(v, list)])
	lst_items : list[tuple[str, str]] = []
	for k, v in metrics.items() :
		if not isinstance(v, list) : continue
		lst_items.append((k, ", ".join(["{:.4f}".format(x) for x in v])))
	
	output += ", " + ", ".join([f"{k}: [{v}]" for k, v in lst_items])
	return output

def train_epoch(model : nn.Module, dataLoader : torch.utils.data.DataLoader, criterion, optimizer, scheduler = None) -> dict :
	model.train()

	tot_loss = 0.0
	all_preds = []
	all_targets = []

	for prev_durr, prev_pitc, prev_tone, curr_durr, curr_pitc, curr_tone, prev_mask, curr_mask in dataLoader :
		prev_durr = prev_durr.to(models.device)
		prev_pitc = prev_pitc.to(models.device)
		prev_tone = prev_tone.to(models.device)

		curr_durr = curr_durr.to(models.device)
		curr_pitc = curr_pitc.to(models.device)
		curr_tone = curr_tone.to(models.device)
		
		prev_mask = prev_mask.to(models.device)
		curr_mask = curr_mask.to(models.device)
		# model = model.cpu()
		
		# model = model.cpu()
		optimizer.zero_grad()
		res = model(
			prev_durr, prev_pitc, prev_tone, 
			curr_durr, curr_pitc, 
			prev_mask, curr_mask,
			targets = curr_tone)   # (batch_size, seq_len, tone_vocab_size)
		# print(loss, logits)
		if isinstance(res, tuple) :
			loss, logits = res
			preds_flat, targets_flat = get_valid(logits, curr_tone, curr_mask)
		else :
			logits = res
			preds_flat, targets_flat = get_valid(logits, curr_tone, curr_mask)
			loss = models.loss_func(preds_flat, targets_flat)
		loss.backward(torch.ones_like(loss))
		optimizer.step()

		preds_flat, targets_flat = get_valid(logits, curr_tone, curr_mask)

		tot_loss += loss.sum().item()

		all_preds.append(preds_flat.detach().cpu())
		all_targets.append(targets_flat.detach().cpu())

	if scheduler is not None :
		scheduler.step()

	epoch_loss = tot_loss / len(dataLoader)
	metrics = get_metrics(all_preds, all_targets)
	metrics['loss'] = epoch_loss
	return metrics

def eval_epoch(model : nn.Module, dataLoader : torch.utils.data.DataLoader, criterion) -> dict :
	model.eval()
	tot_loss = 0.0
	all_preds = []
	all_targets = []
	with torch.no_grad() :
		for prev_durr, prev_pitc, prev_tone, curr_durr, curr_pitc, curr_tone, prev_mask, curr_mask in dataLoader :
			prev_durr = prev_durr.to(models.device)
			prev_pitc = prev_pitc.to(models.device)
			prev_tone = prev_tone.to(models.device)

			curr_durr = curr_durr.to(models.device)
			curr_pitc = curr_pitc.to(models.device)
			curr_tone = curr_tone.to(models.device)
			
			prev_mask = prev_mask.to(models.device)
			curr_mask = curr_mask.to(models.device)

			res = model(
				prev_durr, prev_pitc, prev_tone, 
				curr_durr, curr_pitc, 
				prev_mask, curr_mask, 
				targets = curr_tone)   # (batch_size, seq_len, tone_vocab_size)

			if isinstance(res, tuple) :
				loss, logits = res
				preds_flat, targets_flat = get_valid(logits, curr_tone, curr_mask)
			else :
				logits = res
				preds_flat, targets_flat = get_valid(logits, curr_tone, curr_mask)
				loss = models.loss_func(preds_flat, targets_flat)
			tot_loss += loss.sum().item()

			all_preds.append(preds_flat.detach().cpu())
			all_targets.append(targets_flat.detach().cpu())
	
	epoch_loss = tot_loss / len(dataLoader)
	metrics = get_metrics(all_preds, all_targets)
	metrics['loss'] = epoch_loss
	return metrics

def eval_confusion_matrix(model : nn.Module, dataLoader : torch.utils.data.DataLoader) -> np.ndarray :
	model.eval()
	all_preds = []
	all_targets = []
	with torch.no_grad() :
		for prev_durr, prev_pitc, prev_tone, curr_durr, curr_pitc, curr_tone, prev_mask, curr_mask in dataLoader :
			prev_durr = prev_durr.to(models.device)
			prev_pitc = prev_pitc.to(models.device)
			prev_tone = prev_tone.to(models.device)

			curr_durr = curr_durr.to(models.device)
			curr_pitc = curr_pitc.to(models.device)
			curr_tone = curr_tone.to(models.device)
			
			prev_mask = prev_mask.to(models.device)
			curr_mask = curr_mask.to(models.device)

			logits = model(prev_durr, prev_pitc, prev_tone, curr_durr, curr_pitc, prev_mask, curr_mask)   # (batch_size, seq_len, tone_vocab_size)

			preds_flat, targets_flat = get_valid(logits, curr_tone, curr_mask)

			all_preds.append(preds_flat.detach().cpu())
			all_targets.append(targets_flat.detach().cpu())

	all_preds_tensor = torch.cat(all_preds, dim=0)
	all_targets_tensor = torch.cat(all_targets, dim=0)
	pred_labels = all_preds_tensor.argmax(dim=1).cpu().numpy()
	true_labels = all_targets_tensor.cpu().numpy()
	conf_matrix = confusion_matrix(true_labels, pred_labels)
	return conf_matrix

def write_log(name : str, metrics : dict, epoch : int, logger : SummaryWriter) :
	for k, v in metrics.items() :
		if isinstance(v, list) :
			for i, val in enumerate(v) :
				logger.add_scalar(f"{name}/{k}_{i}", val, epoch)
		else :
			logger.add_scalar(f"{name}/{k}", v, epoch)

# split data into train and val sets and try to make ratio of train set and original set close
def split_data(dataset : data.ToneDataset, train_size : int) -> torch.utils.data.Dataset :
	bst_split_indices = None
	ratio_orig = data.get_ratio(dataset)
	bst_diff = float('inf')

	random.seed(random_seed)
	for _ in range(10) :
		indices = list(range(len(dataset)))
		random.shuffle(indices)
		train_indices = indices[:train_size]
		train_subset = torch.utils.data.Subset(dataset, train_indices)
		ratio_train = torch.zeros(4, dtype=torch.float)
		for i in range(len(train_subset)) :
			x : data.DataItem = train_subset[i]
			for j in range(4) :
				ratio_train[j] += (x.curr_tone == j).sum().item()
		ratio_train /= ratio_train.sum()
		diff = torch.abs(ratio_train - ratio_orig).sum()
		if diff < bst_diff :
			bst_diff = diff
			bst_ratio_train = ratio_train
			bst_split_indices = indices
	print(f"ratio_orig: {ratio_orig}, bst_ratio_train: {bst_ratio_train}")
	train_indices = bst_split_indices[:train_size]
	val_indices = bst_split_indices[train_size:]
	train_subset = torch.utils.data.Subset(dataset, train_indices)
	val_subset = torch.utils.data.Subset(dataset, val_indices)	
	
	return train_subset, val_subset
	

if __name__ == "__main__" :
	ratio = torch.zeros(4, dtype=torch.float)
	if split_item :
		dataset = data.ToneDataset(data_dir="dataset/Cantopop-corpus/Humdrum-files", split=1, train=True)
		train_size = int(split_rate * len(dataset))
		val_size = len(dataset) - train_size
		train_dataset, val_dataset = split_data(dataset, train_size)
		ratio = data.get_ratio(train_dataset)
	else :
		train_dataset = data.ToneDataset(data_dir="dataset/Cantopop-corpus/Humdrum-files", split=split_rate, train=True)
		val_dataset = data.ToneDataset(data_dir="dataset/Cantopop-corpus/Humdrum-files", split=split_rate, train=False)
		train_size = len(train_dataset)
		val_size = len(val_dataset)
		ratio = data.get_ratio(train_dataset)
	print(f"Train size: {train_size}, Val size: {val_size}")

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=data.tone_collate_fn)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=data.tone_collate_fn)

	model = models.ToneModel(tone_vocab_size=data.TONE_VOCAB_SIZE).to(models.device)
	# convert ratio to alpha of focal loss
	alpha = (1 / (1 + torch.log(torch.tensor(ratio / ratio.min())))) ** 2
	alpha /= alpha.sum()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
	sceduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250, eta_min=learning_rate / 100)
	# sceduler = None

	bst_model, bst_metric = None, None
	
	model_type = type(model).__name__
	model_name = f"tone_{str(model_type)}_{random_seed}_{learning_rate}_{split_rate}"

	logger = SummaryWriter(logdir=f"logs/{model_name}", flush_secs=10)

	for epoch in tqdm(range(epoch_num)) :
		train_metrics = train_epoch(model, train_loader, criterion, optimizer, sceduler)
		val_metrics = eval_epoch(model, val_loader, criterion)
		need_log = epoch == epoch_num - 1
		if epoch >= epoch_num / 10 and (bst_metric is None or val_metrics['f1'] > bst_metric['f1']) :
			bst_metric = val_metrics
			bst_model = model.state_dict()
			torch.save(bst_model, f"ckpts/{model_name}_best.pth")			
			need_log = True
		write_log("Train", train_metrics, epoch, logger)
		write_log("Val", val_metrics, epoch, logger)
		if need_log :
			tqdm.write(f"Epoch {epoch + 1}/{epoch_num}")
			tqdm.write(f"Train : {print_metrics(train_metrics)}")
			tqdm.write(f"Val   : {print_metrics(val_metrics)}")

	tqdm.write(f"Best Val Metric: {print_metrics(bst_metric)}")

	confuse_mat = eval_confusion_matrix(model, val_loader)
	tqdm.write("Confusion Matrix:")
	tqdm.write(str(confuse_mat))

	# save the final model
	torch.save(model.state_dict(), f"ckpts/{model_name}_final.pth")