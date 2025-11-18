import torch
import torch.nn as nn
from .parse import DataItem, parse_krn, TONE_VOCAB_SIZE
import os
from tqdm import tqdm
import numpy as np
import random

all_items = None

RANDOM_SEED = 42

def get_ratio(dataset : torch.utils.data.Dataset) -> np.ndarray :
	cnt = np.zeros((TONE_VOCAB_SIZE,), dtype=np.int64)
	for item in dataset :
		for i in range(TONE_VOCAB_SIZE) :
			cnt[i] += np.sum(item.curr_tone == i).item()

	return cnt / np.sum(cnt)

class ToneDataset(torch.utils.data.Dataset) :
	def __init__(self, data_dir : str, split : float = 1.0, train : bool = True) :
		super(ToneDataset, self).__init__()
		global all_items
		if all_items is None :
			all_items = os.listdir(data_dir)
			random.seed(RANDOM_SEED)
			random.shuffle(all_items)
		split_index = int(len(all_items) * split)
		if train :
			data_items = all_items[:split_index]
		else :
			data_items = all_items[split_index :]

		self.data_items : list[DataItem] = []
		for item in tqdm(data_items) :
			if item.endswith(".krn") :
				self.data_items.extend(parse_krn(os.path.join(data_dir, item)))
		print(f"Loaded {len(self.data_items)} data items from {data_dir}")

		print(f"Tone ratio (0,2,4,3): {get_ratio(self)}")

	def __len__(self) :
		return len(self.data_items)

	def __getitem__(self, idx) :
		return self.data_items[idx]

def collate_fn(batch : list[DataItem]) :
	batch_size = len(batch)
	batch = sorted(batch, key=lambda x: x.curr_durr.shape[0], reverse=True)
	max_prev_len = max([item.prev_durr.shape[0] for item in batch])
	max_curr_len = max([item.curr_durr.shape[0] for item in batch])

	def toTensorDtype(np_array : np.ndarray) -> torch.dtype :
		if np_array.dtype == np.float32 or np_array.dtype == np.float64 :
			return torch.float32
		elif np_array.dtype == np.int64 or np_array.dtype == np.int32 :
			return torch.int32
		else :
			raise ValueError("Unsupported numpy dtype:", np_array.dtype)
	prev_durr = torch.zeros((batch_size, max_prev_len), dtype=toTensorDtype(batch[0].prev_durr))
	prev_pitc = torch.zeros((batch_size, max_prev_len, batch[0].prev_pitc.shape[1]), dtype=toTensorDtype(batch[0].prev_pitc))
	prev_tones = torch.zeros((batch_size, max_prev_len), dtype=torch.int64)
	curr_durr = torch.zeros((batch_size, max_curr_len), dtype=toTensorDtype(batch[0].curr_durr))
	curr_pitc = torch.zeros((batch_size, max_curr_len, batch[0].prev_pitc.shape[1]), dtype=toTensorDtype(batch[0].curr_pitc))
	curr_tones = torch.zeros((batch_size, max_curr_len), dtype=torch.int64)
	prev_masks = torch.zeros((batch_size, max_prev_len), dtype=torch.bool)
	curr_masks = torch.zeros((batch_size, max_curr_len), dtype=torch.bool)

	for i in range(batch_size) :
		item = batch[i]
		prev_len = item.prev_durr.shape[0]
		curr_len = item.curr_durr.shape[0]

		prev_durr[i, :prev_len] = torch.from_numpy(item.prev_durr)
		prev_pitc[i, :prev_len, :] = torch.from_numpy(item.prev_pitc)
		prev_tones[i, :prev_len] = torch.from_numpy(item.prev_tone)
		curr_durr[i, :curr_len] = torch.from_numpy(item.curr_durr)
		curr_pitc[i, :curr_len, :] = torch.from_numpy(item.curr_pitc)
		curr_tones[i, :curr_len] = torch.from_numpy(item.curr_tone)
		prev_masks[i, :prev_len] = 1
		curr_masks[i, :curr_len] = 1

	return prev_durr, prev_pitc, prev_tones, curr_durr, curr_pitc, curr_tones, prev_masks, curr_masks

if __name__ == "__main__" :
	dataset = ToneDataset("./dataset/Humdrum-files/")
	ratio = get_ratio(dataset)
	print("Tone ratio (0,2,4,3):", ratio)