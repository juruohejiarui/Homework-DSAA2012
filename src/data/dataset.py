import torch
import torch.nn as nn
from .parse import DataItem, parse_krn, TONE_VOCAB_SIZE
import os
from tqdm import tqdm
import numpy as np
import random

all_items = None

class Dataset(torch.utils.data.Dataset) :
	def __init__(self, data_dir : str, split : float = 1.0, train : bool = True) :
		global all_items
		if all_items is None :
			all_items = os.listdir(data_dir)
			random.seed(42)
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

	def __len__(self) :
		return len(self.data_items)

	def __getitem__(self, idx) :
		return self.data_items[idx]
	
def get_ratio(dataset : torch.utils.data.Dataset) -> np.ndarray :
	cnt = np.zeros((TONE_VOCAB_SIZE,), dtype=np.int64)
	for item in dataset :
		for i in range(TONE_VOCAB_SIZE) :
			cnt[i] += np.sum(item.curr_tone == i).item()

	return cnt / np.sum(cnt)

def collate_fn(batch : list[DataItem]) :
	batch_size = len(batch)
	max_prev_len = max([item.prev_durr.shape[0] for item in batch])
	max_curr_len = max([item.curr_durr.shape[0] for item in batch])

	prev_durr = torch.zeros((batch_size, max_prev_len, 2), dtype=torch.float32)
	prev_pitc = torch.zeros((batch_size, max_prev_len, 2), dtype=torch.int64)
	prev_tones = torch.zeros((batch_size, max_prev_len), dtype=torch.int64)
	curr_durr = torch.zeros((batch_size, max_curr_len, 2), dtype=torch.float32)
	curr_pitc = torch.zeros((batch_size, max_curr_len, 2), dtype=torch.int64)
	curr_tones = torch.zeros((batch_size, max_curr_len), dtype=torch.int64)
	prev_masks = torch.zeros((batch_size, max_prev_len), dtype=torch.bool)
	curr_masks = torch.zeros((batch_size, max_curr_len), dtype=torch.bool)

	for i in range(batch_size) :
		item = batch[i]
		prev_len = item.prev_durr.shape[0]
		curr_len = item.curr_durr.shape[0]

		prev_durr[i, :prev_len, :] = torch.from_numpy(item.prev_durr)
		prev_pitc[i, :prev_len, :] = torch.from_numpy(item.prev_pitc)
		prev_tones[i, :prev_len] = torch.from_numpy(item.prev_tone)
		curr_durr[i, :curr_len, :] = torch.from_numpy(item.curr_durr)
		curr_pitc[i, :curr_len, :] = torch.from_numpy(item.curr_pitc)
		curr_tones[i, :curr_len] = torch.from_numpy(item.curr_tone)
		prev_masks[i, :prev_len] = 1
		curr_masks[i, :curr_len] = 1

	return prev_durr, prev_pitc, prev_tones, curr_durr, curr_pitc, curr_tones, prev_masks, curr_masks

if __name__ == "__main__" :
	dataset = Dataset("./dataset/Humdrum-files/")
	ratio = get_ratio(dataset)
	print("Tone ratio (0,2,4,3):", ratio)