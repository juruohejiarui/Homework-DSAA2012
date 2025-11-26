import torch
import torch.nn as nn
from .parse import DataItem, parse_krn, TONE_VOCAB_SIZE
import os
from tqdm import tqdm
import numpy as np
import random
import json
from transformers import AutoTokenizer

template_prompt_sys = """你是一个专业的粤语作词家"""
template_prompt_usr = \
"""根据给定的Pitches，生成与之适配且相同长度的歌词，每个Pitch对应一个繁体中文字符。此外，生成的歌词要与给定的Previous lyrics连贯。同时，生成的最后一个中文字符需要和给定的Rhyme押韵。
Previous lyrics: {prev_lyrics}
Rhyme: {rhyme}
Character Nums: {char_num}
Pitches: {pitches}"""

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

class SFTDataset(torch.utils.data.Dataset) :
	def __init__(self, json_path : str) :
		with open(json_path, "r", encoding="utf-8") as f :
			self.data = json.load(f)
	def __len__(self) :
		return len(self.data)
	def __getitem__(self, idx) :
		return self.data[idx]
	
class GRPODataset(torch.utils.data.Dataset) :
	def __init__(self, json_path : str, shuffle : bool = True) :
		with open(json_path, "r", encoding="utf-8") as f :
			songs = json.load(f)
		self.datas = songs
		
	def __len__(self) :
		return len(self.data)

	def __getitem__(self, idx) -> str :
		return self.data[idx]
	
def tone_collate_fn(batch : list[DataItem]) :
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

def sft_collate_fn(batch: list[dict], tokenizer: AutoTokenizer, max_length: int = 2048):
	"""
	batch: list of dicts each has keys 'prompt' and 'answer' (both str)
	tokenizer: HF tokenizer (确保 tokenizer.pad_token 已设置)
	返回: dict of tensors 可直接喂 Trainer.train()
	"""
	prompts = [b['prompt'] for b in batch]
	answers = [b['answer'] for b in batch]

	# 1) 全部合并的文本（prompt + answer），一次编码（保证 truncation/padding 一致）
	concat_texts = [p + a for p, a in zip(prompts, answers)]
	enc = tokenizer(
		concat_texts,
		padding=True,
		truncation=True,
		max_length=max_length,
		return_tensors="pt",
		add_special_tokens=True,
	)

	# 2) 计算每条 prompt 的 token 长度（使用相同的 max_length/truncation 保证一致）
	#	这里 batch 处理 prompts，注意要用相同的 truncation/max_length
	prompt_enc = tokenizer(
		prompts,
		padding=False,
		truncation=True,
		max_length=max_length,
		add_special_tokens=False,
	)
	# prompt_enc['input_ids'] 是 list[list[int]]
	prompt_lens = [len(x) for x in prompt_enc["input_ids"]]

	# 3) 制作 labels：prompt 部分设为 -100，仅对 answer 部分计算 loss
	labels = enc["input_ids"].clone()
	seq_len = labels.size(1)
	for i, plen in enumerate(prompt_lens):
		# 防止 plen 超过 seq_len（truncation 情况），取 min 保护越界
		cut = min(plen, seq_len)
		if cut > 0:
			labels[i, :cut] = -100

	enc["labels"] = labels

	return enc

def grpo_collate_fn(batch : list[tuple[int, str]]) :
	song_ids = [item[0] for item in batch]
	tones_list = [item[1] for item in batch]
	return song_ids, tones_list

if __name__ == "__main__" :
	dataset = ToneDataset("./dataset/Humdrum-files/")
	ratio = get_ratio(dataset)
	print("Tone ratio (0,2,4,3):", ratio)