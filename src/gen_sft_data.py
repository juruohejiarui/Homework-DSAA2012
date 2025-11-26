from data import RANDOM_SEED, parse_krn, template_prompt_usr, template_prompt_sys
import numpy as np
import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict, Features, List
import random
from tqdm import tqdm

dataset : list[dict] = []

def output_krn(filepath : str) :
	que : list[str] = []
	items = parse_krn(filepath)
	for item in items :
		lyrics = "".join(item.text)
		pitch = item.curr_tone.tolist()
		char_num = len(lyrics)
		rhyme = lyrics[-1]

		prompt_usr = template_prompt_usr.format(
			prev_lyrics = que,
			rhyme = rhyme,
			char_num = char_num,
			pitches = pitch,
		)
		prompt_sys = template_prompt_sys

		prompt = [
			{"role": "system", "content": prompt_sys},
			{"role": "user", "content": prompt_usr},
			{"role": "assistant", "content": lyrics}
		]
		dataset.append(dict(messages=prompt, num_turns=3))

		que.append(lyrics)
		if len(que) >= 10 :
			que = que[-10 :]

if __name__ == "__main__" :
	dir_path = "dataset/Cantopop-corpus/Humdrum-files/"
	items = os.listdir(dir_path)
	for item in tqdm(items) :
		if not item.endswith(".krn") :
			continue
		output_krn(os.path.join(dir_path, item))

	random.seed(RANDOM_SEED)

	random.shuffle(dataset)
	split_index = int(len(dataset) * 0.9)

	train_dataset = dataset[:split_index]
	eval_dataset = dataset[split_index :]

	sft_path = "dataset/sft-train.json"
	with open(sft_path, 'w', encoding='utf-8') as f :
		json.dump(train_dataset, f, ensure_ascii=False, indent=4)
	
	sft_path = "dataset/sft-eval.json"
	with open(sft_path, 'w', encoding='utf-8') as f :
		json.dump(eval_dataset, f, ensure_ascii=False, indent=4)