from data import RANDOM_SEED, parse_krn, template_prompt_usr, template_prompt_usr_mask, template_prompt_sys, map_9to6, map_6to4, map_4toToken
import numpy as np
import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict, Features, List
import random
from tqdm import tqdm
from hanziconv import HanziConv
from pycantonese import characters_to_jyutping as char2jyut

dataset : list[dict] = []

ext_sft_dir : str = "dataset/ext_sft"

def gen_item(prev_lyrics : list[str], rhyme : str, char_num : int, pitches : str, output : str) -> dict:
	if len(output) > 1 and random.random() < 0.3 :
		# choose contiguous chars to mask
		mask_len = random.randint(1, min(4, char_num - 1))
		start_idx = random.randint(0, char_num - mask_len)
		masked_lyrics = output[:start_idx] + "￥" * mask_len + output[start_idx + mask_len:]

		prompt_usr = template_prompt_usr_mask.format(
			prev_lyrics = prev_lyrics,
			rhyme = rhyme,
			char_num = char_num,
			masked_lyrics = masked_lyrics,
			pitches = pitches,
		)
		
	else :
		prompt_usr = template_prompt_usr.format(
			prev_lyrics = prev_lyrics,
			rhyme = rhyme,
			char_num = char_num,
			pitches = pitches,
		)
	prompt_sys = template_prompt_sys

	prompt = [
		{"role": "system", "content": prompt_sys},
		{"role": "user", "content": prompt_usr},
		{"role": "assistant", "content": output}
	]
	return dict(messages=prompt, num_turns=3)
	
def output_krn(filepath : str) :
	que : list[str] = []
	items = parse_krn(filepath)
	for item in items :
		as_mask : bool = random.random() < 0.3
		lyrics = "".join(item.text)
		pitch = item.curr_tone.tolist()
		pitch = [map_4toToken[p] for p in pitch]
		pitch = "".join(pitch)
		char_num = len(lyrics)
		rhyme = lyrics[-1]

		dataitem = gen_item(
			prev_lyrics = que,
			rhyme = rhyme,
			char_num = char_num,
			pitches = pitch,
			output = lyrics
		)

		dataitem['source'] = 'krn'

		dataset.append(dataitem)

		que.append(lyrics)
		if len(que) >= 10 :
			que = que[-10 :]

def add_extra_sft_data(ext_path : str) :
	global dataset
	lines = []
	with open(ext_path, 'r', encoding='utf-8') as f :
		lines = f.readlines()
	ext_data = []
	for st in range(1, len(lines) - 1, 6) :
		try :
			raw = "".join(lines[st : st + 6]).strip()
			raw = raw.rstrip(',')
			ext_data.append(json.loads(raw))
		except Exception as e :
			print(f"Error parsing extra SFT data at lines {st} to {st+6} in {ext_path}")
			print(f"error: {e}")
	# examples of items :
	"""{
        "system": "你是一个专业的粤语作词家",
        "instruction": "根据给定的Pitches，生成与之适配且相同长度的歌词，每个Pitch对应一个中文字符。此外，生成的歌词要与给定的Previous lyrics连贯。同时，生成的最后一个中文字符需要和给定的Rhyme押韵。\nPrevious lyrics: ['如花般的女子真诚留住你']\nRhyme: 你\n[Character Nums: 8]",
        "input": "Pitches: 이삼일사사삼삼사",
        "output": "共舞于花海中远飞"
    }"""

	# convert simplified chinese to traditional chinese

	for item in ext_data :
		inst = item['instruction']
		output = item['output']

		has_rhyme = "Rhyme:" in inst

		if not has_rhyme :
			prev_lyrics_line = inst.split("Previous lyrics: ")[1].split("[Character Nums:")[0].strip()
			rhyme_line = ""
		else :
			prev_lyrics_line = inst.split("Previous lyrics: ")[1].split("Rhyme: ")[0].strip()
			rhyme_line = inst.split("Rhyme: ")[1].split("[Character Nums:")[0].strip()
			rhyme_line = HanziConv.toTraditional(rhyme_line)
		prev_lyrics_line = json.loads(prev_lyrics_line.replace("'", '"'))
		prev_lyrics_line = [HanziConv.toTraditional(lyric) for lyric in prev_lyrics_line]

		output = HanziConv.toTraditional(output)

		char_num = len(output)

		# create new pitch with map_4toToken and output
		jyut_res = char2jyut(output)
		pitch_seq : list[str] = []
		for ch, jyut in jyut_res :
			if jyut is None :
				continue
			sep_idx = []
			for i, c in enumerate(jyut) :
				if c.isdigit() :
					sep_idx.append(i)
			tones = [map_4toToken[map_6to4[map_9to6[jyut[i]]]] for i in sep_idx]
			pitch_seq.extend(tones)
		
		if len(pitch_seq) != char_num :
			print("Ignore data: output=", output)
			continue

		pitches = "".join(pitch_seq)
		dataitem = gen_item(
			prev_lyrics = prev_lyrics_line,
			rhyme = rhyme_line,
			char_num = char_num,
			pitches = pitches,
			output = output
		)
		dataitem['source'] = 'ext_sft'
		dataset.append(dataitem)
				

if __name__ == "__main__" :
	random.seed(RANDOM_SEED)
	dir_path = "dataset/Cantopop-corpus/Humdrum-files/"
	items = os.listdir(dir_path)
	for item in tqdm(items) :
		if not item.endswith(".krn") :
			continue
		output_krn(os.path.join(dir_path, item))

	items = os.listdir(ext_sft_dir)
	for item in tqdm(items) :
		if not item.endswith(".json") :
			continue
		add_extra_sft_data(os.path.join(ext_sft_dir, item))
		print(f"Added extra SFT data from {item}, total dataset size: {len(dataset)}")

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