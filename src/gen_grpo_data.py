import os
import data
import json
import models
import torch
import argparse
import numpy as np
from tqdm import tqdm

dataset : list[list[str]] = []

model : torch.nn.Module | None = None

def parse_args() :
	parser = argparse.ArgumentParser()
	parser.add_argument('--tone_model', type=str, default='ckpts/tone_model_best.pth', 
					 help='Path to the trained tone model')
	parser.add_argument('--data_dir', type=str, default='dataset/hsd-dataset/notation',
					 help='Directory containing notation files')
	parser.add_argument('--output_path', type=str, default='dataset/grpo.json',
					 help='Path to save the generated grope data')
	return parser.parse_args()

def make_item(tone_seq : list[int], song_id : int, pit_idx : int) -> str :
	prompt_sys = data.template_prompt_sys
	prompt_usr = data.template_prompt_usr.format(
		prev_lyrics = [],
		rhyme = "",
		char_num = len(tone_seq),
		pitches = tone_seq,
	)
	prompt = [
		{"role": "system", "content": prompt_sys},
		{"role": "user", "content": prompt_usr},
	]
	return dict(prompt=prompt, song_id=song_id, pit_idx=pit_idx)

def parse_notation(note_path : str, song_idx : int) -> list[str] :
	with open(note_path, 'r', encoding='utf-8') as f :
		lines = f.readlines()
	lines = [line.strip() for line in lines]
	dur : list[list[float]] = []
	pit : list[list[int]] = []
	for i, line in enumerate(lines) :
		tks = line.split()
		dur.append([])
		pit.append([])
		for tk in tks :
			tk_dur, tk_pit = tk.split(',')
			tk_dur = float(tk_dur)
			tk_pit = int(tk_pit)

			if tk_pit != 0 :
				dur[-1].append(tk_dur)
				pit[-1].append(tk_pit)
	
	# transform to tensor
	
	dataitems = [
		data.DataItem(
			[[(durr_item, pit_item)] for durr_item, pit_item in zip(curr_durr, curr_pitc)],
			"x",
			["1" for _ in range(len(curr_durr))]
		)
		for curr_durr, curr_pitc in zip(dur, pit)
	]

	# only one melody is enough, use curr_pitc and curr_durr as prev_durr, prev_pitc
	for item in dataitems :
		item.setWhole((item.curr_durr, item.curr_pitc, np.zeros((len(item.curr_durr)))))
		item.normalize()
	subdataset : list[str] = []
	# generate tone sequence
	with torch.no_grad() :
		for idx, item in enumerate(dataitems) :
			curr_durr = torch.tensor(item.curr_durr, dtype=torch.float32).unsqueeze(0).to(models.device)
			curr_pitc = torch.tensor(item.curr_pitc, dtype=torch.int32).unsqueeze(0).to(models.device)
			curr_mask = torch.ones(item.curr_tone.shape, dtype=torch.bool).unsqueeze(0).to(models.device)
			logits = model(
				curr_durr, curr_pitc, None,
				curr_durr, curr_pitc,
				curr_mask, curr_mask,
			)
			pred_tone = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy().tolist()
			map0243 = {0: '0', 1: '2', 2: '4', 3: '3'}
			tone_seq = [int(map0243[tone]) for tone in pred_tone]
			# print(f"Generated tone sequence for {note_path}: {tone_seq}")

			subdataset.append(make_item(tone_seq, song_idx, idx))	
	return subdataset
		
	


if __name__ == "__main__" :
	args = parse_args()

	model = models.ToneModel(data.TONE_VOCAB_SIZE)
	model.load_state_dict(torch.load(args.tone_model))

	model = model.to(models.device)

	items = os.listdir(args.data_dir)

	for idx, item in enumerate(tqdm(items)) :
		if not item.endswith('.txt') :
			continue
		sub_dataset = parse_notation(os.path.join(args.data_dir, item), idx)
		dataset.extend(sub_dataset)

	with open(args.output_path, 'w', encoding='utf-8') as f :
		json.dump(dataset, f, ensure_ascii=False, indent=4)