from data import DataItem, parse_krn
import numpy as np
import os
import json
from tqdm import tqdm

template_system = """你是一个专业的粤语作词家"""
template_inst = \
"""根据给定的Pitches，生成与之适配且相同长度的歌词，每个Pitch对应一个繁体中文字符。此外，生成的歌词要与给定的Previous lyrics连贯。同时，生成的最后一个中文字符需要和给定的Rhyme押韵。
Previous lyrics: {prev_lyrics}
Rhyme: {rhyme}
Character Nums: {char_num}
"""
template_input = """Pitches: {pitches}"""

dataset : list[dict] = []

def make_inst(prev_lyrics : list[str], rhyme : str, char_num : int) -> str :
	return template_inst.format(
        prev_lyrics = prev_lyrics,
        rhyme = rhyme,
        char_num = char_num,
    )

def make_input(pitches : list[int]) -> str :
	return template_input.format(
        pitches = pitches,
    )

def output_krn(filepath : str) :
	que : list[str] = []
	items = parse_krn(filepath)
	for item in items :
		lyrics = "".join(item.text)
		pitch = item.curr_tone.tolist()
		char_num = len(lyrics)
		rhyme = lyrics[-1]

		inst, input = make_inst(que, rhyme, char_num), make_input(pitch)

		dataset.append({
			"system": template_system,
			"instruction": inst,
			"input": input,
			"output": lyrics
		})

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
	
	sft_path = "dataset/sft.json"
	with open(sft_path, "w", encoding="utf-8") as f :
		json.dump(dataset, f, ensure_ascii=False, indent=4)