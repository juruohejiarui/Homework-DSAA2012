import json
import os
from tqdm import tqdm
from data import RANDOM_SEED, template_prompt_sys, template_prompt_usr, map_9to6, map_6to4, map_4toToken
from pycantonese import characters_to_jyutping as char2jyut
import random

dataset : list[dict] = []

src_path = "dataset/tone.json"
char_path = "dataset/char.txt"
char2_path = "dataset/char2.txt"
char3_path = "dataset/char3.txt"

# 常用字0243字典，从char3.txt中获取
ext_dict : dict[str, list[str]] = {}

char_set = set()
unmeet_char_set = set()

# no need to split data, overfit is allowed for pre-sft stage
output_path = "dataset/emb.json"

random.seed(RANDOM_SEED)

def make_char_set() :
	# parse char.txt
	with open(char_path, 'r', encoding='utf-8') as f :
		lines = f.readlines()
	for line in lines :
		line = line.strip()
		if len(line) == 0 :
			continue
		prts = line.split('〔')[1 : ]
		for prt in prts :
			pos = prt.find('〕')
			for ch in prt[: pos] :
				if ch == ' ' or ch == '\n' or ch == ')' or ch == '(' :
					continue
				char_set.add(ch)
	
	# parse char2.txt
	with open(char2_path, 'r', encoding='utf-8') as f :
		lines = f.readlines()
	
	for line in lines :
		line = line.strip()
		if len(line) == 0 :
			continue
		ch = line[0]
		char_set.add(ch)

	with open(char3_path, 'r', encoding='utf-8') as f :
		chars = f.read()
	
	for ch in chars :
		if ch == ' ' or ch == '\n' or ch == ')' or ch == '(' :
			continue
		char_set.add(ch)

		jyut = char2jyut(ch)
		if len(jyut) == 0 or jyut[0][1] is None :
			continue
		jyut = jyut[0][1]
		tone = map_4toToken[map_6to4[map_9to6[jyut[-1]]]]

		if tone not in ext_dict :
			ext_dict[tone] = []
		ext_dict[tone].append(ch)
	
	print(f"Total {len(char_set)} unique characters parsed.")
	print("".join(list(char_set)[:100]) + "...")

	unmeet_char_set = char_set.copy()

def make_item(word : str, jyut : str) -> dict :
	tones : list[int] = []
	for ch in jyut :
		if ch.isdigit() :
			tone9 = ch
			tone6 = map_9to6[tone9]
			tone4 = map_6to4[tone6]
			toneToken = map_4toToken[tone4]
			tones.append(toneToken)
	if len(tones) != len(word) :
		return None
	if len(tones) > 1 :
		return None
	for ch in word :
		if ch in unmeet_char_set :
			unmeet_char_set.remove(ch)
		if ch not in char_set :
			return None
	prompt_sys = template_prompt_sys
	prompt_usr = \
	"""请写出下面繁体汉字的粤语0243声调（每个汉字对应一个声调数字），字符串形式返回音调。汉字：{word}""".format(
		word = word,
	)
	prompt_assistant = "".join(tones)

	if len(word) == 1 :
		# randomly select 50 characters from ext_dict
		ext_chars = ext_dict[tones[0]][: 40] + random.sample(ext_dict[tones[0]][40: ], 10)
		prompt_usr += "\n请额外提供0243音调相同的50个繁体汉字。"

		prompt_assistant += "\n" + ",".join(ext_chars)

	prompt = [
		{"role": "system", "content": prompt_sys},
		{"role": "user", "content": prompt_usr},
		{"role": "assistant", "content": prompt_assistant},
	]
	return dict(messages=prompt, num_turns=3)

def make_item2(tone : str) -> dict :
	prompt_sys = template_prompt_sys
	prompt_usr = \
	"""请写出符合0243声调：{tone} 的繁体汉字50个。字符串形式返回汉字。""".format(
		tone = tone,
	)
	ext_chars = ext_dict[tone][: 30] + random.sample(ext_dict[tone][30: ], 20)
	prompt_assistant = ",".join(ext_chars)

	prompt = [
		{"role": "system", "content": prompt_sys},
		{"role": "user", "content": prompt_usr},
		{"role": "assistant", "content": prompt_assistant},
	]
	return dict(messages=prompt, num_turns=3)


if __name__ == "__main__" :

	make_char_set()

	with open(src_path, 'r', encoding='utf-8') as f :
		data = json.load(f)
	
	for key, val in tqdm(data.items()) :
		item = make_item(key, val)
		if item is not None :
			dataset.append(item)
	
	for ch in unmeet_char_set :
		res = char2jyut(ch)
		if len(res) == 0 or res[0][1] is None :
			continue
		jyut = res[0][1]
		item = make_item(ch, jyut)
		if item is not None :
			dataset.append(item)
	
	for toneToken in ext_dict.keys() :
		for _ in range(100) :
			item = make_item2(toneToken)
			dataset.append(item)

	with open(output_path, 'w', encoding='utf-8') as f :
		json.dump(dataset, f, ensure_ascii=False, indent=4)