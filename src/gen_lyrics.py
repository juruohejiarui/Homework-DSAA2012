from data import (
	map_4toToken, map_6to4, map_9to6,
	TONE_VOCAB_SIZE,
	template_prompt_sys, template_prompt_usr, template_prompt_usr_mask
)
from pycantonese import characters_to_jyutping
from models import ToneModel, device
from queue import PriorityQueue
import torch
import copy
import openai

tone_model : ToneModel = None
lyrics_model : openai.OpenAI = None
temperature = 0.9
num_generate = 16

def setup_models(tone_model_path : str, lyrics_path : str) :
	global tone_model, lyrics_model, lyrics_model_path

	tone_model = ToneModel(TONE_VOCAB_SIZE)
	tone_model.load_state_dict(torch.load(tone_model_path, map_location=device))
	tone_model.to(device)

	lyrics_model = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
	lyrics_model_path = lyrics_path

def make_prompts(prev_lyrics : list[str], pitc : str, masked_lyrics : str | None = None) -> list[dict[str, str]] :
	char_num = len(pitc)
	rhyme = prev_lyrics[-1][-1] if len(prev_lyrics) > 0 else ""

	if masked_lyrics is None :
		prompt_usr = template_prompt_usr.format(
			prev_lyrics=prev_lyrics,
			rhyme=rhyme,
			char_num=char_num,
			pitches=pitc
		)
	else :
		prompt_usr = template_prompt_usr_mask.format(
			prev_lyrics=prev_lyrics,
			rhyme=rhyme,
			char_num=char_num,
			masked_lyrics=masked_lyrics,
			pitches=pitc
		)
	return [
		{"role": "system", "content": template_prompt_sys},
		{"role": "user", "content": prompt_usr}
	]

def generate_lyrics(prompts : list[dict[str, str]], max_len : int = 32) -> list[str] :
	global lyrics_model, lyrics_model_path
	
	responses = lyrics_model.chat.completions.create(
		model=lyrics_model_path,
		messages=prompts,
		temperature=0.7,
		top_p=1.0,
		n=num_generate,
	)
	return [choice.message.content.strip() for choice in responses.choices]

# check correctness of generated lyrics and return masked lyrics
# - no mask (￥) means completely correct
# - mask means partially correct, return the masked lyrics
# - False means completely incorrect
def chk_lyrics(lyrics : str, pitc : str) -> str | bool :
	if len(lyrics) != len(pitc) :
		return False
	
	jyut_lst = characters_to_jyutping(lyrics)
	tones : list[str | int] = []
	ret_lyrics = copy.deepcopy(lyrics)

	idx = 0
	for chs, jyut in jyut_lst :
		if jyut is None : 
			tones.extend([0] * len(chs))
			continue
		for p in range(len(jyut)) :
			if jyut[p].isdigit() :
				tones.append(jyut[p])

	if len(tones) != len(pitc) :
		return False

	for i in range(len(pitc)) :
		if tones[i] == 0 :
			ret_lyrics = ret_lyrics[0 : i] + '￥' + ret_lyrics[i + 1 : ]
			continue
		tone2Pitc = map_4toToken[map_6to4[map_9to6[tones[i]]]]
		if tone2Pitc != pitc[i] :
			ret_lyrics = ret_lyrics[0 : i] + '￥' + ret_lyrics[i + 1 : ]
	
	return ret_lyrics

def search_valid_lyrics(prev_lyrics : list[str], pitc : str, max_iter : int = 200, max_candidate : int = 5) -> str :
	que : PriorityQueue = PriorityQueue()
	que.put((len(pitc), None))   # (cnt_masked, masked_lyrics)
	cnt_iter : int = 0

	valid_candidates : list[str] = []

	for iter in range(max_iter) :
		cnt_masked_ori, masked_lyrics = que.get()

		prompts = make_prompts(prev_lyrics, pitc, masked_lyrics)
		gen_lyrics : list[str] = generate_lyrics(prompts, max_len=len(pitc)*2)

		for entries in gen_lyrics :
			chk_res = chk_lyrics(entries, pitc)
			if chk_res is False :
				que.put((cnt_masked_ori, entries))
			elif isinstance(chk_res, str) :
				if '￥' not in chk_res :
					valid_candidates.append(chk_res)
					if len(valid_candidates) >= max_candidate :
						return valid_candidates
				else :
					cnt_masked = chk_res.count('￥')
					que.put((cnt_masked, chk_res))
		
		cnt_iter += 1
	
	print(f"[Warning] Failed to generate valid lyrics after {cnt_iter} iterations.")
	if len(valid_candidates) == 0 :
		return gen_lyrics[0]
	else :
		return valid_candidates


def from_pitc_to0243(curr_durr, curr_pitc, prev_durr, prev_pitc, curr_mask, prev_mask) -> str :
	global tone_model

	logits = tone_model(
		prev_durr, prev_pitc, None,
		curr_durr, curr_pitc,
		curr_mask, prev_mask,
		)
	preds : torch.Tensor = logits.argmax(dim=-1)

	return ''.join([map_4toToken[tone.item()] for tone in preds[0]])

if __name__ == "__main__" :
	setup_models(
		"ckpts/tone_model_best.pth",
		"./ckpts/lyrics"
	)

	prev_lyrics : list[str] = []
	
	prev_notes : torch.Tensor = None
	curr_notes : list[str] = None

	while True :
		# show prev_lyrics
		print("Previous Lyrics:")
		for line in prev_lyrics :
			print('\t', line)
		
		if curr_notes is None :
			notes = input("Enter the current notes (space-separated, e.g. C D E F G A B c C#): ")
			curr_notes = notes.strip().split()
			
			
		