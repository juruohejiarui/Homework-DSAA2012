from data import (
	map_4toToken, map_6to4, map_9to6,
	TONE_VOCAB_SIZE,
	template_prompt_sys, template_prompt_usr, template_prompt_usr_mask,
	parse_notelist, DataItem
)
from pycantonese import characters_to_jyutping
from models import ToneModel, device
from queue import PriorityQueue
import numpy as np
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
	tone_model.eval()

	lyrics_model = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
	lyrics_model_path = lyrics_path

def _make_prompts(prev_lyrics : list[str], pitc : str, masked_lyrics : str | None = None) -> list[dict[str, str]] :
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

def gen_lyrics(prompts : list[dict[str, str]], max_len : int = 32, num_generate=num_generate, temperature=temperature, top_p : float | None = 1.0) -> list[str] :
	global lyrics_model, lyrics_model_path
	
	responses = lyrics_model.chat.completions.create(
		model=lyrics_model_path,
		messages=prompts,
		temperature=temperature,
		top_p=top_p,
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
	que.put((len(pitc), 0, None))   # (cnt_masked, masked_lyrics)
	cnt_iter : int = 0

	valid_candidates : list[str] = []
	visited = set()

	for iter in range(max_iter) :
		cnt_masked_ori, iter_dp, masked_lyrics = que.get()
		if iter_dp > 4 :
			continue
		if isinstance(masked_lyrics, str) :
			if masked_lyrics in visited :
				continue
			visited.add(masked_lyrics)

		prompts = _make_prompts(prev_lyrics, pitc, masked_lyrics)
		new_cands : list[str] = gen_lyrics(
			prompts,
			max_len=len(pitc)*2,
			num_generate=num_generate * 2 if masked_lyrics is None else num_generate
		)


		for entries in new_cands :
			chk_res = chk_lyrics(entries, pitc)
			if chk_res is False :
				que.put((cnt_masked_ori, iter_dp + 1, entries))
			elif isinstance(chk_res, str) :
				if '￥' not in chk_res :
					valid_candidates.append(chk_res)
					if len(valid_candidates) >= max_candidate :
						return valid_candidates[ : max_candidate]
				else :
					for i in range(4) :
						more_mask = copy.deepcopy(chk_res)
						# randomly mask some more characters
						positions = [j for j in range(len(chk_res)) if chk_res[j] != '￥']
						if positions == [] :
							break
						np.random.shuffle(positions)

						num_to_mask = max(1, len(positions) // (4 - i))
						for k in range(num_to_mask) :
							more_mask = more_mask[0 : positions[k]] + '￥' + more_mask[positions[k] + 1 : ]
							que.put((cnt_masked_ori + num_to_mask, iter_dp + 1, more_mask))
					cnt_masked = chk_res.count('￥')
					que.put((cnt_masked, iter_dp + 1, chk_res))
		
		cnt_iter += 1
	
	if len(valid_candidates) == 0 :
		raise Exception("No valid lyrics found.")
	return valid_candidates


def _from_pitc_to2043(curr_durr, curr_pitc, prev_durr, prev_pitc, curr_mask, prev_mask, max_num_cands : int = 10) -> list[str] :
	global tone_model

	logits : torch.Tensor = tone_model(
		prev_durr, prev_pitc, None,
		curr_durr, curr_pitc,
		prev_mask, curr_mask
		)
	cands = []
	bst = logits.argmax(dim=-1)
	cands.append([map_4toToken[bst_item.item()] for bst_item in bst[0]])
	for i in range(logits.shape[1]) :
		logits_clone = logits.detach().clone()
		logits_clone[0, i, bst[0, i]] = -1e9
		bst_clone = logits_clone.argmax(dim=-1)
		cands.append([map_4toToken[bst_item.item()] for bst_item in bst_clone[0]])
	
	for i in range(logits.shape[1]) :
		for j in range(i + 1, logits.shape[1]) :
			logits_clone = logits.detach().clone()
			logits_clone[0, i, bst[0][i]] = -1e9
			logits_clone[0, j, bst[0][j]] = -1e9
			bst_clone = logits_clone.argmax(dim=-1)
			cands.append([map_4toToken[bst_item.item()] for bst_item in bst_clone[0]])
	return [''.join(cand) for cand in cands]
		

def generate_0243(prev_notes : list[str], curr_notes : str) -> str :
	prev_notes = [prev_note.split(' ') for prev_note in prev_notes]
	curr_notes : list[str] = curr_notes.split(' ')

	prev_notes = [parse_notelist(prev_note) for prev_note in prev_notes]
	curr_notes = parse_notelist(curr_notes)

	items = [
		DataItem(prev_note, ['1'] * len(prev_note), '1' * len(prev_note)) for prev_note in prev_notes
	]
	items.append(DataItem(curr_notes, ['1'] * len(curr_notes[0]), '1' * len(curr_notes[0])))

	# convert to tensors
	prev_durr = []
	prev_pitc = []

	for item in items[: -1] :
		prev_durr.append(item.curr_durr)
		prev_pitc.append(item.curr_pitc)
		prev_durr.append(np.zeros(1, dtype=np.float32))
		prev_pitc.append(np.zeros(1, dtype=np.int64))
	
	prev_durr.append(items[-1].curr_durr)
	prev_pitc.append(items[-1].curr_pitc)

	item = items[-1]
	item.setWhole((
		np.concatenate(prev_durr, axis=0),
		np.concatenate(prev_pitc, axis=0),
		np.array([TONE_VOCAB_SIZE], dtype=np.int64)
	))

	item.normalize()

	# convert to torch tensors
	prev_durr_tensor = torch.from_numpy(item.prev_durr).unsqueeze(0).to(device)
	prev_pitc_tensor = torch.from_numpy(item.prev_pitc).unsqueeze(0).to(device)
	curr_durr_tensor = torch.from_numpy(item.curr_durr).unsqueeze(0).to(device)
	curr_pitc_tensor = torch.from_numpy(item.curr_pitc).unsqueeze(0).to(device)

	curr_mask = torch.ones((1, curr_durr_tensor.shape[1]), dtype=torch.bool).to(device)
	prev_mask = torch.ones((1, prev_durr_tensor.shape[1]), dtype=torch.bool).to(device)

	return _from_pitc_to2043(
		curr_durr_tensor, curr_pitc_tensor,
		prev_durr_tensor, prev_pitc_tensor,
		curr_mask, prev_mask
	)