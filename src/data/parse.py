import kernpy as kp
import os
import numpy as np
from tqdm import tqdm

MAX_PITCH = 48
TONE_VOCAB_SIZE = 4
PREV_SIZE = 3

map_9to6 = {
	'1': 1,
	'2': 2,
	'3': 3,
	'4': 4,
	'5': 5,
	'6': 6,
	'7': 1,
	'8': 3,
	'9': 6
}

map_6to4 = {
	1: 3,
	2: 3,
	3: 2,
	4: 0,
	5: 2,
	6: 1
}

map_4toToken = {
	0: '일',
	1: '이',
	2: '삼',
	3: '사',
}

map_tokenTo4 = {
	'일': 0,
	'이': 1,
	'삼': 2,
	'사': 3,
}

def to0243(tones : np.ndarray) -> np.ndarray :
	map0243 = np.ones(tones.shape, dtype=np.int64) * TONE_VOCAB_SIZE
	tones = np.array([map_9to6[str(tone)] for tone in tones])
	# 3
	map0243[tones == 1] = 3
	map0243[tones == 2] = 3
	# 4
	map0243[tones == 3] = 2
	map0243[tones == 5] = 2
	# 2
	map0243[tones == 6] = 1
	# 0
	map0243[tones == 4] = 0
	if max(map0243) == TONE_VOCAB_SIZE :
		print("Warning: tone outside 1-6 found:", tones, " mapped to", map0243)
	return map0243

class DataItem:
	def __init__(self, note : list[list[int, int]], text : list[str], jyut : list[str]) :
		curr_note : np.ndarray = []
		self.text = text
		self.curr_tone : np.ndarray = []
		
		for note_seq in note :
			if len(note_seq) == 1 :
				curr_note.append((note_seq[0][0], note_seq[0][1]))
			else :
				# get pitc with max duration
				max_dur = -1
				max_pit = 128
				tot_dur = 0
				for dur, pit in note_seq :
					if dur > max_dur :
						max_dur = dur
						max_pit = pit
					tot_dur += dur
				curr_note.append((tot_dur, max_pit))

		curr_note = np.array([(dur, pitc) for dur, pitc in curr_note])
		self.text = text
		self.curr_durr = np.array([durr for durr, _ in curr_note], dtype=np.float32)
		self.curr_pitc = np.array([pitc for _, pitc in curr_note], dtype=np.int64)

		assert len(self.curr_pitc) == 1 or abs(self.curr_pitc[1] - self.curr_pitc[0]) <= 24, f"Adjacent pitch difference too large: {self.curr_pitc} in text {self.text}"

		self.curr_tone = to0243(np.array([int(jyutone[-1]) for jyutone in jyut], dtype=np.int64))

	def setWhole(self, whole) :
		self.prev_durr, self.prev_pitc, self.prev_tone = whole
	def normalize(self) :
		curr_pitc_min = self.curr_pitc[self.curr_pitc > 0].min()
		curr_pitc = self.curr_pitc - curr_pitc_min + 1
		curr_pitc_d = np.zeros((curr_pitc.shape[0], 6), dtype=np.int64)
		curr_pitc_d[1:, 0] = (curr_pitc[1:] - curr_pitc[:-1]) // 12 + 3
		curr_pitc_d[:-1, 1] = (curr_pitc[:-1] - curr_pitc[1:]) // 12 + 3
		curr_pitc_d[:, 2] = (curr_pitc - curr_pitc.min()) // 12 + 3
		curr_pitc_d[1:, 3] = (curr_pitc[1:] - curr_pitc[:-1]) % 12 + 11
		curr_pitc_d[:-1, 4] = (curr_pitc[:-1] - curr_pitc[1:]) % 12 + 11
		curr_pitc_d[:, 5] = (curr_pitc - curr_pitc.min()) % 12
		self.curr_pitc = np.concatenate([curr_pitc.reshape((-1, 1)), curr_pitc_d], axis=-1)
		
		prev_pitc_min = self.prev_pitc[self.prev_pitc > 0].min()
		prev_pitc = self.prev_pitc.copy()
		prev_pitc[self.prev_pitc > 0] = self.prev_pitc[self.prev_pitc > 0] - prev_pitc_min + 1
		prev_pitc_d = np.zeros((prev_pitc.shape[0], 6), dtype=np.int64)
		prev_pitc_d[1:, 0] = (prev_pitc[1:] - prev_pitc[:-1]) // 12 + 3
		prev_pitc_d[:-1, 1] = (prev_pitc[:-1] - prev_pitc[1:]) // 12 + 3
		prev_pitc_d[:, 2] = (prev_pitc - prev_pitc.min()) // 12 + 3
		prev_pitc_d[1:, 3] = (prev_pitc[1:] - prev_pitc[:-1]) % 12 + 11
		prev_pitc_d[:-1, 4] = (prev_pitc[:-1] - prev_pitc[1:]) % 12 + 11
		prev_pitc_d[:, 5] = (prev_pitc - prev_pitc.min()) % 12
		self.prev_pitc = np.concatenate([prev_pitc.reshape((-1, 1)), prev_pitc_d], axis=-1)

		assert self.curr_pitc.min() >= 0 and self.prev_pitc.min() >= 0, f"Pitch normalization error: pitch <= 0 found. {self.curr_pitc}, {self.prev_pitc}"
		assert self.curr_pitc[:, 0].max() <= MAX_PITCH and self.prev_pitc[:, 0].max() <= MAX_PITCH, f"Pitch normalization error: pitch > MAX_PITCH found. {self.curr_pitc}, {self.prev_pitc} {self.text}"
		assert self.curr_pitc[:, 1:4].max() < 6 and self.prev_pitc[:, 1:4].max() < 6, f"Pitch normalization error: pitch delta octave out of range. {self.curr_pitc}, {self.prev_pitc} {self.text}"
		assert self.curr_pitc[:, 4:].max() < 24 and self.prev_pitc[:, 4:].max() < 24, f"Pitch normalization error: pitch delta in octave out of range. {self.curr_pitc}, {self.prev_pitc} {self.text}"

	def __str__(self):
		return f"notes={self.curr_pitc} {self.curr_durr}, text={self.text}, tones={self.curr_tone})"
def parse_pitch(pit_str : str) -> int :
	assert 'r' not in pit_str and 'R' not in pit_str, "Rest note found in parse_pitch"
	main_pit = pit_str[0]
	shift_pit = pit_str[-1] if pit_str[-1] in ['#', '-'] else ''
	pit_str = pit_str.rstrip('#-')
	oct_shift = 0
	if len(pit_str) == 2 :
		if 'A' <= pit_str[0] <= 'G' :
			oct_shift = -12
		else :
			oct_shift = 12 * 2
	else :
		if 'a' <= pit_str <= 'g' :
			oct_shift = 12
	
	# use map to get pitch
	base_map = {
		'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71
	}
	shift_map = {
		'#': 1, '-': -1, '': 0
	}
	return base_map[main_pit.upper()] + shift_map[shift_pit] + oct_shift

print(parse_pitch('C'), parse_pitch('C#'), parse_pitch('G'), parse_pitch('A'), parse_pitch('B'), 
	  parse_pitch('c'), parse_pitch('a'), parse_pitch('g'), parse_pitch('gg'), parse_pitch('c'))

def parse_note(note_str : str, tot : int) -> tuple[int, int] :
	# remove any '[' ']' '(' ')' and 'P' and '.'
	remove_chr = ['[', ']', '(', ')', 'P', '.', '{', '}', '_', 'J', 'L', 'q']
	for ch in remove_chr :
		note_str = note_str.replace(ch, '')
	# get digital part
	spec_tot = -1
	if note_str.count('%') > 0 :
		pos = note_str.index('%') + 1
	else : pos = 0
	while pos < len(note_str) and note_str[pos].isdigit() :
		pos += 1
	if note_str.count('%') > 0 :
		dur = int(note_str[note_str.index('%') + 1 : pos])
		spec_tot = int(note_str[0 : note_str.index('%')])
	else :
		dur = int(note_str[0 : pos])
		spec_tot = tot
	dur = dur * tot / spec_tot
	pit = note_str[pos :]
	if dur == 0 :
		assert False, f"Duration is zero in note string: {note_str}"
	return (dur, pit)

def parse_sentence(tpls : list[tuple[str, str, str, int]]) -> DataItem :
	notes : list[list[tuple[int, int]]] = []
	texts : list[str] = []
	jyuts : list[str] = []
	for i in range(len(tpls)) :
		if tpls[i][1] == '.' :
			# append this notes to last element
			if len(notes) > 0 and tpls[i][0] != 'r' :
				notes[-1].append(parse_note(tpls[i][0], tpls[i][3]))
		else :
			# start a new element
			notes.append([parse_note(tpls[i][0], tpls[i][3])])
			texts.append(tpls[i][1])
			jyuts.append(tpls[i][2])
			if jyuts[-1] == '.' : 
				return None
	
	# merge save note for each element
	merge_notes = []
	for note_seq in notes :
		merged_seq = []
		cur_dur = 0
		cur_pit = None
		for dur, pit in note_seq :
			if pit == cur_pit :
				cur_dur += dur
			else :
				if cur_pit is not None :
					merged_seq.append((cur_dur, cur_pit))
				cur_dur = dur
				cur_pit = pit
		if cur_pit is not None :
			merged_seq.append((cur_dur, cur_pit))
		merged_seq = [(dur, parse_pitch(pit)) for dur, pit in merged_seq]
		merge_notes.append(merged_seq)
	return DataItem(merge_notes, texts, jyuts)

def parse_krn(file_path: str) -> list[DataItem] :
	doc : kp.Document
	doc, err = kp.load(file_path)
	tks = doc.get_all_tokens_encodings()
	st_kern, st_text, st_jyut = 0, 0, 0
	for i in range(len(tks)) :
		if tks[i] == '**kern' : st_kern = i
		elif tks[i] == '**text' : st_text = i
		elif tks[i] == '**jyutping' : st_jyut = i
	
	tpls : list[tuple[str, str, str, int]] = []
	cur_tot = 0
	for i in range(st_text - st_kern) :
		kern_pos = st_kern + i
		text_pos = st_text + i
		jyut_pos = st_jyut + i
		if tks[kern_pos].startswith('*M') and tks[kern_pos].find('/') != -1 :
			tot_str = tks[kern_pos][2 :].split('/')[-1]
			cur_tot = int(tot_str)
			# print("Set tot to", cur_tot)
			continue
		if tks[kern_pos].count('r') > 0 or tks[kern_pos].count('R') > 0 :
			# rest note
			# just skip
			continue
		tpls.append((tks[kern_pos], tks[text_pos], tks[jyut_pos], cur_tot))
	
	sentences : list[DataItem] = []
	cur_sentence = None
	for tpl in tpls :
		if tpl[0].startswith('*') :
			continue
		if tpl[0][0] == '{' :
			# start a sentence
			cur_sentence = []
		if cur_sentence is not None :
			if tpl[0] != '=' :
				cur_sentence.append(tpl + (cur_tot,))
		if tpl[0][-1] == '}' :
			# end a sentence
			if cur_sentence is not None :
				item = parse_sentence(cur_sentence)
				if item is not None :
					sentences.append(item)
				cur_sentence = None
	
	# get whole note info for each sentence
	whole_durr = []
	whole_pitc = []
	whole_tone = []
	for i in range(len(sentences)) :
		whole_durr.append(np.zeros(1, dtype=np.float32))
		whole_durr.append(sentences[i].curr_durr)
		whole_pitc.append(np.zeros(1, dtype=np.int64))
		whole_pitc.append(sentences[i].curr_pitc)
		whole_tone.append(np.array([TONE_VOCAB_SIZE], dtype=np.int64))
		whole_tone.append(sentences[i].curr_tone)
	
	for i in range(len(sentences)) :
		posL = max(0, (i - PREV_SIZE) * 2 + 1)
		posR = min(len(whole_durr), (i + PREV_SIZE) * 2 + 2)
		inte_durr = np.concatenate(whole_durr[posL : posR], axis=0)
		inte_pitc = np.concatenate(whole_pitc[posL : posR], axis=0)
		inte_tone = np.concatenate(whole_tone[posL : posR], axis=0)
		sentences[i].setWhole((inte_durr, inte_pitc, inte_tone))
		sentences[i].normalize()
		
	return sentences

# assume this note is on C4 (MIDI 60)
def parse_notelist(notes : list[str]) -> np.array :
	note_list = []
	for note_str in notes :
		note_list.append([(1, parse_pitch(note_str))])
	return np.array(note_list)
	
if __name__ == "__main__" :
	items = os.listdir("./dataset/Humdrum-files/")
	sentenses = []
	for item in tqdm(items) :
		if item.endswith(".krn") :
			sentenses.extend(parse_krn(f"./dataset/Humdrum-files/{item}"))