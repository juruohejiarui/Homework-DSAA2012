import kernpy as kp
import os
import numpy as np
from tqdm import tqdm

MAX_PITCH = 46
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

def to0243(tones : np.ndarray) -> np.ndarray :
	map0243 = np.ones(tones.shape, dtype=np.int64) * TONE_VOCAB_SIZE
	tones = np.array([map_9to6[str(tone)] for tone in tones])
	# 3
	map0243[tones == 1] = 0
	map0243[tones == 2] = 0
	# 4
	map0243[tones == 3] = 1
	map0243[tones == 5] = 1
	# 0
	map0243[tones == 4] = 2
	# 2
	map0243[tones == 6] = 3
	if max(map0243) == TONE_VOCAB_SIZE :
		print("Warning: tone outside 1-6 found:", tones, " mapped to", map0243)
	return map0243

class DataItem:
	def __init__(self, note : list[list[int, str]], text : list[str], jyut : list[str], prev : tuple[np.ndarray, np.ndarray, np.ndarray]) :
		self.curr_note : np.ndarray = []
		self.text = text
		self.curr_tone : np.ndarray = []
		
		for note_seq in note :
			if len(note_seq) == 1 :
				self.curr_note.append((note_seq[0][0] / 2, note_seq[0][0] / 2, note_seq[0][1], note_seq[0][1]))
			elif len(note_seq) > 2 :
				# get max dur and second max dur
				sorted_seq = sorted(note_seq, key=lambda x: x[0], reverse=True)
				self.curr_note.append((sorted_seq[0][0], sorted_seq[1][0], sorted_seq[0][1], sorted_seq[1][1]))
			else :
				self.curr_note.append((note_seq[0][0], note_seq[1][0], note_seq[0][1], note_seq[1][1]))

		self.curr_note = np.array([(dur1, dur2, pits, pite) for dur1, dur2, pits, pite in self.curr_note])
		self.curr_durr = np.array([[dur1, dur2] for dur1, dur2, _, _ in self.curr_note], dtype=np.float32)
		self.curr_pitc = np.array([[pits, pite] for _, _, pits, pite in self.curr_note], dtype=np.int64)

		self.curr_tone = to0243(np.array([int(jyutone[-1]) for jyutone in jyut], dtype=np.int64))

		self.prev_durr, self.prev_pitc, self.prev_tone = prev
	
	def getConcatPrev(self) -> tuple[np.ndarray, np.ndarray] :
		# give up the first segement if too long
		if PREV_SIZE == 0 :
			return self.prev_durr, self.prev_pitc, self.prev_tone
		if (self.prev_tone == TONE_VOCAB_SIZE).sum() > PREV_SIZE :
			pos = 0
			while self.prev_tone[pos] != TONE_VOCAB_SIZE :
				pos += 1
			return \
				np.concatenate([self.prev_durr[pos + 1 : ], np.array([[0, 0]]), self.curr_durr], axis=0), \
				np.concatenate([self.prev_pitc[pos + 1 : ], np.array([[0, 0]]), self.curr_pitc], axis=0), \
				np.concatenate([self.prev_tone[pos + 1 : ], np.array([4]), self.curr_tone], axis=0)
		else :
			return \
				np.concatenate([self.prev_durr, np.array([[0, 0]]), self.curr_durr], axis=0), \
				np.concatenate([self.prev_pitc,  np.array([[0, 0]]), self.curr_pitc], axis=0), \
				np.concatenate([self.prev_tone, np.array([4]), self.curr_tone], axis=0)
	
	def __str__(self):
		return f"notes={self.curr_note}, text={self.text}, tones={self.curr_tone})"
	
def parse_pitch(pit_str : str) -> int :
	shift = 0
	if pit_str.endswith('-') : shift = -1
	elif pit_str.endswith('+') : shift = 1
	base_pit_str = pit_str.rstrip('+-')
	def toreal(x : int) :
		return x - 2 if x >= 2 else x + 5
	if ord(base_pit_str[0]) >= ord('A') and ord(base_pit_str[0]) <= ord('G') :
		base_pit = toreal(ord(base_pit_str[0]) - ord('A')) + 1
	elif ord(base_pit_str[0]) >= ord('a') and ord(base_pit_str[0]) <= ord('g') :
		base_pit = toreal(ord(base_pit_str[0]) - ord('a')) + 1 + 7
	else :
		assert False, f"Invalid pitch string: {pit_str}"
	if len(base_pit_str) > 1 :
		# for 'a' to 'g' , shift by 8
		# for 'A' to 'G' , shift by -7
		if ord(base_pit_str[0]) >= ord('a') and ord(base_pit_str[0]) <= ord('g') :
			shift_base = 7
		else :
			shift_base = -7
	else :
		shift_base = 0
	return (shift_base + base_pit) * 2 + shift + 66

print(parse_pitch('C'), parse_pitch('C+'), parse_pitch('G'), parse_pitch('A'), parse_pitch('B'), 
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

def parse_sentence(tpls : list[tuple[str, str, str, int]], prev : tuple) -> DataItem :
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
	return DataItem(merge_notes, texts, jyuts, prev)

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
	
	sentences = []
	cur_sentence = None
	prev_item : DataItem = None
	for tpl in tpls :
		note = tpl[0].removeprefix('{').removesuffix('}').removeprefix('.')
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
				if prev_item is None :
					prev_durr, prev_pitc, prev_tone = np.array([[0, 0]], dtype=np.float32), np.array([[0, 0]], dtype=np.int64), np.array([4], dtype=np.int64)
				else :
					prev_durr, prev_pitc, prev_tone = prev_item.getConcatPrev()
				item = parse_sentence(cur_sentence, (prev_durr, prev_pitc, prev_tone))
				if item is not None :
					sentences.append(item)
					prev_item = item
				cur_sentence = None

	return sentences

# assume this note is on C4 (MIDI 60)
def parse_notelist(notes : list[int]) -> np.array :
	note_list = []
	base = parse_pitch('C')
	for note in notes :
		note_pitch = (note - 1) * 2 + base
		note_list.append(note_pitch)
	return np.array([note_list, note_list], dtype=np.int64).T
	
if __name__ == "__main__" :
	items = os.listdir("./dataset/Humdrum-files/")
	sentenses = []
	for item in tqdm(items) :
		if item.endswith(".krn") :
			sentenses.extend(parse_krn(f"./dataset/Humdrum-files/{item}"))