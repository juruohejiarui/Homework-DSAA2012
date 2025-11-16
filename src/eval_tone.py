import torch
import models
import data

if __name__ == "__main__":
	print("Eval Tone Module")
	
	model = models.ToneModel(data.TONE_VOCAB_SIZE)
	model.load_state_dict(torch.load("./ckpts/best_tone_model.pth"))
	model.eval()
	
	model = model.to(models.device)
	
	# prev_notes = [[5,5], [8,8], [7,7], [8,8], [0, 0], [1, 1], [1, 1], [1, 1]]
	# prev_tones = [0, 4, 3, 4, -1, 0, 0, 0]
	# prev_notes = [[5,5], [8,8], [7,7], [8,8], [0, 0]]
	# prev_tones = [0, 4, 3, 4, -1]
	# prev_notes = [[5,5], [8,8], [7,7], [8,8], [0, 0]]
	# prev_tones = [2, 1, 3, 1, 4]
	# prev_notes = [[1, 1], [1, 1], [5, 5], [5, 5], [6, 6], [6, 6], [5, 5], [0, 0]]
	# prev_tones = [0, 0, 4, 4, 3, 3, 4, -1]
	prev_notes = [[0, 0]]
	prev_tones = [-1]
	map0243 = [3, 4, 0, 2]

	# convert to indice of model
	prev_tones = [map0243.index(tone) if tone in map0243 else 4 for tone in prev_tones]
	print("Prev tones mapped:", prev_tones)
	valid_indices = [i for i in range(len(prev_tones)) if prev_tones[i] != 4]

	_, C_pitch = data.parse_note('4C', 1)
	C_pitch = data.parse_pitch(C_pitch)
	for i in range(len(valid_indices)) :
		idx = valid_indices[i]
		prev_notes[idx] = [(note - 1) * 2 + C_pitch for note in prev_notes[idx]]
	print("Prev notes mapped:", prev_notes)
	# prev_notes = [[0, 0]]
	# prev_tones = [4]
	prev_durr = torch.ones((1, len(prev_notes), 2), dtype=torch.float32).repeat(2, 1, 1).to(models.device)
	prev_pitc = (torch.tensor(prev_notes)).unsqueeze(0).repeat(2, 1, 1).to(models.device)
	prev_tone = torch.tensor(prev_tones).repeat(2, 1).to(models.device)

	# curr_notes = [5, 6, 8, 9, 10, 10, 9, 10] # 千千阙歌 1
	# curr_notes = [10, 9, 8, 9, 9, 6, 6] # 千千阙歌 2
	# curr_notes = [1, 5, 6, 7, 5] # 必杀技 2
	# curr_notes = [1, 1, 1, 1, 5, 6, 7, 5] # 必杀技 4
	curr_notes = [1, 1, 5, 5, 6, 6, 5] # 小星星 1
	# curr_notes = [5, 8, 7, 8] # 必杀技 1
	# curr_notes = [4, 4, 3, 3, 2, 2, 1] # 小星星 2
	# curr_notes = [1, 2, 3, 5, 5] # 必杀技 3
	# curr_notes = [1, 2, 3, 3, 2, 3, 2, 1, 0, -1] # 明年今日 1
	# curr_notes = [1, 2, 4, 6] # 明年今日 2
	curr_durr = torch.ones((1, len(curr_notes), 2), dtype=torch.float32).repeat(2, 1, 1).to(models.device)
	curr_pitch = torch.from_numpy(data.parse_notelist(curr_notes)).unsqueeze(0).repeat(2, 1, 1).to(models.device)
	print("Curr notes mapped: ", curr_pitch.tolist())
	prev_mask = torch.ones((1, len(prev_notes)), dtype=torch.bool).repeat(2, 1).to(models.device)
	curr_mask = torch.ones((1, len(curr_notes)), dtype=torch.bool).repeat(2, 1).to(models.device)

	print("shapes: ", curr_pitch.shape, curr_durr.shape, prev_pitc.shape, prev_durr.shape, prev_tone.shape)

	logits = model(prev_durr, prev_pitc, prev_tone, curr_durr, curr_pitch, prev_mask, curr_mask)
	
	
	preds = torch.argmax(logits, dim=-1)
	print("Predicted tones:", preds)
	print("Mapped tones:", [map0243[pred.item()] for pred in preds[0]])
	

