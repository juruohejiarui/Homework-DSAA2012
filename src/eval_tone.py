import torch
import models
import data
import numpy as np

if __name__ == "__main__":
	print("Eval Tone Module")
	
	model = models.ToneModel(data.TONE_VOCAB_SIZE)
	model.load_state_dict(torch.load("./ckpts/tone_model_final.pth"))
	model.eval()
	
	model = model.to(models.device)

	curr_notes = ['c', 'c', 'B', 'B', 'G', 'C', 'G'] # 她说 1 should be 3 3 3 3 4 0 3/4
	# curr_notes = ['c', 'c', 'B', 'B', 'G', 'C', 'A'] # 她说 2 should be 3 3 3 3 4 0 3
	# curr_notes = ['c', 'c', 'B', 'B', 'c', 'd', 'd'] # 她说 3 should be 4 4 2 2 4 3 3
	# curr_notes = ['c', 'c', 'B', 'B', 'c', 'd', 'd', 'c', 'e'] # 她说 4 should be 4 4 2 2 4 3 3 4 3
	# curr_notes = ['d', 'e', 'f', 'e', 'f', 'e', 'f'] # 她说 5 should be 0 2 4 2 4 2 4
	# curr_notes = ['B', 'A', 'G', 'A', 'B'] # 她说 6 should be 3 3 4 3 3
	# curr_notes = ['e', 'f', 'g', 'c', 'b'] # 她说 7 should be 2 4 3 0/4 3
	# curr_notes = ['d', 'e', 'f', 'e', 'f', 'e', 'f', 'g', 'e'] # 她说 7 should be 0 2 4 2 4 2 4 3 2/3/4
	# curr_notes = ['E', 'D', 'C', 'G'] # 开不了口 1 # should be 4 2 0 3
	# curr_notes = ['E', 'G', 'B', 'A', 'B', 'A', 'A'] # 开不了口 2 # should be 0/2 4 3 4 3 (4 4)/(3 3)
	# curr_notes = ['A', 'B', 'A', 'A', 'A', 'A', 'A', 'G'] # 开不了口 3 # should be 4 3 3 3 3 3 3 3/4
	# curr_notes = ['D', 'D', 'F', 'G', 'C', 'C', 'C', 'C'] # 开不了口 4 # should be (0 0)/(2 2) 4 3 (4 4 4 4)/(0 0 0 0)
	# curr_notes = ['E', 'G', 'B', 'A', 'B', 'B', 'B'] # 开不了口 5 # should be 0/2 4 3 3 (3 3)/(4 4)
	# curr_notes = ['A', 'A', 'G', 'G', 'A', 'G'] # 开不了口 6 should be 3 3 (4 4)/(3 3) 3 4/3 or 2 2 0 0 2 0
	# curr_notes = ['D', 'D', 'D', 'D', 'D', 'C'] # 开不了口 6
	# curr_notes = ['G', 'A', 'c', 'd', 'e', 'e', 'd', 'e'] # 千千阙歌 1 should be 0 2 4 3 3 4 3
	# curr_notes = ['e', 'd', 'c', 'd', 'c', 'A', 'A'] # 千千阙歌 2 should be 3 3 4 3 3 2 2
	# curr_notes = ['D', 'D', 'D', 'E', 'F'] # 千千阙歌 3 should be 0 0 0 2 4
	# curr_notes = ['G', 'A', 'c', 'B', 'B', 'B', 'G', 'E'] # 千千阙歌 4 should be 0 2 3 3 3 3 4 2/0
	# curr_notes = ['d', 'c', 'd', 'c', 'A', 'c', 'c'] # 千千阙歌 5 should be 3 4 3 4 2 4 4
	# curr_notes = ['e', 'e', 'd', 'd', 'c', 'd', 'c', 'A'] # 千千阙歌 6 should be 3 3 3 3 4 3 (4 2)/(4 0)/(3 2)
	# curr_notes = ['G', 'A', 'B', 'B', 'A', 'B'] # 千千阙歌 7 should be 4 4 3 3 4 3 / 0 0 2 2 0 2
	# curr_notes = ['C', 'C', "G", 'G', 'A', 'A', 'G'] # 小星星 1 should be 0 0 4 4 3 3 4
	# curr_notes = ['F', 'F', "E", 'E', 'D', 'D', 'C'] # 小星星 2 should be 3 3 3 3 3 3 4/3
	# curr_notes = ['G', 'c', 'B', 'c'] # 必杀技 1 should be 0 4 2 4
	# curr_notes = ['C', 'D', 'E', 'G', 'G'] # 必杀技 2 should be 0 0 2 (4 4)/(3 3)
	# curr_notes = ['G', 'c', 'c', 'B', 'B', 'A', 'E', 'A'] # 小幸运 1 should be 2/0 3 3 3 3 (3 0 4)/(4 0 4)(3 2 3)
	# curr_notes = ['B', 'e', 'e', 'B', 'B', 'G', 'E', 'G'] # 小幸运 2 should be 2 3 3 3 3 (4 0 4/0)/(3 2 3/4)/(4 2 4)
	# curr_notes = ['e', 'G', 'd', 'e', 'G', 'd', 'e'] # 小幸运 3 should be 3 0 4/3 3 0 4/3 3
	# curr_notes = ['e', 'c', 'c', 'e', 'd', 'c'] # 小幸运 4 should be 3 4 4 3 3 4
	curr_jyuts = ['1' for _ in range(len(curr_notes))]
	curr_pitc = data.parse_notelist(curr_notes)
	print("Input pitches:", curr_pitc[:, 0, 1].tolist())
	
	item = data.DataItem(curr_pitc, "x", curr_jyuts)
	item.setWhole((item.curr_durr, item.curr_pitc, np.zeros((len(item.curr_durr)))))
	item.normalize()
	curr_durr = torch.tensor(item.curr_durr, dtype=torch.float32).unsqueeze(0).to(models.device)
	curr_pitc = torch.tensor(item.curr_pitc, dtype=torch.int32).unsqueeze(0).to(models.device)
	curr_mask = torch.ones(item.curr_tone.shape, dtype=torch.bool).unsqueeze(0).to(models.device)
	logits = model(
		curr_durr, curr_pitc, None,
		curr_durr, curr_pitc,
		curr_mask, curr_mask,
		)
	map0243 = {0: 0, 1: 2, 2: 4, 3: 3}
	preds = logits.argmax(dim=-1)
	print("Predicted tones:", preds)
	print("Predicted Probs:", torch.softmax(logits, dim=-1))
	print("Mapped tones:", [map0243[pred.item()] for pred in preds[0]])
	

