import torch
import torch.nn as nn
import torch.nn.functional as F
		

class EmbedLSTM(nn.Module) :
	def __init__ (self, tone_vocab_size : int,
			   pitc_embed_size : int = 128, tone_embed_size : int = 32,
			   lstm_hidden_size : int = 512, lstm_layers : int = 2, 
			   fc_hidden_size : int = 512,
			   dropout : float = 0.5) :
		super(EmbedLSTM, self).__init__()
		self.tone_emb = nn.Embedding(tone_vocab_size + 1, tone_embed_size, padding_idx=0)
		# self.pitc_emb = nn.Sequential(
			# nn.Linear(1, pitc_embed_size),
			# nn.RReLU()
		# )
		self.pitc_emb = nn.Embedding(  # pitch embedding
			128 + 1, pitc_embed_size, padding_idx=0)  #

		self.prev_lstm = nn.LSTM(input_size=pitc_embed_size * 2 + tone_embed_size, 
						  hidden_size=lstm_hidden_size, 
						  num_layers=lstm_layers, 
						  dropout=dropout,
						  batch_first=True,
						  bidirectional=True)
		self.attn = nn.MultiheadAttention(embed_dim=lstm_hidden_size * 2,
											 num_heads=8,
											 dropout=dropout,
											 batch_first=True)
		self.curr_lstm = nn.LSTM(input_size=pitc_embed_size * 2,
						  hidden_size=lstm_hidden_size,
						  num_layers=lstm_layers,
						  dropout=dropout,
						  batch_first=True,
						  bidirectional=True)
		
		
	
		self.fc = nn.Sequential(
			nn.Linear(lstm_hidden_size * 2, fc_hidden_size),
			nn.RReLU(),
			nn.Linear(fc_hidden_size, tone_vocab_size),
			# nn.RReLU()
		)
	
	def forward(self, 
			 prev_durr : torch.Tensor, 
			 prev_pitc : torch.Tensor, 
			 prev_tone : torch.Tensor, 
			 curr_durr : torch.Tensor, 
			 curr_pitc : torch.Tensor, 
			 prev_mask : torch.Tensor, 
			 curr_mask : torch.Tensor) :
		B, currL, _ = curr_durr.shape
		_, prevL, _ = prev_durr.shape

		# print("prev_note:", prev_note.shape)
		# print("curr_note:", curr_note.shape)

		# curr_pitc_st = self.pitc_emb(curr_pits[:, :, 0].unsqueeze(-1) / 128)
		# curr_pitc_ed = self.pitc_emb(curr_pits[:, :, 1].unsqueeze(-1) / 128)
		curr_pitc_st = self.pitc_emb(curr_pitc[:, :, 0])
		curr_pitc_ed = self.pitc_emb(curr_pitc[:, :, 1])

		# normalize duration for each sequence
		curr_durr = curr_durr / (curr_durr.sum(dim=2, keepdim=True) + 1e-8)
		curr_emb = torch.cat([curr_pitc_st * curr_durr[:, :, 0].unsqueeze(-1), curr_pitc_ed * curr_durr[:, :, 1].unsqueeze(-1)], dim=-1)

		curr_lstm_in = nn.utils.rnn.pack_padded_sequence(curr_emb, 
												   lengths=curr_mask.sum(dim=1).cpu(),
												   batch_first=True,
												   enforce_sorted=False)

		curr_lstm_out, _ = self.curr_lstm(curr_lstm_in)
		curr_lstm_out, _ = nn.utils.rnn.pad_packed_sequence(curr_lstm_out, batch_first=True)

		prev_tone_emb = self.tone_emb(prev_tone)
		# prev_pitc_st = self.pitc_emb(prev_pitc[:, :, 0].unsqueeze(-1) / 128)
		# prev_pitc_ed = self.pitc_emb(prev_pitc[:, :, 1].unsqueeze(-1) / 128)
		prev_pitc_st = self.pitc_emb(prev_pitc[:, :, 0])
		prev_pitc_ed = self.pitc_emb(prev_pitc[:, :, 1])

		prev_durr = prev_durr / (prev_durr.sum(dim=2, keepdim=True) + 1e-8)
		# multiply pitch embedding with duration
		prev_emb = torch.cat([prev_pitc_st * prev_durr[:, :, 0].unsqueeze(-1), prev_pitc_ed * prev_durr[:, :, 1].unsqueeze(-1), prev_tone_emb], dim=-1)

		prev_lstm_in = nn.utils.rnn.pack_padded_sequence(prev_emb,
												   lengths=prev_mask.sum(dim=1).cpu(),
												   batch_first=True,
												   enforce_sorted=False)
		prev_lstm_out, _ = self.prev_lstm(prev_lstm_in)
		prev_lstm_out, _ = nn.utils.rnn.pad_packed_sequence(prev_lstm_out, batch_first=True)
		# attn_out: (B, currL, H)	
		attn_out, _ = self.attn(curr_lstm_out, prev_lstm_out, prev_lstm_out,
									key_padding_mask=(prev_mask == 0))
		# multiply the two outputs
		# (B, currL, H) * (B, currL, H) -> (B, currL, H)
		feat = F.rrelu(curr_lstm_out) * attn_out
		
		return self.fc(feat)