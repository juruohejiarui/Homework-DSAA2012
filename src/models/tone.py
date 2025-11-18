import torch
import torch.nn as nn
import torch.nn.functional as F
import TorchCRF as torchcrf
import data

class EmbedLSTM(nn.Module) :
	def __init__ (self, tone_vocab_size : int,
			   pitc_embed_size : int = 32,
			   lstm_hidden_size : int = 512, lstm_layers : int = 2, 
			   fc_hidden_size : int = 512,
			   dropout : float = 0.5) :
		super(EmbedLSTM, self).__init__()
		self.pitc_emb = nn.Embedding(  # pitch embedding
			data.MAX_PITCH + 1, pitc_embed_size, padding_idx=0)  #
		self.pitc_d_oct_emb = nn.Embedding(  # pitch delta in octave embedding
			6, pitc_embed_size, padding_idx=0)  #
		self.pitc_d_inoct_emb = nn.Embedding(  # pitch delta mod12 embedding
			24, pitc_embed_size, padding_idx=0)  #

		self.prev_lstm = nn.LSTM(input_size=pitc_embed_size * 7, 
						  hidden_size=lstm_hidden_size, 
						  num_layers=lstm_layers,
						  dropout=dropout,
						  batch_first=True,
						  bidirectional=True)
		self.attn = nn.MultiheadAttention(embed_dim=lstm_hidden_size * 2,
											 num_heads=2,
											 dropout=dropout,
											 batch_first=True)
		self.curr_lstm = nn.LSTM(input_size=pitc_embed_size * 7,
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

		self.crf = torchcrf.CRF(tone_vocab_size)
	
	def forward(self, 
			 prev_durr : torch.Tensor, 
			 prev_pitc : torch.Tensor,
			 prev_tone : torch.Tensor, 
			 curr_durr : torch.Tensor, 
			 curr_pitc : torch.Tensor, 
			 prev_mask : torch.Tensor, 
			 curr_mask : torch.Tensor,
			 targets : torch.Tensor = None) -> torch.Tensor :
		# print("prev_note:", prev_note.shape)
		# print("curr_note:", curr_note.shape)

		# curr pitch embedding		
		curr_pitc_emb = self.pitc_emb(curr_pitc[:, :, 0])
		curr_pitc_d_oct_1 = self.pitc_d_oct_emb(curr_pitc[:, :, 1])
		curr_pitc_d_oct_2 = self.pitc_d_oct_emb(curr_pitc[:, :, 2])
		curr_pitc_d_oct_3 = self.pitc_d_oct_emb(curr_pitc[:, :, 3])
		curr_pitc_d_inoct_1 = self.pitc_d_inoct_emb(curr_pitc[:, :, 4])
		curr_pitc_d_inoct_2 = self.pitc_d_inoct_emb(curr_pitc[:, :, 5])
		curr_pitc_d_inoct_3 = self.pitc_d_inoct_emb(curr_pitc[:, :, 6])
		curr_pitc_input = torch.cat([
			curr_pitc_emb,
			curr_pitc_d_oct_1,
			curr_pitc_d_oct_2,
			curr_pitc_d_oct_3,
			curr_pitc_d_inoct_1,
			curr_pitc_d_inoct_2,
			curr_pitc_d_inoct_3,
		], dim=-1)  # (B, currL, pitc_embed_size * 7)

		curr_lstm_in = nn.utils.rnn.pack_padded_sequence(curr_pitc_input, 
												   lengths=curr_mask.sum(dim=1).cpu(),
												   batch_first=True)

		curr_lstm_out, _ = self.curr_lstm(curr_lstm_in)
		curr_lstm_out, _ = nn.utils.rnn.pad_packed_sequence(curr_lstm_out, batch_first=True)

		# calculate prev embedding
		prev_pitc_emb = self.pitc_emb(prev_pitc[:, :, 0])
		prev_pitc_d_oct_1 = self.pitc_d_oct_emb(prev_pitc[:, :, 1])
		prev_pitc_d_oct_2 = self.pitc_d_oct_emb(prev_pitc[:, :, 2])
		prev_pitc_d_oct_3 = self.pitc_d_oct_emb(prev_pitc[:, :, 3])
		prev_pitc_d_inoct_1 = self.pitc_d_inoct_emb(prev_pitc[:, :, 4])
		prev_pitc_d_inoct_2 = self.pitc_d_inoct_emb(prev_pitc[:, :, 5])
		prev_pitc_d_inoct_3 = self.pitc_d_inoct_emb(prev_pitc[:, :, 6])
		prev_pitc_input = torch.cat([
			prev_pitc_emb,
			prev_pitc_d_oct_1,
			prev_pitc_d_oct_2,
			prev_pitc_d_oct_3,
			prev_pitc_d_inoct_1,
			prev_pitc_d_inoct_2,
			prev_pitc_d_inoct_3,
		], dim=-1)  # (B, prevL, pitc_embed_size * 7)

		prev_lstm_in = nn.utils.rnn.pack_padded_sequence(prev_pitc_input,
												   lengths=prev_mask.sum(dim=1).cpu(),
												   batch_first=True,
												   enforce_sorted=False)
		prev_lstm_out, _ = self.curr_lstm(prev_lstm_in)
		prev_lstm_out, _ = nn.utils.rnn.pad_packed_sequence(prev_lstm_out, batch_first=True)
		# attn_out: (B, currL, H)	
		attn_out, _ = self.attn(curr_lstm_out, prev_lstm_out, prev_lstm_out,
									key_padding_mask=(prev_mask == 0))
		# multiply the two outputs
		# (B, currL, H) * (B, currL, H) -> (B, currL, H)
		feat = F.rrelu(curr_lstm_out) * attn_out
		
		logits : torch.Tensor = self.fc(feat)  # (B, currL, tone_vocab_size)

		return logits