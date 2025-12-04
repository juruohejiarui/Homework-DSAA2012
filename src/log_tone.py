import tensorboard
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import data
import models
import torch
import torch.nn as nn
import train_tone

model_path = "ckpts/tone_model_best.pth"

model = models.ToneModel(data.TONE_VOCAB_SIZE)
model.load_state_dict(torch.load(model_path))
model.eval()

model.to(models.device)

val_dataset = data.ToneDataset(data_dir="dataset/Cantopop-corpus/Humdrum-files", split=train_tone.split_rate, train=False)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=data.tone_collate_fn)

train_dataset = data.ToneDataset(data_dir="dataset/Cantopop-corpus/Humdrum-files", split=train_tone.split_rate, train=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=data.tone_collate_fn)

report = train_tone.eval_epoch(model, val_loader, nn.CrossEntropyLoss())
report_train = train_tone.eval_epoch(model, train_loader, nn.CrossEntropyLoss())
confu_mtx = train_tone.eval_confusion_matrix(model, val_loader)
confu_mtx_train = train_tone.eval_confusion_matrix(model, train_loader)

with open("logs/tone_evaluation.txt", "w") as f:
	f.write("Evaluation Report:\n")
	f.write(str(report))
	f.write("\nConfusion Matrix:\n")
	f.write(str(confu_mtx))

	f.write("\n\nTraining Set Report:\n")
	f.write(str(report_train))
	f.write("\nTraining Set Confusion Matrix:\n")
	f.write(str(confu_mtx_train))