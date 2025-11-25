import argparse
import torch
import data
import models
import os
from functools import partial
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
	AutoTokenizer,
	AutoModelForCausalLM,
	TrainingArguments,
	Trainer,
	get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from time import sleep

def setup_model_and_tokenizer(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float, use_bf16: bool = True):
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token = tokenizer.eos_token

	dtype = torch.bfloat16 if use_bf16 else None
	print(f"> Loading model {model_name} with dtype={dtype}")
	model = AutoModelForCausalLM.from_pretrained(
		model_name,
		type=dtype,
		device_map="auto",
		trust_remote_code=True,
		low_cpu_mem_usage=True,
	)

	lora_config = LoraConfig(
		r=lora_r,
		lora_alpha=lora_alpha,
		target_modules=None,
		lora_dropout=lora_dropout,
		bias="none",
		task_type=TaskType.CAUSAL_LM,
	)
	model = get_peft_model(model, lora_config)
	return model, tokenizer

def eval(model, tokenizer, val_loader):
	model.eval()
	total_loss = 0.0
	total_count = 0
	with torch.no_grad():
		for batch in val_loader:
			batch = {k: v.to(models.device) for k, v in batch.items()}
			outputs = model(**batch)
			loss = outputs.loss
			batch_size = batch['input_ids'].size(0)
			total_loss += loss.item() * batch_size
			total_count += batch_size
	avg_loss = total_loss / total_count
	model.train()
	return avg_loss

# -------------------------
# 训练函数
# -------------------------
def train(
	model,
	tokenizer,
	dataset_path: str,
	output_dir: str,
	max_length: int,
	batch_size: int,
	epochs: int,
	lr: float,
	weight_decay: float,
	warmup_steps: int,
	gradient_accumulation_steps: int,
	val_size : float = 0.1,
	val_log_steps : int = 500,
	cold_down_steps : int = 100,
):
	os.makedirs(output_dir, exist_ok=True)

	dataset = data.SFTDataset(dataset_path)
	# split dataset
	val_size = int(len(dataset) * val_size)
	train_size = len(dataset) - val_size
	train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
	print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
	
	collate = partial(data.sft_collate_fn, tokenizer=tokenizer, max_length=max_length)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

	try:
		model.to(models.device)
	except Exception:
		pass  # 如果模型使用了 device_map 则跳过

	# optimizer only for trainable params (LoRA)
	optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
	total_steps = max(1, (len(train_loader) // gradient_accumulation_steps) * epochs)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

	# 训练 loop
	model.train()
	global_step = 0
	use_bf16_autocast = True if next(model.parameters()).dtype == torch.bfloat16 else False

	for epoch in range(epochs):
		pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
		running_loss = 0.0
		for step, batch in enumerate(pbar):
			batch = {k: v.to(models.device) for k, v in batch.items()}

			# 前向：在 bf16 下用 autocast
			if models.device == "cuda" and use_bf16_autocast:
				with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
					outputs = model(**batch)
					loss = outputs.loss
			else:
				outputs = model(**batch)
				loss = outputs.loss

			loss = loss / gradient_accumulation_steps
			loss.backward()
			running_loss += loss.item()

			if (step + 1) % gradient_accumulation_steps == 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()
				global_step += 1
				pbar.set_postfix({"train": running_loss / max(1, global_step)})
				
			if (global_step % val_log_steps) == 0:
				val_loss = eval(model, tokenizer, val_loader)
				tqdm.write(f"Step {global_step}: val_loss = {val_loss:.4f}")

			# cold down GPU
			if models.device == "cuda" and (step + 1) % cold_down_steps == 0:
				sleep(20)
				

		# epoch 结束保存
		print(f"Epoch {epoch+1} done. Saving to {output_dir}")
		model.save_pretrained(output_dir)
		tokenizer.save_pretrained(output_dir)

	print("Training finished. Final save to", output_dir)
	model.save_pretrained(output_dir)
	tokenizer.save_pretrained(output_dir)


# -------------------------
# CLI 入口（只需在此设置 / 通过命令行传参）
# -------------------------
def parse_args():
	p = argparse.ArgumentParser()
	# 必要项：模型名、数据、输出目录
	p.add_argument("--model_name", type=str, default="pretrained_models/Qwen3-4B-Instruct-2507", help="HF 模型名或本地路径")
	p.add_argument("--data_file", type=str, default="dataset/sft.json", help="JSON 数据文件路径（数组格式）")
	p.add_argument("--output_dir", type=str, default="ckpts/sft_lora", help="保存目录")

	# 训练参数（常改）
	p.add_argument("--max_length", type=int, default=512)
	p.add_argument("--batch_size", type=int, default=8)
	p.add_argument("--epochs", type=int, default=2)
	p.add_argument("--lr", type=float, default=2e-5)
	p.add_argument("--weight_decay", type=float, default=0.0)
	p.add_argument("--warmup_steps", type=int, default=50)
	p.add_argument("--gradient_accumulation_steps", type=int, default=1)

	# LoRA 参数（可改）
	p.add_argument("--lora_r", type=int, default=16)
	p.add_argument("--lora_alpha", type=int, default=16)
	p.add_argument("--lora_dropout", type=float, default=0.05)

	# 设备 / dtype
	p.add_argument("--device", type=str, default="cuda")
	p.add_argument("--use_bf16", default=True, action="store_true", help="尝试使用 bf16 加载模型（需硬件支持）")

	return p.parse_args()


if __name__ == "__main__":
	args = parse_args()

	model, tokenizer = setup_model_and_tokenizer(
		model_name=args.model_name,
		lora_r=args.lora_r,
		lora_alpha=args.lora_alpha,
		lora_dropout=args.lora_dropout,
		use_bf16=args.use_bf16,
	)

	# 仅需设置数据路径与训练参数（其余逻辑已封装）
	train(
		model=model,
		tokenizer=tokenizer,
		dataset_path=args.data_file,
		output_dir=args.output_dir,
		max_length=args.max_length,
		batch_size=args.batch_size,
		epochs=args.epochs,
		lr=args.lr,
		weight_decay=args.weight_decay,
		warmup_steps=args.warmup_steps,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
	)