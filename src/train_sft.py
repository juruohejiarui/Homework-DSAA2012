import argparse
import torch
import data
import models
import os
from functools import partial
from torch.utils.data import DataLoader
from transformers import (
	AutoTokenizer,
	AutoModelForCausalLM,
	get_linear_schedule_with_warmup
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig

def setup_model_and_tokenizer(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float, use_bf16: bool = True):
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token = tokenizer.eos_token

	dtype = torch.bfloat16 if use_bf16 else None
	print(f"> Loading model {model_name} with dtype={dtype}")
	model = AutoModelForCausalLM.from_pretrained(
		model_name,
		dtype=dtype,
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

	# 打印可训练参数
	tot = 0
	trainable = 0
	for name, param in model.named_parameters():
		num_param = param.numel()
		tot += num_param
		if param.requires_grad:
			trainable += num_param
	print(f"Total parameters: {tot}, Trainable parameters: {trainable}, Ratio: {trainable / tot:.6f}")
	return model, tokenizer

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
	val_log_steps : int = 10,
	cold_down_steps : int = 100,
):
	os.makedirs(output_dir, exist_ok=True)

	train_dataset = load_dataset("json", data_files=dataset_path[0])
	eval_dataset = load_dataset("json", data_files=dataset_path[1])


	cfg = SFTConfig(
		bf16=True,
		per_device_eval_batch_size=batch_size,
		per_device_train_batch_size=batch_size,
		gradient_accumulation_steps=gradient_accumulation_steps,
		learning_rate=lr,
		num_train_epochs=epochs,
		lr_scheduler_type='cosine',
		warmup_steps=warmup_steps,
		weight_decay=weight_decay,
		max_grad_norm=0.3,
		logging_steps=val_log_steps,
		output_dir=output_dir,
		report_to='tensorboard',
		logging_dir="logs/sft",
		gradient_checkpointing=False,
		eval_strategy="steps",
		eval_steps=cold_down_steps
	)

	trainer = SFTTrainer(
		model=model,
		processing_class=tokenizer,
		train_dataset=train_dataset["train"],
		eval_dataset=eval_dataset["train"],
		args=cfg,
	)
	trainer.train()

		


# -------------------------
# CLI 入口（只需在此设置 / 通过命令行传参）
# -------------------------
def parse_args():
	p = argparse.ArgumentParser()
	# 必要项：模型名、数据、输出目录
	p.add_argument("--model_name", type=str, default="pretrained_models/Qwen3-4B-Instruct-2507", help="HF 模型名或本地路径")
	p.add_argument("--data_file", type=str, default="dataset/sft", help="JSON 数据文件路径（数组格式）")
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
		dataset_path=(args.data_file + "-train.json", args.data_file + "-eval.json"),
		output_dir=args.output_dir,
		max_length=args.max_length,
		batch_size=args.batch_size,
		epochs=args.epochs,
		lr=args.lr,
		weight_decay=args.weight_decay,
		warmup_steps=args.warmup_steps,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
	)