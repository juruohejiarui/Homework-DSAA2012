import argparse
import torch
import data
import models
import os
from functools import partial
from torch.utils.data import DataLoader
from transformers import (
	Qwen2TokenizerFast,
	Qwen2Model,
	AutoTokenizer,
	AutoModelForCausalLM,
	get_linear_schedule_with_warmup
)
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig

def setup_model_and_tokenizer(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float, use_bf16: bool = True) -> tuple[Qwen2Model, Qwen2TokenizerFast]:
	tokenizer : Qwen2TokenizerFast = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
	
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token = tokenizer.eos_token

	dtype = torch.bfloat16 if use_bf16 else None
	print(f"> Loading model {model_name} with dtype={dtype}")
	model : Qwen2Model = AutoModelForCausalLM.from_pretrained(
		model_name,
		dtype=dtype,
		device_map="auto",
		trust_remote_code=True,
		low_cpu_mem_usage=True,
	)

	# add pitch tokens
	if model_name.startswith("pretrained_models/") :
		print("Adding pitch tokens to tokenizer...")
		for keys in data.map_tokenTo4.keys() :
			tokenizer.add_tokens(keys)
		model.resize_token_embeddings(len(tokenizer))

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
		# 因为更新了embed,所以使其可以被训练
		if "embed" in name or "lm_head" in name:
			print(f"Embedding/LM_head: {name} - {num_param}")
			param.requires_grad = True
		if param.requires_grad:
			trainable += num_param
			print(f"Trainable: {name} - {num_param}")
		
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
	warmup_ratio: int,
	gradient_accumulation_steps: int,
	log_step: int = 20,
	eval_step: int = 1000,
):
	os.makedirs(output_dir, exist_ok=True)

	if isinstance(dataset_path, str) :
		train_dataset = load_dataset("json", data_files=dataset_path)['train']
		eval_dataset = None
	else :

		train_dataset = load_dataset("json", data_files=dataset_path[0])['train']
		eval_dataset = load_dataset("json", data_files=dataset_path[1])['train']

	emb_dataset = load_dataset("json", data_files="dataset/emb.json")['train']
	train_dataset = concatenate_datasets([train_dataset, emb_dataset])

	logging_dir = os.path.join("logs", os.path.basename(output_dir))

	cfg = SFTConfig(
		bf16=True,
		per_device_eval_batch_size=batch_size,
		per_device_train_batch_size=batch_size,
		gradient_accumulation_steps=gradient_accumulation_steps,
		learning_rate=lr,
		num_train_epochs=epochs,
		lr_scheduler_type='cosine',
		warmup_ratio=warmup_ratio,
		weight_decay=weight_decay,
		max_grad_norm=0.5,
		logging_steps=log_step,
		output_dir=output_dir,
		report_to='tensorboard',
		logging_dir=logging_dir,
		gradient_checkpointing=False,
		eval_strategy="steps" if eval_dataset is not None else "no",
		eval_steps=eval_step,
		save_total_limit=3,
	)

	trainer = SFTTrainer(
		model=model,
		processing_class=tokenizer,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		args=cfg,
	)
	trainer.train()

	# 合并之后保存整个模型
	print("Saving merged model...")

	trainer.save_model(args.output_dir)


# -------------------------
# CLI 入口（只需在此设置 / 通过命令行传参）
# -------------------------
def parse_args():
	p = argparse.ArgumentParser()
	# 必要项：模型名、数据、输出目录
	p.add_argument("--model_name", type=str, default="pretrained_models/Qwen2.5-7B-Instruct", help="HF 模型名或本地路径")
	p.add_argument("--data_file", type=str, default="dataset/sft", help="JSON 数据文件路径（数组格式）")
	p.add_argument("--output_dir", type=str, default="ckpts/sft_lora", help="保存目录")

	# 训练参数（常改）
	p.add_argument("--max_length", type=int, default=512)
	p.add_argument("--batch_size", type=int, default=8)
	p.add_argument("--epochs", type=int, default=3)
	p.add_argument("--lr", type=float, default=2e-5)
	p.add_argument("--weight_decay", type=float, default=0.0)
	p.add_argument("--warmup_ratio", type=float, default=0.1)
	p.add_argument("--gradient_accumulation_steps", type=int, default=1)

	# LoRA 参数（可改）
	p.add_argument("--lora_r", type=int, default=64)
	p.add_argument("--lora_alpha", type=int, default=16)
	p.add_argument("--lora_dropout", type=float, default=0.05)

	# 设备 / dtype
	p.add_argument("--device", type=str, default="cuda")
	p.add_argument("--use_bf16", default=True, action="store_true", help="尝试使用 bf16 加载模型（需硬件支持）")

	p.add_argument("--eval_step", type=int, default=1000, help="评估间隔步数")
	p.add_argument("--log_step", type=int, default=20, help="日志记录间隔步数")

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
	if args.data_file.endswith(".json"):
		dataset_path = args.data_file
	else:
		dataset_path = (args.data_file + "-train.json", args.data_file + "-eval.json")
	train(
		model=model,
		tokenizer=tokenizer,
		dataset_path=dataset_path,
		output_dir=args.output_dir,
		max_length=args.max_length,
		batch_size=args.batch_size,
		epochs=args.epochs,
		lr=args.lr,
		weight_decay=args.weight_decay,
		warmup_ratio=args.warmup_ratio,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		log_step=args.log_step,
		eval_step=args.eval_step,
	)