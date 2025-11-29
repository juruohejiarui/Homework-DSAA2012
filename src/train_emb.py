from transformers import (
	AutoTokenizer,
	AutoModelForCausalLM,
	Qwen2TokenizerFast,
	Qwen2Model,
)
import data
import torch
import os
import argparse
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

def parse_args() :
	parser = argparse.ArgumentParser(description="Train Embedding for Cantonese Tones")
	parser.add_argument('--model_name', type=str, default="pretrained_models/Qwen2.5-7B-Instruct", help='Pretrained model name or path')
	parser.add_argument('--use_bf16', action='store_true', default=True, help='Use bfloat16 precision')
	parser.add_argument('--data_file', type=str, default="dataset/emb.json", help='Path to training data file (JSON format)')
	parser.add_argument('--output_dir', type=str, default="ckpts/emb", help='Directory to save the trained model')

	parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
	parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
	parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
	parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
	parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
	
	return parser.parse_args()

def setup_model_and_tokenizer(model_name : str, use_bf16 : bool = True) -> tuple[Qwen2Model, Qwen2TokenizerFast] :
	tokenizer : Qwen2TokenizerFast = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
	
	if tokenizer.pad_token_id is None :
		tokenizer.pad_token = tokenizer.eos_token
	for keys in data.map_tokenTo4.keys() :
		tokenizer.add_tokens(keys)

	dtype = torch.bfloat16 if use_bf16 else None
	print(f"> Loading model {model_name} with dtype={dtype}")
	model : Qwen2Model = AutoModelForCausalLM.from_pretrained(
		model_name,
		dtype=dtype,
		device_map="auto",
		trust_remote_code=True,
		low_cpu_mem_usage=True,
	)
	model.resize_token_embeddings(len(tokenizer))
	for name, param in model.named_parameters() :
		param.requires_grad = False
		print(f"Frozen: {name} - {param.numel()}")
	for name, param in model.named_parameters() :
		if "embed" in name or "lm_head" in name :
			param.requires_grad = True
			print(f"Trainable: {name} - {param.numel()}")
	return model, tokenizer

def save_model(model, tokenizer, output_dir : str) :
	if not os.path.exists(output_dir) :
		os.makedirs(output_dir)
	print(f"> Saving model to {output_dir}")
	model.save_pretrained(output_dir)
	tokenizer.save_pretrained(output_dir)

if __name__ == "__main__" :
	args = parse_args()

	model, tokenizer = setup_model_and_tokenizer(args.model_name, args.use_bf16)

	dataset = load_dataset('json', data_files=args.data_file)['train']

	sft_cfg = SFTConfig(
		bf16=True,
		per_device_eval_batch_size=args.batch_size,
		per_device_train_batch_size=args.batch_size,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		learning_rate=args.lr,
		num_train_epochs=args.epochs,
		lr_scheduler_type="cosine",
		warmup_steps=0,
		weight_decay=0.0,
		logging_steps=10,
		output_dir=args.output_dir,
		report_to='tensorboard',
		logging_dir=os.path.join('logs', os.path.basename(args.output_dir)),
		gradient_checkpointing=False,
		save_total_limit=2,
	)

	trainer = SFTTrainer(
		model=model,
		processing_class=tokenizer,
		train_dataset=dataset,
		args=sft_cfg,
	)

	trainer.train()

	save_model(model, tokenizer, args.output_dir)

