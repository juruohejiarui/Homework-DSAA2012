import os
import models
import data
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig
from pycantonese import characters_to_jyutping as char2jyut
from gen_sft_data import make_input, make_inst, template_system

def parse_args() :
	parser = argparse.ArgumentParser()
	parser.add_argument('--sft_lora', type=str, default='ckpts/sft_lora')
	parser.add_argument('--base_model', type=str, default='pretrained_models/Qwen3-4B-Instruct-2507')
	parser.add_argument('--output_path', type=str, default='ckpts/grpo_lora')

	parser.add_argument('--data_path', type=str, default='dataset/grpo.json')

	parser.add_argument('--max_steps', type=int, default=1000)
	parser.add_argument('--batch_size', type=int, default=4)
	parser.add_argument('--lr', type=float, default=1e-5)
	parser.add_argument('--max_new_tokens',type=int, default=32)
	parser.add_argument('--top_p', type=float, default=0.95)
	parser.add_argument('--temperature', type=float, default=0.9)

	args = parser.parse_args()
	return args

def format_reward_func(prompts, completions, completions_ids=None, **kwargs) -> list[float] : 
	print(*completions, sep='\n---\n')
	rewards = []
	for prompt, completion in zip(prompts, completions) :
		
	while True : pass
	

def correct_reward_func(prompts, completions, completions_ids=None, **kwargs) -> list[float] :
	pass

def load_model(base_path : str, lora_path : str, trainable : bool = True) :
	print(f'load lora model from {base_path} + {lora_path} ...')
	base_model = AutoModelForCausalLM.from_pretrained(
		base_path,
		dtype=torch.bfloat16,
		device_map='auto',
	)
	
	model = PeftModel.from_pretrained(
		base_model,
		lora_path,
		dtype=torch.bfloat16,
		device_map='auto',
		is_trainable=trainable,
	)

	for name, param in model.named_parameters() :
		param.requires_grad = False
	
	if trainable :
		for name, param in model.named_parameters() :
			if "lora" in name.lower() :
				param.requires_grad = True
		
		tot = 0
		trainable = 0
		for name, param in model.named_parameters() :
			num_param = param.numel()
			tot += num_param
			if param.requires_grad :
				trainable += num_param
		print(f"Total parameters: {tot}, Trainable parameters: {trainable}, Ratio: {trainable / tot:.6f}")		
	
	return model

if __name__ == "__main__" :
	args = parse_args()

	sft_model = load_model(
		base_path = args.base_model,
		lora_path = args.sft_lora,
		trainable = False
	)

	sft_tokenizer = AutoTokenizer.from_pretrained(args.sft_lora)

	grpo_model = load_model(
		base_path = args.base_model,
		lora_path = args.sft_lora,
		trainable = True
	)
	grpo_tokenizer = AutoTokenizer.from_pretrained(args.sft_lora)

	sft_model.to(models.device)
	grpo_model.to(models.device)

	dataset = data.GRPODataset(args.data_path, shuffle=True)
	dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False, collate_fn=data.grpo_collate_fn)

	grpo_cfg = GRPOConfig(
		bf16=True,
		per_device_train_batch_size=args.batch_size,
		gradient_accumulation_steps=8,
		learning_rate=args.lr,
		num_train_epochs=2,
		lr_scheduler_type='cosine',
		warmup_ratio=0.05,
		max_grad_norm=0.3,
		logging_steps=20,
		output_dir=args.output_path,
		report_to='tensorboard',
		logging_dir="logs/grpo",
		max_prompt_length=512,
		max_completion_length=args.max_new_tokens,
		num_generations=8,
		use_vllm=False,
		shuffle_dataset=False,

		temperature=args.temperature,
		top_p=args.top_p,
		repetition_penalty=1.0,
	)

	trainer = GRPOTrainer(
		model=grpo_model,
		processing_class=grpo_tokenizer,
		reward_funcs=[
			format_reward_func,
			correct_reward_func
		],
		args=grpo_cfg,
		train_dataset=dataset
	)

	trainer.train()
	