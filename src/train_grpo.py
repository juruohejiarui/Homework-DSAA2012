import os
import models
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig
from pycantonese import characters_to_jyutping as char2jyut
from datasets import load_dataset
from tqdm import tqdm
import itertools

def parse_args() :
	parser = argparse.ArgumentParser()
	parser.add_argument('--sft_lora', type=str, default='ckpts/sft_lora')
	parser.add_argument('--base_model', type=str, default='pretrained_models/Qwen3-4B-Instruct-2507')
	parser.add_argument('--output_path', type=str, default='ckpts/grpo_lora')

	parser.add_argument('--data_path', type=str, default='dataset/grpo.json')

	parser.add_argument('--num_epochs', type=int, default=2)
	parser.add_argument('--batch_size', type=int, default=4)
	parser.add_argument('--lr', type=float, default=1e-5)
	parser.add_argument('--max_new_tokens',type=int, default=32)
	parser.add_argument('--top_p', type=float, default=0.95)
	parser.add_argument('--temperature', type=float, default=0.9)

	args = parser.parse_args()
	return args
	

def correct_reward_func(prompts, completions : list[list[dict[str, str]]], completions_ids=None, **kwargs) -> list[list[float]] :
	INVALID_REWARD = -10.0

	map0243 = {6:0, 9:0, 4:2, 3:4, 5:4, 1:3, 2:3, 7:3, 8:3}

	rw : list[list[float]] = []
	
	for i in range(len(prompts)) :
		prompt, completion = prompts[i], completions[i]
		rw_this = []

		prompt_usr = prompt[-1]['content']
		pit_str = prompt_usr.split("Pitches: ")[-1].strip()
		pit_list = pit_str[1:-1].split(',')
		pit_list = [int(pit.strip()) for pit in pit_list]

		for comp in completion :
			comp = comp['content'].strip()
			res = char2jyut(comp)
			tone2043 = []
			rw_val = 0.0
			for ch, jyut in res :
				if jyut is None :
					rw_val = INVALID_REWARD
					break
				
				# split jyut by digital
				sep_idx = []
				for i, c in enumerate(jyut) :
					if c.isdigit() :
						sep_idx.append(i)
				if len(sep_idx) == 0 :
					rw_val = INVALID_REWARD
					break
					
				tones = [map0243[int(jyut[i])] for i in sep_idx]
				
				tone2043.extend(tones)
			if rw_val != INVALID_REWARD :
				if len(tone2043) != len(pit_list) :
					rw_val = INVALID_REWARD
				else :
					match_cnt = sum([1 for t1, t2 in zip(tone2043, pit_list) if t1 == t2])
					rw_val = match_cnt / len(pit_list)

					tqdm.write(f"Comp: {comp}")
			rw.append(rw_val)
	return rw


	
				

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

	sft_tokenizer = AutoTokenizer.from_pretrained(args.sft_lora, padding_side='left')

	grpo_model = load_model(
		base_path = args.base_model,
		lora_path = args.sft_lora,
		trainable = True
	)
	grpo_tokenizer = AutoTokenizer.from_pretrained(args.sft_lora, padding_side='left')

	sft_model.to(models.device)
	grpo_model.to(models.device)

	dataset = load_dataset("json", data_files=args.data_path)

	print(f"Loaded dataset with {len(dataset['train'])} items.")

	dataset = dataset['train']
	grpo_cfg = GRPOConfig(
		bf16=True,
		per_device_train_batch_size=args.batch_size,
		gradient_accumulation_steps=8,
		num_train_epochs=args.num_epochs,

		learning_rate=args.lr,
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
		shuffle_dataset=True,

		temperature=args.temperature,
		top_p=args.top_p,

		save_strategy="steps",
		save_steps=500,

		repetition_penalty=1.0,
	)

	trainer = GRPOTrainer(
		model=grpo_model,
		processing_class=grpo_tokenizer,
		reward_funcs=[
			correct_reward_func
		],
		args=grpo_cfg,
		train_dataset=dataset,
	)

	trainer.train()
	