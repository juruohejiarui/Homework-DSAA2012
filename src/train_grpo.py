import os
import models
import torch
import argparse
from data import map_tokenTo4, map_9to6, map_6to4
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from trl import GRPOTrainer, GRPOConfig
from pycantonese import characters_to_jyutping as char2jyut
from datasets import load_dataset
from tqdm import tqdm

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np

NUM_GENERATIONS = 16

def char_set(s):
	return set([c for c in s if not c.isspace()])
smooth = SmoothingFunction().method1

def parse_args() :
	parser = argparse.ArgumentParser()
	parser.add_argument('--lora_r', type=int, default=8)
	parser.add_argument('--lora_alpha', type=int, default=16)
	parser.add_argument('--lora_dropout', type=float, default=0.05)
	parser.add_argument('--base_model', type=str, default='pretrained_models/Qwen2.5-7B-Instruct')
	parser.add_argument('--output_path', type=str, default='ckpts/grpo_lora')

	parser.add_argument('--data_path', type=str, default='dataset/grpo.json')

	parser.add_argument('--num_epochs', type=int, default=2)
	parser.add_argument('--batch_size', type=int, default=8)
	parser.add_argument('--lr', type=float, default=2e-5)
	parser.add_argument('--max_new_tokens',type=int, default=32)
	parser.add_argument('--top_p', type=float, default=0.95)
	parser.add_argument('--temperature', type=float, default=0.9)

	args = parser.parse_args()
	return args

log_comp_interval = 10
log_comp_cnt = 0

def avg_pairwise_overlap(sentences):
	# sentences: list of str
	n = len(sentences)
	if n <= 1:
		return 0.0
	overlaps = []
	for i in range(n):
		si = char_set(sentences[i])
		for j in range(i+1, n):
			sj = char_set(sentences[j])
			denom = min(len(si), len(sj)) if min(len(si), len(sj))>0 else 1
			overlaps.append(len(si & sj) / denom)
	return 1 - float(np.mean(overlaps)) if overlaps else 1.0

def self_bleu_diversity(sentences):
	# higher value => more similar; we return 1 - avg_bleu as diversity
	n = len(sentences)
	if n <= 1:
		return 1.0
	scores = []
	for i in range(n):
		refs = [list(s) for j,s in enumerate(sentences) if j!=i]  # use char-level tokens
		hyp = list(sentences[i])
		# compute BLEU with smoothing
		try:
			score = sentence_bleu(refs, hyp, weights=(0.5,0.5), smoothing_function=smooth)
		except Exception:
			score = 0.0
		scores.append(score)
	avg_bleu = float(np.mean(scores))
	return 1.0 - avg_bleu

def self_bleu_diversity(sentences):
	# higher value => more similar; we return 1 - avg_bleu as diversity
	n = len(sentences)
	if n <= 1:
		return 1.0
	scores = []
	for i in range(n):
		refs = [list(s) for j,s in enumerate(sentences) if j!=i]  # use char-level tokens
		hyp = list(sentences[i])
		# compute BLEU with smoothing
		try:
			score = sentence_bleu(refs, hyp, weights=(0.5,0.5), smoothing_function=smooth)
		except Exception:
			score = 0.0
		scores.append(score)
	avg_bleu = float(np.mean(scores))
	return 1.0 - avg_bleu

def distinct_n(sentences, ngram=2):
	tokens = []
	for s in sentences:
		tokens += list(s)
	if len(tokens) == 0:
		return 0.0
	total_ngrams = max(1, len(tokens) - ngram + 1)
	ngs = set()
	for i in range(len(tokens)-ngram+1):
		ngs.add(tuple(tokens[i:i+ngram]))
	return len(ngs) / total_ngrams

def tone_match_ratio(senteces : list[str], pit_strs : list[str]) -> list[float] :
	rw = []
	INVALID_REWARD = -2.0
	for sent, pit_str in zip(senteces, pit_strs) :
		pit_list = [map_tokenTo4[pit] for pit in pit_str]
		res = char2jyut(sent)
		tone0243 = []
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
				
			tones = [map_6to4[map_9to6[jyut[i]]] for i in sep_idx]
			
			tone0243.extend(tones)
		if rw_val != INVALID_REWARD :
			if len(tone0243) != len(pit_list) :
				rw_val = INVALID_REWARD
			else :
				match_cnt = sum([1 for t1, t2 in zip(tone0243, pit_list) if t1 == t2])
				rw_val = match_cnt / len(pit_list)

				global log_comp_cnt, log_comp_interval
				log_comp_cnt += 1
				if log_comp_cnt % log_comp_interval == 0 :
					tqdm.write(f"Comp: {sent} rw: {rw_val:.4f}")
		rw.append(rw_val)
	return rw

def distinct_rw_func(prompts, completions : list[list[dict[str, str]]], completions_ids=None, **kwargs) -> list[list[float]] :
	
	rw : list[list[float]] = []

	sentences = []
	
	for i in range(len(prompts)) :
		prompt, completion = prompts[i], completions[i]
		sentence = completion[0]['content'].strip()

		sentences.append(sentence)
	
	for i in range(0, len(prompts), NUM_GENERATIONS) :
		rw_distinct = distinct_n(sentences[i : i+NUM_GENERATIONS], ngram=2)
		rw_distinct = [rw_distinct] * min(NUM_GENERATIONS, len(prompts) - i)
		rw.extend(rw_distinct)
	return rw

def overlap_rw_func(prompts, completions : list[list[dict[str, str]]], completions_ids=None, **kwargs) -> list[list[float]] :
	
	rw : list[list[float]] = []

	sentences = []
	
	for i in range(len(prompts)) :
		prompt, completion = prompts[i], completions[i]
		sentence = completion[0]['content'].strip()

		sentences.append(sentence)

	for i in range(0, len(prompts), NUM_GENERATIONS) :
		rw_overlap = avg_pairwise_overlap(sentences[i : i+NUM_GENERATIONS])
		rw_overlap = [rw_overlap] * min(NUM_GENERATIONS, len(prompts) - i)
		rw.extend(rw_overlap)
	return rw

def selfbleu_rw_func(prompts, completions : list[list[dict[str, str]]], completions_ids=None, **kwargs) -> list[list[float]] :
	
	rw : list[list[float]] = []

	sentences = []
	
	for i in range(len(prompts)) :
		prompt, completion = prompts[i], completions[i]
		sentence = completion[0]['content'].strip()

		sentences.append(sentence)
	
	for st in range(0, len(prompts), NUM_GENERATIONS) :
		rw_selfbleu = self_bleu_diversity(sentences[st : st+NUM_GENERATIONS])
		rw_selfbleu = [rw_selfbleu] * min(NUM_GENERATIONS, len(prompts) - st)
		rw.extend(rw_selfbleu)

	return rw


def tone_rw_func(prompts, completions : list[list[dict[str, str]]], completions_ids=None, **kwargs) -> list[list[float]] :

	rw : list[list[float]] = []

	sentences = []
	pit_strs = []

	
	for i in range(len(prompts)) :
		prompt, completion = prompts[i], completions[i]
		sentence = completion[0]['content'].strip()

		prompt_usr = prompt[-1]['content']
		pit_str = prompt_usr.split("Pitches: ")[-1].strip()
		pit_strs.append(pit_str)

		sentences.append(sentence)
	rw_tones = tone_match_ratio(sentences, pit_strs)
	
	return rw_tones

def load_model(base_path : str, lora_r : int, lora_alpha : int, lora_dropout : float) -> PeftModel :
	print(f'load lora model from {base_path} + lora_r={lora_r} ...')
	base_model = AutoModelForCausalLM.from_pretrained(
		base_path,
		dtype=torch.bfloat16,
		device_map='auto',
	)
	
	lora_config = LoraConfig(
		r=lora_r,
		lora_alpha=lora_alpha,
		target_modules=None,
		lora_dropout=lora_dropout,
		bias="none",
		task_type=TaskType.CAUSAL_LM,
	)
	model = get_peft_model(base_model, lora_config)
	
	return model

if __name__ == "__main__" :
	args = parse_args()

	grpo_model = load_model(
		base_path = args.base_model,
		lora_r=args.lora_r,
		lora_alpha=args.lora_alpha,
		lora_dropout=args.lora_dropout,
	)

	grpo_model.to(models.device)

	grpo_tokenizer = AutoTokenizer.from_pretrained(
		args.base_model,
		use_fast=True,
		trust_remote_code=True,
		padding_side="left"
	)

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
		num_generations=NUM_GENERATIONS,

		use_vllm=False,
		shuffle_dataset=True,

		temperature=args.temperature,
		top_p=args.top_p,

		save_strategy="steps",
		save_steps=500,

		repetition_penalty=0.9,
	)

	trainer = GRPOTrainer(
		model=grpo_model,
		processing_class=grpo_tokenizer,
		reward_funcs=[
			distinct_rw_func,
			overlap_rw_func,
			selfbleu_rw_func,
			tone_rw_func
		],
		args=grpo_cfg,
		train_dataset=dataset,
	)

	trainer.train()
	