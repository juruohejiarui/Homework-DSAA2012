import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def parse_args() :
	parser = argparse.ArgumentParser()
	parser.add_argument("--base_model", type=str, default="pretrained_models/Qwen3-4B-Instruct-2507",
		help="Path to the base model or model identifier from huggingface.co/models")
	parser.add_argument("--lora_path", type=str, default="ckpts/sft_lora",
		help="Path to the lora directory")
	parser.add_argument("--output_path", type=str, default="ckpts/sft",
		help="Path to save the merged model")
	return parser.parse_args()

if __name__ == "__main__" :
	args = parse_args()

	tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
	if tokenizer.pad_token_id is None :
		tokenizer.pad_token = tokenizer.eos_token
	
	model = AutoModelForCausalLM.from_pretrained(
		args.base_model,
		trust_remote_code=True,
		dtype=torch.bfloat16,
		device_map="auto",
		low_cpu_mem_usage=True,
	)

	peft_model = PeftModel.from_pretrained(
		model,
		args.lora_path,
		is_trainable=False
	)

	merge_model = peft_model.merge_and_unload()

	merge_model.save_pretrained(args.output_path)
	tokenizer.save_pretrained(args.output_path)

	print(f"Merged model saved to {args.output_path}")