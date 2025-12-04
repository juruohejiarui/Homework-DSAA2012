# Homework-DSAA2012

## Proposal

## Training and Evaluation

### Environment Setup

We use conda for environment management. To create the environment, run:

```bash
conda create -n DSAA2012 python=3.13
conda activate DSAA2012
cd ./src
mkdir ckpts pretrained_models
pip install -r requirements.txt
```

### Dataset and Pretrained Models

For submodule initialization :

```bash
git submodule update --init --recursive
```

For pretrained models, you can download from Hugging Face:

```bash
hf download Qwen/Qwen2.5-7B-Instruct --local-dir ./pretrained_models/Qwen2.5-7B-Instruct
```

or from modelscope :
```bash
modelscope download model Qwen/Qwen2.5-7B-Instruct --local_dir ./pretrained_models/Qwen2.5-7B-Instruct
```

### Training

First train and store the tone model:

```bash
python train_tone.py
```

Then generate dataset for embedding training and SFT training of lyrics model:

```bash
python gen_emb_data.py
python gen_sft_data.py
```

Finally train SFT lyrics model using :
1. Finetuning Embedding layer and LM head:

```bash
python train_emb.py --epochs 2
python ./train_sft.py \
	--model_name ckpts/emb \
	--lr 4e-5 \
	--gradient_accumulation_steps 2 \
	--lora_r 64 \
	--lora_alpha 128 \
	--lora_dropout 0.1 \
	--data_file dataset/sft \
	--output_dir ckpts/sft_lora \
	--eval_step 2000 \
	--log_step 10
```

Then finally merge lora and original model:

```bash
python ./merge.sft.py \
	--lora_path ckpts/sft_lora \
	 --base_model ./ckpts/emb \
	 --output_path ./ckpts/lyrics
```

### Inference

We apply `vllm` as inference engine for faster and batch processing. To run inference, use:

```bash
vllm serve ckpts/lyrics
```

to launch server.

Then you can use CLI interface to interact with the model:

```bash
python ./interface.py
```