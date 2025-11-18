# Homework-DSAA2012

## Proposal

## Training and Evaluation

### Environment Setup

We use conda for environment management. To create the environment, run:

```bash
conda env create -f environment.yml
conda activate DSAA2012
```

### Dataset and Pretrained Models

For submodule initialization :

```bash
git submodule update --init --recursive
```

For pretrained models:

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir ./src/pretrained_models/Qwen3-4B-Instruct-2507
```

### Training

First train and store the tone model:

```bash
cd ./src
mkdir ckpts
python train_tone.py
# or you can use torchrun train_tone.py
```

Then generate dataset and train SFT lyrics model :

```bash
python generate_lyrics_sft_data.py
python train_lyrics_sft.py
```

Then generate dataset and use GRPO for further training :

```bash
python generate_lyrics_grpo_data.py
python train_lyrics_grpo.py
```

### Inference

First leverage tone model for generating proper tones and generate prompts for lyrics model:

```bash
python infer_tone.py
python generate_infer_prompts.py
```

Then use lyrics model for inference:

```bash
python infer_lyrics.py
```


