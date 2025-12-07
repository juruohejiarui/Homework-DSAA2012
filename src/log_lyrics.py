# eval_multithread.py
import backend
import json
from tqdm import tqdm
from data import map_9to6, map_6to4, map_4toToken
from pycantonese import characters_to_jyutping
import concurrent.futures
import threading
import time

# ---- 后端模型初始化（保持跟你原来一致） ----
# 如果 backend.setup_models 会将模型装载到 GPU，并且你希望每个线程共享同一个模型实例
# 就在主线程里初始化一次（如下）。
# backend.setup_models("ckpts/tone_model_best.pth", "pretrained_models/Qwen2.5-7B-Instruct")
backend.setup_models("ckpts/tone_model_best.pth", "ckpts/")

# ---- 你的 calc_tone_acc（未改动，直接复用） ----
def calc_tone_acc(model, output: str, target: str) -> tuple[int, int]:
	tot_num = len(target)
	acc_num = 0

	# remove all blank spaces
	output = output.strip().replace(" ", "")
	target = target.strip().replace(" ", "")

	pred_jyuts = characters_to_jyutping(output)

	pred_tones = []

	for _, tk in pred_jyuts:
		if tk is None:
			return 0, tot_num
		for c in tk:
			if c.isdigit():
				pred_tones.append(c)

	if len(pred_tones) != len(target):
		return 0, tot_num

	pred_tones = [map_4toToken[map_6to4[map_9to6[t]]] for t in pred_tones]

	for i in range(len(target)):
		if pred_tones[i] == target[i]:
			acc_num += 1

	return acc_num, tot_num

# ---- file paths ----
train_data_path = "dataset/sft-train.json"
val_data_path = "dataset/sft-eval.json"

with open(train_data_path, "r", encoding="utf-8") as f:
	train_data = json.load(f)

with open(val_data_path, "r", encoding="utf-8") as f:
	val_data = json.load(f)

# ---- 帮助函数：从 messages 提取 target ----
def get_target(messages: list[dict]) -> str:
	prompt = messages[-1]["content"]
	target = prompt.split("Pitches: ")[-1].strip()
	return target

# ---- worker: 处理单个数据项 ----
def evaluate_item(item):
	"""
	返回 (acc_num, tot_num)
	"""
	try:
		msgs = item["messages"]
		prompts = msgs[:-1]
		target = get_target(prompts)

		# 注意：如果 backend.gen_lyrics 线程不安全，这里会发生问题
		outputs = backend.gen_lyrics(prompts, num_generate=1, temperature=0.0)
		output = outputs[0]
		
		# print(output, target)

		acc_num, tot_num = calc_tone_acc(backend.tone_model, output, target)
		return acc_num, tot_num
	except Exception as e:
		# 出错时返回 (0, len_target) 以保证统计继续进行，并把异常信息打印
		print("Error evaluating item:", e)
		try:
			target = get_target(item["messages"])
			return 0, len(target)
		except Exception:
			return 0, 0

# ---- 通用评估函数：并行版 ----
def evaluate_dataset(dataset, desc="Evaluating", max_workers=8):
	acc_lock = threading.Lock()
	acc_total = 0
	tot_total = 0

	start_time = time.time()
	# 使用 ThreadPoolExecutor
	with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
		futures = [executor.submit(evaluate_item, item) for item in dataset]

		# as_completed + tqdm 逐个更新进度
		for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=desc):
			acc_num, tot_num = fut.result()
			# 保护共享计数器
			with acc_lock:
				acc_total += acc_num
				tot_total += tot_num
				
				tqdm.write(f"Processed item: {acc_num}/{tot_num} (Total: {acc_total}/{tot_total})")

	elapsed = time.time() - start_time
	accuracy = acc_total / tot_total if tot_total > 0 else 0.0
	return accuracy, acc_total, tot_total, elapsed

# ---- 运行评估 ----
if __name__ == "__main__":
	# 你可以根据机器实际情况调整 max_workers（例如 4 / 8 / 16）
	# 如果有 GPU 并且 backend.gen_lyrics 会触发 GPU 同步调用，过多线程可能效果不好。
	MAX_WORKERS = 4

	# train_acc, train_acc_num, train_tot, train_time = evaluate_dataset(
	#	 train_data, desc="Training set tone accuracy", max_workers=MAX_WORKERS
	# )
	# print(f"Tone Accuracy on Training Set: {train_acc * 100:.2f}%  ({train_acc_num}/{train_tot})  time: {train_time:.1f}s")

	val_acc, val_acc_num, val_tot, val_time = evaluate_dataset(
		val_data, desc="Validation set tone accuracy", max_workers=MAX_WORKERS
	)
	print(f"Tone Accuracy on Validation Set: {val_acc * 100:.2f}%  ({val_acc_num}/{val_tot})  time: {val_time:.1f}s")
