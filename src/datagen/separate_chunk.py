import os
import whisper
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm
import torch
import json
import numpy as np
import lyrics2labfile as lrc2lab

model = None

def create_chunk_dir(audio_name : str) -> str :
    chunk_dir = os.path.join("../dataset/chunks", audio_name)
    if not os.path.exists(chunk_dir) :
        os.makedirs(chunk_dir, exist_ok=True)
    else :
        return None
    return chunk_dir

def separate_auto(audio : str, min_silence_len=300, silence_thresh=-33, keep_silence=200):
    global model
    if model is None :
        model = whisper.load_model("large", device="cuda" if torch.cuda.is_available() else "cpu")
    audio_name = audio[audio.rfind("/") + 1: ]
    wav : AudioSegment = AudioSegment.from_file(os.path.join(audio, "vocals.wav"))
    
    chunks = split_on_silence(
        wav,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
   
    results = []
    
    chunk_dir = create_chunk_dir(audio_name)

    if chunk_dir is None :
        return

    for i, chunk in enumerate(chunks) :
        chunk_path = os.path.join(chunk_dir, f"{i}.wav")
        chunk.export(chunk_path, format="wav")
        text = model.transcribe(chunk_path)
        results.append({"audio" : i, "text": text['text']})
    with open(os.path.join(chunk_dir, "metadata.json"), 'w', encoding="utf-8") as f :
        json.dump(results, f, indent=4, ensure_ascii=False)
    return chunks

def separate_manual(audio : str, timestamp : str) :
    global model
    if model is None :
        model = whisper.load_model("large", device="cuda" if torch.cuda.is_available() else "cpu")
    audio_name = audio[audio.rfind("/") + 1 : ]
    wav : AudioSegment = AudioSegment.from_file(os.path.join(audio, "vocals.wav"))
    with open(timestamp, 'r') as f :
        seg = f.readlines()
    def toIdx(line : str) :
        ele = line.split(':')
        result = 0
        result += int(ele[0]) * 3600
        result += int(ele[1]) * 60
        result += int(ele[2])
        result += int(ele[3]) / 25
        return result * 1000
    results = []

    chunk_dir = create_chunk_dir(audio_name)

    if chunk_dir is None :
        return 

    for i, (st, ed) in enumerate(zip(seg[:-1], seg[1:])) :
        tmp = st.strip().split(' ')
        
        if len(tmp) > 1:
            st = tmp[0].strip()
            text = tmp[1].strip()
        else :
            text = None

        if isinstance(text, str) and text == "DELETE" :
            continue

        ed = ed.split(' ')[0]
        part = wav[toIdx(st) - 5 : toIdx(ed) + 5]
        chunk_path = os.path.join(chunk_dir, f"{i}.wav")
        part.export(chunk_path, format="wav")

        text = text if text is not None else model.transcribe(chunk_path)['text']

        results.append({"audio" : i, "text": text if isinstance(text, str) else text['text']})
    
    with open(os.path.join(chunk_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__" :
    items = os.listdir("../dataset/audios/")
    for item in tqdm(items) :
        if os.path.exists(os.path.join("../dataset/timestamp", f"{item}.txt")) :
            separate_manual(os.path.join("../dataset/audios", item), os.path.join("../dataset/timestamp", f"{item}.txt"))
        else :
            separate_auto(os.path.join("../dataset/audios/", item))

        with open(os.path.join("../dataset/chunks", item, "metadata.json"), 'r', encoding='utf-8') as f :
            metas : list[dict] = json.load(f)
            
        for meta in metas :
            lab, pho = lrc2lab.convert(meta['text'])
            meta['phoneme'] = pho
            with open(os.path.join("../dataset/chunks", item, f"{meta['audio']}.lab"), 'w', encoding='utf-8') as f :
                f.write(lab.strip() + "\n")
        
        with open(os.path.join("../dataset/chunks", item, "metadata.json"), 'w', encoding='utf-8') as f :
            json.dump(metas, f, indent=4, ensure_ascii=False)