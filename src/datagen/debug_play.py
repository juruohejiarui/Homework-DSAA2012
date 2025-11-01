from textgrid import TextGrid, IntervalTier
from pydub import AudioSegment
from pydub.playback import play
import time
from argparse import ArgumentParser

GAP = 0.3

def play_chunk(audio_name : str, index : int) :
	wav_path = f"../dataset/chunks/{audio_name}/{index}.wav"
	textgrid_path = f"../dataset/chunks/{audio_name}/TextGrid/{index}.TextGrid"
	wav : AudioSegment = AudioSegment.from_file(wav_path)
	tg = TextGrid.fromFile(textgrid_path)

	intervals : IntervalTier = tg[0]

	for interval in intervals :
		st, ed = interval.minTime, interval.maxTime
		st, ed = int(round(st * 1000)), int(round(ed * 1000))

		part = wav[st : ed]

		part = part.apply_gain(-1.0 - part.max_dBFS)

		play(part)

		time.sleep(GAP)

parser = ArgumentParser()
parser.add_argument("audio_name", type=str, help="Audio chunk folder name")
parser.add_argument("index", type=int, help="Chunk index to play")
if __name__ == "__main__" :
	args = parser.parse_args()
	play_chunk(args.audio_name, args.index)