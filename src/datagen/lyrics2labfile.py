from pycantonese import characters_to_jyutping, characters2jyutping
import re
import json
import argparse

def convert(line : str) -> tuple[str, str]:
    # pycantonese returns a space-separated jyutping for each character
    jy_seq = characters_to_jyutping(line)  # e.g. [('我', 'ngo5'), ('係', 'hai6'), ('香港人', 'hoeng1gong2jan4')]

    # separate each second element of jy_seq by digital (treat digital as the end of each segement) and joins all elements by itertools.chain
    jy_tokens : list[str] = []
    # print(jy_seq)
    for chars, jy in jy_seq :
        if chars == '，' or chars == '。' or chars == '！' or chars == '？':
            continue
        splits = re.split(r'(\d)', jy)
        for i in range(0, len(splits)-1, 2):
            jy_tokens.append(splits[i] + splits[i+1])
    
    chars = list(line.strip())
    # lab, phoneme
    result_lab, result_pho = "", ""
    for jy in jy_tokens:
        result_lab += f" {jy[ : -1].strip()}"
    result_pho = " ".join(jy_tokens)
    return result_lab, result_pho

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Input text file with one line of Chinese characters per line.")
parser.add_argument("--output", type=str, required=True, help="Output lab file.")

if __name__ == "__main__":
    args = parser.parse_args()
    # 示例
    line : str = args.input
    out, _ = convert(line)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    # 如果你要把每个字的 phoneme 写入 lab 文件，每行一字：
    with open(args.output, "w", encoding="utf-8") as f:
       f.write(out.strip() + "\n")
    print(f"Wrote phonemes to {args.output}")