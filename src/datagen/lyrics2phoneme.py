from pycantonese import characters_to_jyutping
import re
import json

ONSETS = ["b","p","m","f","d","t","n","l","g","k","ng","h","gw","kw","w","z","c","s","j","jy","ts","dz",""]

# 为稳定匹配，我们用按最长前缀匹配的方式
def split_onset_rime(jy):
    # jy example: "ngo5", "hoeng1", "gwong2", "yu4"
    # strip tone digits at end
    m = re.match(r"^([a-z]+)(\d)$", jy)
    if not m:
        return None, None, None
    base, tone = m.group(1), m.group(2)
    # choose onset by longest match from a prepared list
    # prepare candidate onsets sorted by length desc
    candidates = sorted([o for o in ONSETS if o!=''], key=lambda x: -len(x))
    onset = ""
    for o in candidates:
        if base.startswith(o):
            onset = o
            break
    rime = base[len(onset):] if onset else base
    return onset, rime, tone

def char_line_to_phonemes(line):
    # pycantonese returns a space-separated jyutping for each character
    jy_seq = characters_to_jyutping(line)  # e.g. "ngo5 hai6 hoeng1 gong2 jan4"
    print(jy_seq)
    
    # pycantonese sometimes returns empty tokens for punctuation; split by spaces
    jy_tokens = jy_seq.strip().split()
    chars = list(line.strip())
    # align length: if mismatch, we fall back to char-by-char conversion using pycantonese.char_to_jyutping
    if len(jy_tokens) != len(chars):
        # try per-char conversion
        jy_tokens = []
        from pycantonese import char_to_jyutping
        for ch in chars:
            got = char_to_jyutping(ch)
            if got and len(got)>0:
                jy_tokens.append(got[0])  # take first possible reading
            else:
                jy_tokens.append("")  # placeholder
    results = []
    for ch, jy in zip(chars, jy_tokens):
        if jy == "" or jy is None:
            results.append({'char': ch, 'jy': '', 'onset': '', 'rime': '', 'tone': '', 'phonemes': ''})
            continue
        onset, rime, tone = split_onset_rime(jy)
        if onset is None:
            onset = ""
            rime = jy
            tone = ""
        # form phoneme sequence: we separate onset and rime into tokens; tone kept as separate token
        # e.g., "gwong2" -> "gw ong 2"  (you can map rime->phones further)
        phoneme_tokens = []
        if onset: phoneme_tokens.append(onset)
        # naive split rime into nucleus + coda: many finals are complex; here we keep rime as single token
        if rime: phoneme_tokens.append(rime)
        if tone: phoneme_tokens.append(tone)
        results.append({'char': ch, 'jy': jy, 'onset': onset, 'rime': rime, 'tone': tone, 'phonemes': ' '.join(phoneme_tokens)})
    return results

if __name__ == "__main__":
    # 示例
    line = "我係香港人"
    out = char_line_to_phonemes(line)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    # 如果你要把每个字的 phoneme 写入 lab 文件，每行一字：
    with open("line01.lab", "w", encoding="utf-8") as f:
        for item in out:
            # 输出格式示例："我	ngo5	ng o 5"
            f.write(f"{item['char']}\t{item['jy']}\t{item['phonemes']}\n")
    print("已生成 line01.lab")