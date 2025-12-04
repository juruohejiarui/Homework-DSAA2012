import curses
import backend
import copy
import math
import traceback
from data import map_tokenTo4, map_4toToken

map_4to0243 = {
	0: 0,
	1: 2,
	2: 4,
	3: 3,
}
map_0243to4 = {
	0: 0,
	2: 1,
	4: 2,
	3: 3,
}

prev_lyrics : list[str] = []
prev_notes : list[list[str]] = []
prev_tones : list[list[int]] = []

align_prev_content : list[str] = []

curr_notes : str = ""
cands_0243 : list[list[int]] = []
cands_lyrics : list[list[str]] = []

prev_scroll_h = 0
prev_scroll_c = 0
cands_0243_scroll_h = 0
cands_0243_scroll_c = 0
cands_0243_sel = None
cands_lyrics_scroll_h = 0
cands_lyrics_scroll_c = 0
cands_lyrics_sel = None
cands_0243_used : str = ""
notes_used : str = ""

stages = [
	'notes', 'tones', 'lyrics'
]
views = [
	'prev',
	'input',
	'cands_0243',
	'cands_lyrics',
]
cur_stage = stages[0]
focus_view = views[0]

KEY_UP = curses.KEY_UP
KEY_DOWN = curses.KEY_DOWN

def align(lyrics : str, notes : list[str], tones : list[int]) -> tuple[list[str], list[str], list[int]] :
	lyric_chars = list(lyrics)
	aligned_lyrics = []
	aligned_notes = []
	aligned_tones = []
	for ch, note, tone in zip(lyric_chars, notes, tones) :
		if ch == ' ' or ch == '\n' :
			continue
		max_len = int(math.ceil(max(4, len(note), len(str(tone))) / 4) * 4)
		aligned_lyrics.append(ch.ljust(max_len - 1))
		aligned_notes.append(note.ljust(max_len))
		aligned_tones.append(str(tone).ljust(max_len))
	aligned_lyrics = "".join(aligned_lyrics)
	aligned_notes = "".join(aligned_notes)
	aligned_tones = "".join(aligned_tones)
	return aligned_lyrics, aligned_notes, aligned_tones

def render_scroll_view(scr : curses.window, title : str, content_lines : list[str], view_x : int, view_y : int, view_h : int, view_w : int, cur_scroll_h : int, cur_scroll_c : int, line_stride : int = 1, col : int = -1, selected_line : int | None = None) -> tuple[int, int] :
	cnt_items = len(content_lines)
	max_align_len = max(len(line) for line in content_lines) if cnt_items > 0 else 0
	view_number_w = len(str(len(content_lines))) + 1  # for ' '
	view_content_w = view_w - view_number_w - 3
	max_align_len = max(max_align_len, view_content_w)
	view_content_h = view_h - 3
	
	cur_scroll_c = min(cur_scroll_c, max(0, max_align_len - view_content_w))
	cur_scroll_h = min(cur_scroll_h, max(0, cnt_items - view_content_h))

	scroll_bar_h = ['#' if cur_scroll_h <= idx / view_content_h * cnt_items <= cur_scroll_h + view_content_h else ' ' for idx in range(view_content_h)]
	scroll_bar_c = ['#' if cur_scroll_c <= idx / view_content_w * max_align_len <= cur_scroll_c + view_content_w else ' ' for idx in range(view_content_w + view_number_w + 1)]

	scroll_bar_c = ''.join(scroll_bar_c)

	if col != -1 :
		scr.attron(curses.color_pair(col))

	scr.addstr(view_y, view_x, '+' + title + '-' * (view_w - len(title) - 2) + '+')

	for i in range(view_content_h) :
		idx = i + cur_scroll_h
		if idx == selected_line :
			scr.attron(curses.A_REVERSE)
		else :
			scr.attroff(curses.A_REVERSE)
		if idx < cnt_items :
			if idx % line_stride == 0 :
				lineId = str(idx // line_stride + 1).ljust(view_number_w)
			else :
				lineId = ' ' * view_number_w

			content = content_lines[idx].ljust(max_align_len)
			st_pos, ed_pos, cnt = 0, 0, 0
			while cnt < cur_scroll_c :
				if content[st_pos] == ' ' : cnt += 1
				else : cnt += 2
				st_pos += 1
			ed_pos = st_pos
			cnt_zh = 0
			while cnt < cur_scroll_c + view_content_w :
				if content[ed_pos].isascii() : 
					cnt += 1
				else : 
					cnt += 2
					cnt_zh += 1
				ed_pos += 1
			clip_content = content[st_pos : ed_pos]
			clip_content = clip_content.ljust(view_content_w - cnt_zh)
			scr.addstr(view_y + 1 + i, view_x, '|' + lineId + clip_content + scroll_bar_h[i] + '|')
		else :
			scr.addstr(view_y + 1 + i, view_x, '|' + ' ' * (view_w - 3) + scroll_bar_h[i] + '|')
	scr.attroff(curses.A_REVERSE)
	scr.addstr(view_y + view_h - 2, view_x, '|' + scroll_bar_c + '|')
	scr.addstr(view_y + view_h - 1, view_x, '+' + '-' * (view_w - 2) + '+')

	if col != -1 :
		scr.attroff(curses.color_pair(col))

	return cur_scroll_h, cur_scroll_c

def update(scr : curses.window) -> bool :
	h, w = scr.getmaxyx()
	global cur_stage
	global prev_scroll_h, prev_scroll_c
	global cands_0243_scroll_h, cands_0243_scroll_c
	global cands_lyrics_scroll_h, cands_lyrics_scroll_c
	global prev_lyrics, prev_notes, prev_tones, curr_notes
	global cands_0243, cands_lyrics
	global cands_0243_sel, cands_lyrics_sel
	global focus_view
	global notes_used, cands_0243_used
	
	prev_view_h = (h // 2 - 3) // 3 * 3 + 3
	prev_view_w = ((w - 3) // 4) * 4 + 3
	title_prev = " Previous Input "
	prev_scroll_h, prev_scroll_c = render_scroll_view(
		scr,
		title_prev,
		align_prev_content,
		0, 0,
		prev_view_h,
		prev_view_w,
		prev_scroll_h,
		prev_scroll_c,
		col=1 if focus_view == 'prev' else -1,
		line_stride = 3
	)

	if focus_view == 'input' :
		scr.attron(curses.color_pair(1))
	hint_input_y = prev_view_h + 1
	hint_input_x = 0
	hint_input = "Input your notes (separated by SPACE, get 0243 by ENTER):"
	scr.addstr(hint_input_y, hint_input_x, hint_input.ljust(w))

	note_input_y = hint_input_y + 1
	note_input_x = 0
	scr.addstr(note_input_y, note_input_x, curr_notes.ljust(w))
	if focus_view == 'input' :
		scr.attroff(curses.color_pair(1))
	
	cands_0243_view_h = h - note_input_y - 3
	cands_0243_view_w = (w - 3) // 2
	title_cands_0243 = " 0243 Candidates "
	cands_0243_scroll_h, cands_0243_scroll_c = render_scroll_view(
		scr,
		title_cands_0243,
		[''.join([str(tone) for tone in cand]) for cand in cands_0243],
		0, note_input_y + 2,
		cands_0243_view_h,
		cands_0243_view_w,
		cands_0243_scroll_h, cands_0243_scroll_c,
		col=1 if focus_view == 'cands_0243' else -1,
		selected_line=cands_0243_sel if focus_view == 'cands_0243' else None
	)

	cands_lyrics_view_h = h - note_input_y - 3
	cands_lyrics_view_w = w - cands_0243_view_w - 3
	title_cands_lyrics = " Lyrics Candidates "
	cands_lyrics_scroll_h, cands_lyrics_scroll_c = render_scroll_view(
		scr,
		title_cands_lyrics,
		cands_lyrics,
		cands_0243_view_w + 3, note_input_y + 2,
		cands_lyrics_view_h,
		cands_lyrics_view_w,
		cands_lyrics_scroll_h, cands_lyrics_scroll_c,
		col=1 if focus_view == 'cands_lyrics' else -1,
		selected_line=cands_lyrics_sel if focus_view == 'cands_lyrics' else None
	)

	while True :
		c = scr.getch()
		if c == curses.KEY_UP :
			if focus_view == 'prev' :
				prev_scroll_h = max(0, prev_scroll_h - 3)
			elif focus_view == 'cands_0243' :
				cands_0243_sel = max(0, (cands_0243_sel or 0) - 1) if cands_0243_sel is not None else None
				if cands_0243_sel is not None and cands_0243_sel < cands_0243_scroll_h :
					cands_0243_scroll_h = cands_0243_sel
			elif focus_view == 'cands_lyrics' :
				cands_lyrics_sel = max(0, (cands_lyrics_sel or 0) - 1) if cands_lyrics_sel is not None else None
				if cands_lyrics_sel is not None and cands_lyrics_sel < cands_lyrics_scroll_h :
					cands_lyrics_scroll_h = cands_lyrics_sel
			else :
				pass
			break
		elif c == curses.KEY_DOWN :
			if focus_view == 'prev' :
				prev_scroll_h += 3
			elif focus_view == 'cands_0243' :
				cands_0243_sel = min(len(cands_0243) - 1, cands_0243_sel + 1) if cands_0243_sel is not None else None
				if cands_0243_sel is not None and cands_0243_sel >= cands_0243_scroll_h + (cands_0243_view_h - 3) :
					cands_0243_scroll_h = cands_0243_sel - (cands_0243_view_h - 3) + 1
			elif focus_view == 'cands_lyrics' :
				cands_lyrics_sel = min(len(cands_lyrics) - 1, cands_lyrics_sel + 1) if cands_lyrics_sel is not None else None
				if cands_lyrics_sel is not None and cands_lyrics_sel >= cands_lyrics_scroll_h + (cands_lyrics_view_h - 3) :
					cands_lyrics_scroll_h = cands_lyrics_sel - (cands_lyrics_view_h - 3) + 1
			else :
				pass
			break
		elif c == curses.KEY_LEFT :
			if focus_view == 'prev' :
				prev_scroll_c = max(0, prev_scroll_c - 4)
			elif focus_view == 'cands_0243' :
				cands_0243_scroll_c = max(0, cands_0243_scroll_c - 2)
			elif focus_view == 'cands_lyrics' :
				cands_lyrics_scroll_c = max(0, cands_lyrics_scroll_c - 2)
			else :
				pass
			break
		elif c == curses.KEY_RIGHT :
			if focus_view == 'prev' :
				prev_scroll_c += 4
			elif focus_view == 'cands_0243' :
				cands_0243_scroll_c += 2
			elif focus_view == 'cands_lyrics' :
				cands_lyrics_scroll_c += 2
			else :
				pass
			break
		elif c == 9 :
			focus_view_idx = views.index(focus_view)
			focus_view = views[(focus_view_idx + 1) % len(views)]
			break
		
		elif c == ord('q') :
			if focus_view != 'input' :
				return False
			else :
				curr_notes += 'q'
		elif c == curses.KEY_ENTER or c == ord('\n') :
			if focus_view == 'input' :
				try :
					cands_0243_raw = backend.generate_0243(
						[" ".join(prev_note) for prev_note in prev_notes],
						curr_notes.strip())
					cands_lyrics = []
					cands_0243 = []
					for cand_0243 in cands_0243_raw :
						tones = cand_0243.strip()
						cands_0243.append([map_4to0243[map_tokenTo4[tone]] for tone in tones])
					cands_0243_sel = 0 if len(cands_0243) > 0 else None
					notes_used = curr_notes.strip()
				except Exception as e :
					cands_lyrics = []
					cands_0243 = [str(e)]
					cands_0243.extend(traceback.format_exc().replace('\t', '  ').split('\n'))
					cands_0243_sel = None
			elif focus_view == 'cands_0243' :
				if cands_0243_sel is not None :
					selected_tones = cands_0243[cands_0243_sel]
					try :
						cands_lyrics = backend.search_valid_lyrics(
							prev_lyrics,
							"".join([map_4toToken[map_0243to4[tone]] for tone in selected_tones]),
							max_candidate=20,
							max_iter=1000,
						)
						cands_lyrics_sel = 0 if len(cands_lyrics) > 0 else None
						cands_0243_used = selected_tones
					except Exception as e :
						cands_lyrics = [str(e)]
						cands_lyrics.extend(traceback.format_exc().replace('\t', '  ').split('\n'))
						cands_lyrics_sel = None
			elif focus_view == 'cands_lyrics' :
				if cands_lyrics_sel is not None :
					lyric = cands_lyrics[cands_lyrics_sel]
					notes = notes_used.split(' ')
					tones = [int(tone) for tone in cands_0243_used]
					prev_lyrics.append(lyric)
					prev_notes.append(notes)
					prev_tones.append(tones)
					aligned_lyr, aligned_not, aligned_ton = align(lyric, notes, tones)
					align_prev_content.extend([aligned_not, aligned_ton, aligned_lyr])
					
					# reset input
					curr_notes = ""
					notes_used = ""
					cands_0243 = []
					cands_0243_sel = None
					cands_0243_used = ""
					cands_lyrics = []
					cands_lyrics_sel = None
			break
		elif c == curses.KEY_BACKSPACE :
			if len(curr_notes) > 0 :
				curr_notes = curr_notes[:-1]
			break
		else :
			if focus_view == 'input' :
				curr_notes += chr(c)
			break
	return True

if __name__ == "__main__" :
	print("Initializing Models...")
	backend.setup_models("ckpts/tone_model_best.pth", "ckpts/lyrics")
	 
	scr = curses.initscr()
	curses.noecho()
	curses.cbreak()
	scr.keypad(True)
	curses.start_color()
	curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
	curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)

	while True :
		scr.clear()
		if not update(scr) :
			break
		else :
			scr.refresh()

	curses.nocbreak()
	scr.keypad(False)
	curses.echo()
	curses.endwin()