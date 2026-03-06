from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
from textgrid import TextGrid, IntervalTier

def get_words_per_filename(alignments_dir, lang):
    wrd_path = alignments_dir / f"{lang}.wrd"

    wrd_per_filename = {}
    with open(wrd_path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Reading word alignments"):
            filename, start_time, end_time, word = line.strip().split()
            filename = str(filename)
            start_time = float(start_time)
            end_time = float(end_time)
            wrd_per_filename.setdefault(filename, []).append((start_time, end_time, word))
    wrd_per_filename = {k: sorted(v, key=lambda x: x[0]) for k, v in wrd_per_filename.items() if v}

    return wrd_per_filename

def get_phns_per_filename(alignments_dir, lang):    
    phn_path = alignments_dir / f"{lang}.phn"

    phns_per_filename = {}
    with open(phn_path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Reading phoneme alignments"):
            filename, start_time, end_time, label = line.strip().split()
            filename = str(filename)
            start_time = float(start_time)
            end_time = float(end_time)

            if label != "SIL":
                phns_per_filename.setdefault(filename, []).append((start_time, end_time, label))
    phns_per_filename = {k: sorted(v, key=lambda x: x[0]) for k, v in phns_per_filename.items() if v}
    return phns_per_filename

def get_segments_per_filename(words_per_filename, lang):

    segments_per_filename = {}
    for filename, words in tqdm(words_per_filename.items(), desc="Determining segments per filename"):
        if not words:
            continue
        words = sorted(words, key=lambda x: x[0])
        segments = []
        current_segment_start = words[0][0]
        current_segment_end = words[0][1]

        for start_time, end_time, word in words[1:]:
            if start_time - current_segment_end > 0.1:
                segments.append((current_segment_start, current_segment_end))
                current_segment_start = start_time
            current_segment_end = max(current_segment_end, end_time)

        segments.append((current_segment_start, current_segment_end))
        segments_per_filename[filename] = segments    
    
    return segments_per_filename

def process(audio_dir, alignments_dir, lang, only_alignments=False):
    audio_dir = Path(audio_dir) / lang
    audio_out_dir = audio_dir.parent / f"{lang}_feature_sliced"
    align_out_dir = alignments_dir / f"{lang}_feature_sliced"
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    align_out_dir.mkdir(parents=True, exist_ok=True)

    wrd_per_filename = get_words_per_filename(alignments_dir, lang)
    phns_per_filename = get_phns_per_filename(alignments_dir, lang)
    segments_per_filename = get_segments_per_filename(wrd_per_filename, lang)

    for filename in tqdm(segments_per_filename.keys(), desc="Exporting audio segments"):
        audio_path = audio_dir / f"{filename}.wav"
        audio = AudioSegment.from_wav(audio_path)
        for start, end in segments_per_filename[filename]:
            segment = audio[start * 1000:end * 1000]
            tg = TextGrid()
            wrd_tier = IntervalTier(name="words")
            phn_tier = IntervalTier(name="phones")

            if filename in wrd_per_filename and filename in phns_per_filename:
                valid_words = [w for w in wrd_per_filename[filename] if w[0] >= start and w[1] <= end]
                valid_phones = [p for p in phns_per_filename[filename] if p[0] >= start and p[1] <= end]

                for w_start, w_end, word in valid_words:
                    phones_in_this_word = [
                        (p_s, p_e, p) for p_s, p_e, p in valid_phones 
                        if p_s >= w_start - 1e-6 and p_e <= w_end + 1e-6
                    ]

                    if phones_in_this_word:
                        w_start = round(w_start - start, 2)
                        w_end = round(w_end - start, 2)
                        wrd_tier.add(w_start, w_end, word)
                        for p_s, p_e, p in phones_in_this_word:
                            p_s = round(p_s - start, 2)
                            p_e = round(p_e - start, 2)
                            phn_tier.add(p_s, p_e, p)
                    else:
                        print(f"Warning: No phonemes found for word '{word}' [{w_start}, {w_end}] in segment {filename} from {start} to {end} seconds.")
                        
            if wrd_tier.intervals and phn_tier.intervals:   
                tg.append(wrd_tier)
                tg.append(phn_tier)
                segment_filename = f"{filename}-{float(start)}.wav"
                segment_path = audio_out_dir / segment_filename
                tg_path = align_out_dir / f"{segment_filename.replace('.wav', '.TextGrid')}"
                segment.export(segment_path, format="wav")
                tg.write(tg_path)
            else:
                print(f"Warning: No valid words or phonemes found for segment {filename} from {start} to {end} seconds. Skipping export.")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Pre-process ZRC data for feature slicing.")
    parser.add_argument("audio_dir", type=Path, help="Directory containing original audio files.")
    parser.add_argument("align_dir", type=Path, help="Directory containing .phn and .wrd files.")
    parser.add_argument("language", type=str, help="Language code for processing (e.g., english, french, german)") 
    parser.add_argument("--only_alignments", action="store_true", help="If set, only process alignments without slicing audio.")
    args = parser.parse_args()

    process(args.audio_dir, args.align_dir, args.language, args.only_alignments)