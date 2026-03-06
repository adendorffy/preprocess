import argparse
from pathlib import Path
from glob import glob
from tqdm import tqdm
import torchaudio
import numpy as np
import textgrids
from syllabify import syllabify

def get_frame_num(seconds, ms_per_frame=20):
    return np.floor(np.round((seconds / ms_per_frame * 1000), 1) + 0.5)

def load_features(file):
    feature = np.load(file)
    if len(feature.shape) == 1: 
        feature = feature.unsqueeze(0)
    return feature

def output_segment(extact_feat, extact_grid, extract_audio, features_dir, align_dir, align_file, words, current_grid, num_sub_utterances):

    current_grid.xmin = words[0].xmin
    current_grid.xmax = words[-1].xmax

    if extact_feat:
        feature_file = align_file.replace(align_dir, features_dir).with_suffix('.npy')
        feature_file.parent.mkdir(parents=True, exist_ok=True)
        features = load_features(feature_file)

        start_frame = int(get_frame_num(current_grid.xmin))
        end_frame = int(get_frame_num(current_grid.xmax))

        feature_out_file = features_dir.parent / features_dir.name + '_sliced' / feature_file.relative_to(features_dir).parent / feature_file.stem + f'-{num_sub_utterances:04d}' + feature_file.suffix
        feature_out_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_out_file, current_grid.features)

    if extact_grid:
        current_grid["words"] = textgrids.Tier(words)
        phones = [p for p in textgrids.TextGrid(align_file)["phones"] if p.xmin >= current_grid.xmin and p.xmax <= current_grid.xmax]
        current_grid["phones"] = textgrids.Tier(phones)

        phone_transcription = [p.text for p in phones]
        phone_xmins = [p.xmin for p in phones]
        phone_xmaxs = [p.xmax for p in phones]
        syllables = syllabify.syllabify(phone_transcription)
        syl_intervals = textgrids.Tier()
        for syl in syllables:
            syl = [item for sublist in syl for item in sublist]
            syl_len = len(syl)
            syl_intervals.append(textgrids.Interval(' '.join(syl), phone_xmins[0], phone_xmaxs[syl_len-1]))
            del phone_xmins[:syl_len]
            del phone_xmaxs[:syl_len]
        current_grid["syllables"] = syl_intervals

        if current_grid.xmin > 0.0:
            time_offset = current_grid.xmin
            for tier in current_grid.values():
                tier.xmin -= time_offset
                tier.xmax -= time_offset
                for interval in tier:
                    interval.xmin -= time_offset
                    interval.xmax -= time_offset
            current_grid.xmin = 0.0
            current_grid.xmax -= time_offset
        
        align_out_file = align_dir.parent / align_dir.name + '_sliced' / align_file.relative_to(align_dir).parent / align_file.stem + f'-{num_sub_utterances:04d}' + align_file.suffix
        align_out_file.parent.mkdir(parents=True, exist_ok=True)
        current_grid.write(str(align_out_file))
    
    if extract_audio:
        audio_file = align_file.replace("alignments", "audio").with_suffix('.flac')

        wav, sr = torchaudio.load(str(audio_file))
        start_sample = int(current_grid.xmin * sr)
        end_sample = int(current_grid.xmax * sr)

        new_folder_name = audio_file.parent.name + "_sliced"
        audio_out_dir = audio_file.parent.parent / new_folder_name
        audio_out_path = audio_out_dir / f"{Path(audio_file).stem}-{num_sub_utterances:04d}.flac"
        audio_out_path.parent.mkdir(parents=True, exist_ok=True)

        wav_sliced = wav[:, start_sample:end_sample]
        torchaudio.save(str(audio_out_path), wav_sliced, sr)

def main(args):
    extract_feat = args.features
    extract_grid = args.grids
    extract_audio = args.audio

    print(f"Extracting features: {extract_feat}, Extracting grid: {extract_grid}, Extracting audio: {extract_audio}")
    align_files = list(Path(args.align_dir).rglob('*.TextGrid'))
    feature_files = list(Path(args.features_dir).rglob('*.npy'))
    assert len(align_files) == len(feature_files), "Number of alignments and features must match"

    for align_file in tqdm(align_files, desc="Processing alignments", total=len(align_files), unit="file"):
        current_grid = textgrids.TextGrid(align_file)
        words = []
        num_sub_utterances = 0
        for word in current_grid["words"]:
            if word.text in ["<unk>", ""]: 
                if len(words) == 0: continue
                output_segment(extract_feat, extract_grid, extract_audio, features_dir, align_dir, align_file, words, current_grid, num_sub_utterances)

                words = []
                current_grid = textgrids.TextGrid(align_file)
                num_sub_utterances += 1
            else:
                words.append(word)
        
        if len(words) > 0:
            output_segment(extract_feat, extract_grid, extract_audio, features_dir, align_dir, align_file, words, current_grid, num_sub_utterances)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice features, alignments, and audio based on TextGrid word intervals.")
    parser.add_argument("features_dir", type=str, help="Directory containing extracted features (.npy files).")
    parser.add_argument("align_dir", type=str, help="Directory containing TextGrid alignments.")
    parser.add_argument("--features", action='store_true', help="Whether to extract and save sliced features.")
    parser.add_argument("--grids", action='store_true', help="Whether to extract and save sliced TextGrids.")
    parser.add_argument("--audio", action='store_true', help="Whether to extract and save sliced audio.")
    
    args = parser.parse_args()
    main(args)
        