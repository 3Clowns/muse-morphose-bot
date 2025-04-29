#!/usr/bin/env python3
import pandas as pd
import glob, os, re
import argparse

def create_song_splits(meta_df, per_Q=8, seed=1):
    """ Sample per-quadrant songs for test & val, remainder is train. """
    splits = {}
    # meta_df indexed by songID, has column 'DominantQ' in {1,2,3,4}
    for split in ['test','val']:
        sel = []
        for q in [1,2,3,4]:
            songs_in_q = meta_df[meta_df['DominantQ']==q].index.tolist()
            sampled = pd.Series(songs_in_q).sample(per_Q, random_state=seed).tolist()
            sel += sampled
        # record and remove from meta_df for next round
        splits[split] = set(sel)
        meta_df = meta_df.drop(index=sel)
    splits['train'] = set(meta_df.index.tolist())
    return splits

def assign_clips_to_splits(midi_dir, song_splits, pattern=r'Q[1-4]_(.*?)_\d+\.mid'):
    """
    Scan midi_dir for .mid files, extract songID via regex, 
    assign each clip to train/val/test per song_splits.
    """
    clip_splits = {'train':[], 'val':[], 'test':[]}
    regex = re.compile(pattern)
    for path in glob.glob(os.path.join(midi_dir, '*.mid')):
        fn = os.path.basename(path)
        m = regex.match(fn)
        if not m:
            print(f"Warning: filename '{fn}' didn't match pattern")
            continue
        songID = m.group(1)
        for split, songs in song_splits.items():
            if songID in songs:
                clip_splits[split].append(fn)
                break
    return clip_splits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_by_song', type=str, default='metadata_by_song.csv')
    parser.add_argument('--midi_dir',      type=str, default='midis')
    parser.add_argument('--out_dir',       type=str, default='split')
    parser.add_argument('--per_Q',         type=int, default=8,
                        help='Number of songs per quadrant for test/val')
    parser.add_argument('--seed',          type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load metadata_by_song.csv
    meta = pd.read_csv(args.meta_by_song, index_col='songID')
    print(f"Loaded {len(meta)} songs")

    # 2. Sample splits at song level
    song_splits = create_song_splits(meta.copy(), per_Q=args.per_Q, seed=args.seed)
    for split in ['train','val','test']:
        df = pd.DataFrame({'songID': sorted(song_splits[split])})
        df.to_csv(os.path.join(args.out_dir, f'{split}_songs.csv'), index=False)
        print(f"{split}: {len(df)} songs")

    # 3. Assign each MIDI clip to a split
    clip_splits = assign_clips_to_splits(args.midi_dir, song_splits)
    for split in ['train','val','test']:
        df = pd.DataFrame({'clip': clip_splits[split]})
        df.to_csv(os.path.join(args.out_dir, f'{split}_clips.csv'), index=False)
        print(f"{split}: {len(df)} clips")

if __name__ == '__main__':
    main()
