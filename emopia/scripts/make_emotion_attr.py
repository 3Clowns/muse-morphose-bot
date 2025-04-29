#!/usr/bin/env python3
"""
make_emotion_attr.py
Create attr_cls/emotion/<piece>.pkl for every REMI file in the dataset.

Expected directory tree
    remi_dataset_emopia/
        ├── train / val / test     (REMI pickles you made earlier)
        └── pickles/remi_vocab.pkl

Output
    remi_dataset_emopia/attr_cls/emotion/<same-file-name>.pkl
        –  a list[int] of length n_bars (0‒3)  one entry per bar
"""
import os, pickle, argparse, glob, re
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True, help='root of REMI dataset')
    args = p.parse_args()

    attr_dir = os.path.join(args.data_dir, 'attr_cls', 'emotion')
    os.makedirs(attr_dir, exist_ok=True)

    # locate every REMI pickle in all splits
    remi_files = glob.glob(os.path.join(args.data_dir, '*', '*.pkl'))
    regex = re.compile(r'Q([1-4])_')        # quadrant encoded in original midi file-name

    for f in remi_files:
        fn = os.path.basename(f)            # e.g.  Q3_song42_000.pkl  → Q3
        m  = regex.match(fn)
        if m is None:
            raise ValueError(f"Cannot find quadrant in filename {fn}")

        quad = int(m.group(1)) - 1          # 0-based class: 0,1,2,3

        # number of bars = len(bar_pos) – 1
        bar_pos, _ = pickle.load(open(f, 'rb'))
        n_bars = len(bar_pos) - 1
        cls = [quad]*n_bars                 # one class per bar

        out = os.path.join(attr_dir, fn)    # same stem, .pkl extension
        pickle.dump(cls, open(out,'wb'))

    print(f'✓  wrote {len(remi_files)} emotion attribute files → {attr_dir}')

if __name__ == '__main__':
    main()
