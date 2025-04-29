#!/usr/bin/env python3
# tokenize_emopia.py
"""
Convert EMOPIA MIDI clips to MuseMorphose REMI pickles.

Output: for each clip  <id>.pkl  containing
    bar_pos : list[int]     # cumulative length after each bar, + final sentinel
    events  : list[dict]    # {'name': str, 'value': str}

A global remi_vocab.pkl (event2idx, idx2event) is created as in the
original implementation.
"""
import argparse, os, pickle, glob, re, collections
from tqdm import tqdm
import miditoolkit            # requirements.txt already lists it
import numpy as np
import pandas as pd

###############################################################################
# Constants – MUST match MuseMorphose dataloader.py                           #
###############################################################################
PPQ               = 480                    # ticks / quarter
BEAT_FRACTION     = 16                     # 16 = 16th-note grid
BAR_RESOL         = PPQ*4                 # one bar of 4/4
BEAT_RESOL        = BAR_RESOL // BEAT_FRACTION
VELOCITY_BINS     = np.linspace(1,127,32,dtype=int)   # quantise to 32 classes
DURATION_BINS     = np.array(
    [BEAT_RESOL//4, BEAT_RESOL//2, BEAT_RESOL, BEAT_RESOL*2,
     BEAT_RESOL*3, BEAT_RESOL*4, BEAT_RESOL*6, BEAT_RESOL*8])     # as REMI-pop

SPECIAL = ['Bar_None','EOS_None','PAD_None']                # always id 0..2

###############################################################################
def quantise(x, bins):                        # helper for vel & dur
    return str(int(bins[(np.abs(bins-x)).argmin()]))

def encode_clip(midi):
    """Return bar_pos[], events[] for one MIDIFile."""
    inst = midi.instruments[0]               # EMOPIA is mono-track piano
    notes = sorted(inst.notes, key=lambda n: n.start)

    # bar index for each note (integer division)
    bar_of_note = lambda n: n.start // BAR_RESOL
    n_bars = bar_of_note(notes[-1]) + 1

    # pre-allocate events
    events, bar_pos = [], []

    for bar in range(n_bars):
        events.append({'name':'Bar','value':'None'})
        beat_cursor_this_bar = 0

        # gather notes that start inside this bar
        in_bar = [n for n in notes if bar_of_note(n)==bar]
        in_bar.sort(key=lambda n:n.start)

        for note in in_bar:
            # Beat token if we have skipped grid positions
            beat_idx = (note.start - bar*BAR_RESOL) // BEAT_RESOL
            if beat_idx != beat_cursor_this_bar:
                beat_cursor_this_bar = beat_idx
                events.append({'name':'Beat','value':str(beat_idx)})

            events.extend([
                {'name':'Note_Pitch',   'value':str(note.pitch)},
                {'name':'Note_Velocity','value':quantise(note.velocity, VELOCITY_BINS)},
                {'name':'Note_Duration','value':quantise(note.end-note.start, DURATION_BINS)}
            ])

    # final sentinel bar_pos as required by dataloader.py :contentReference[oaicite:1]{index=1}
    bar_indices = [i for i,e in enumerate(events) if e['name']=='Bar']+[len(events)]
    bar_pos = bar_indices

    return bar_pos, events

###############################################################################
def build_vocab(all_events):
    """Rebuild REMI vocabulary, preserving original ordering."""
    counter = collections.Counter(all_events)
    idx2event = SPECIAL.copy()
    # stable sort by (name,value) to match original file order
    idx2event += sorted(counter.keys())
    event2idx  = {e:i for i,e in enumerate(idx2event)}
    # sanity check – must reproduce index of PAD token expected by dataloader
    assert event2idx['PAD_None'] == 2
    return event2idx, idx2event

###############################################################################
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--midi_dir',    required=True)
    p.add_argument('--split_csv',   nargs='+', required=True,
                   help='train/val/test clip csv files (header "clip")')
    p.add_argument('--out_dir',     required=True)
    p.add_argument('--quarter_resolution', type=int, default=16,
                   help='#grid divisions per bar (MuseMorphose uses 16)')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # gather clips in each split
    splits = {}
    for csv_path in args.split_csv:
        split = os.path.basename(csv_path).split('_')[0]     # train / val / test
        clips = list(pd.read_csv(csv_path)['clip'])
        splits[split]=clips

    all_event_strings = []

    for split, clip_list in splits.items():
        split_dir = os.path.join(args.out_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        for i, clip_fn in enumerate(tqdm(clip_list, desc=f'[{split}]')):
            midi_path = os.path.join(args.midi_dir, clip_fn)
            midi = miditoolkit.MidiFile(midi_path)
            bar_pos, events = encode_clip(midi)

            # accumulate for vocab
            all_event_strings += [f"{e['name']}_{e['value']}" for e in events]

            stem = os.path.splitext(clip_fn)[0]          # keeps 'Q3_song42_000'
            out_path = os.path.join(split_dir, f"{stem}.pkl")
            pickle.dump((bar_pos, events), open(out_path, 'wb'))

    # ----------------------------------------------------------------------------
    event2idx, idx2event = build_vocab(all_event_strings)
    vocab_path = os.path.join(args.out_dir,'pickles')
    os.makedirs(vocab_path, exist_ok=True)
    pickle.dump((event2idx, idx2event),
                open(os.path.join(vocab_path,'remi_vocab.pkl'),'wb'))
    print(f'Vocabulary size = {len(event2idx)} (PAD id={event2idx["PAD_None"]})')

if __name__ == '__main__':
    main()
