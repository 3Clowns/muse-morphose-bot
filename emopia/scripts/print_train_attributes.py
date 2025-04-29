"""
Add missing Note_Pitch tokens (0-127) to remi_vocab.pkl
so data augmentation can't step outside the vocabulary.
"""
import pickle, os

VOCAB_PATH = "pickles/remi_vocab.pkl"   # adjust if needed

event2idx, idx2event = pickle.load(open(VOCAB_PATH, "rb"))

added = 0
for p in range(128):
    token = f"Note_Pitch_{p}"
    if token not in event2idx:
        idx = len(idx2event)
        event2idx[token] = idx
        idx2event.append(token)
        added += 1

if added:
    pickle.dump((event2idx, idx2event), open(VOCAB_PATH, "wb"))
    print(f"✓  Added {added} new pitch tokens, vocab size → {len(idx2event)}")
else:
    print("Vocabulary already contains Note_Pitch_0–127 — nothing to do.")