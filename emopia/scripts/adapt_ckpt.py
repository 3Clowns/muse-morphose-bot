#!/usr/bin/env python3
"""
Shrink Pop909 MuseMorphose checkpoint → EMOPIA vocab (and wider seg emb).

Usage
-----
python adapt_ckpt.py \
       --old_ckpt pretrained_musemorphose.pt \
       --old_vocab pop909_remi_vocab.pkl \
       --new_vocab pickles/remi_vocab.pkl \
       --out_ckpt pretrained_emopia_init.pt
"""
import torch, pickle, argparse

p = argparse.ArgumentParser()
p.add_argument('--old_ckpt', required=True)
p.add_argument('--old_vocab', required=True)
p.add_argument('--new_vocab', required=True)
p.add_argument('--out_ckpt', required=True)
args = p.parse_args()

old_sd = torch.load(args.old_ckpt, map_location='cpu')
old_e2i, old_i2e = pickle.load(open(args.old_vocab,'rb'))
new_e2i, new_i2e = pickle.load(open(args.new_vocab,'rb'))

# --- 1. shrink / reorder embedding + out_proj ------------------------------
def remap_rows(mat, keep_idx):
    """Return matrix with only rows keep_idx (in that order)."""
    return mat[torch.tensor(keep_idx)]

common = [tok for tok in new_i2e if tok in old_e2i]
print(f"tokens in common: {len(common)} / {len(new_i2e)}")

keep_rows = [old_e2i[tok] for tok in common]
new_rows  = [new_e2i[tok] for tok in common]

old_emb = old_sd['token_emb.emb_lookup.weight']
old_out_w = old_sd['dec_out_proj.weight']
old_out_b = old_sd['dec_out_proj.bias']

new_emb = torch.zeros(len(new_i2e), old_emb.size(1))
new_out_w = torch.zeros_like(new_emb)
new_out_b = torch.zeros(len(new_i2e))

new_emb[new_rows]   = remap_rows(old_emb, keep_rows)
new_out_w[new_rows] = remap_rows(old_out_w, keep_rows)
new_out_b[new_rows] = old_out_b[keep_rows]

old_sd['token_emb.emb_lookup.weight'] = new_emb
old_sd['dec_out_proj.weight']         = new_out_w
old_sd['dec_out_proj.bias']           = new_out_b

# --- 2. pad seg_emb_proj for extra 64-lat (emotion 32 + latent diff if any)
proj = old_sd['decoder.seg_emb_proj.weight']          # [512,256]
pad_cols = 320 - proj.size(1)                         # want [512,320]
if pad_cols > 0:
    old_sd['decoder.seg_emb_proj.weight'] = torch.nn.functional.pad(
        proj, (0,pad_cols) )
    print(f"padded seg_emb_proj with {pad_cols} zero columns")

torch.save(old_sd, args.out_ckpt)
print(f"✓  wrote adapted checkpoint → {args.out_ckpt}")
