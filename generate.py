import sys
sys.path.append('./model')
import os
import torch
import yaml
import miditoolkit
import numpy as np
from model.musemorphose import MuseMorphose
from dataloader import REMIFullSongTransformerDataset
from utils import numpy_to_tensor, tensor_to_numpy
from remi2midi import remi2midi
from collections import defaultdict

class PositionTracker:
    def __init__(self):
        self.current_bar = 0
        self.current_beat = 0
        self.position = 0
        self.last_beat = -1

    def update(self, event: str):
        if 'Bar_' in event:
            self.current_bar += 1
            self.current_beat = 0
            self.position = 0
        elif 'Beat_' in event:
            self.current_beat = int(event.split('_')[1])
            self.position = 0
        elif 'Position_' in event:
            self.position = int(event.split('_')[1])

def main():
    config_path = 'config/default.yaml'
    ckpt_path = 'musemorphose_pretrained_weights.pt'
    out_dir = 'generations/'
    input_midi = 'input.mid'

    config = yaml.safe_load(open(config_path))
    device = torch.device(config['training']['device'])

    dset = REMIFullSongTransformerDataset(
        config['data']['data_dir'],
        config['data']['vocab_path'],
        do_augment=False,
        model_enc_seqlen=config['data']['enc_seqlen'],
        model_dec_seqlen=config['generate']['dec_seqlen'],
        model_max_bars=config['generate']['max_bars'],
        pieces=[],
        pad_to_same=False
    )
    
    mconf = config['model']
    model = MuseMorphose(
        mconf['enc_n_layer'], 
        mconf['enc_n_head'], 
        mconf['enc_d_model'], 
        mconf['enc_d_ff'],
        mconf['dec_n_layer'], 
        mconf['dec_n_head'], 
        mconf['dec_d_model'],
        mconf['dec_d_ff'],
        mconf['d_latent'],
        mconf['d_embed'], 
        dset.vocab_size,
        d_polyph_emb=mconf['d_polyph_emb'],
        d_rfreq_emb=mconf['d_rfreq_emb'],
        cond_mode=mconf['cond_mode']
    )
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval() 

    midi = miditoolkit.MidiFile(input_midi)
    remi_events = midi_to_remi(midi, dset.event2idx)
    
    enc_data = prepare_encoder_input(remi_events, dset, config)
    
    generated_midi = generate_music(
        model=model,
        midi=midi,
        remi_events=remi_events,
        enc_data=enc_data,
        dset=dset,
        config=config,
        device=device
    )
    
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, os.path.basename(input_midi))
    remi2midi(generated_midi, f"{output_path}_generated.mid")
    
def random_shift_attr_cls(n_bars: int, max_shift: int = 2, min_shift: int = -2) -> np.ndarray:
    return np.random.randint(min_shift, max_shift + 1, size=n_bars)

def process_attributes(orig_values: np.ndarray, shifts: np.ndarray, clamp_range: tuple = (0, 7)) -> np.ndarray:
    shifted = orig_values + shifts
    return np.clip(shifted, *clamp_range)

def midi_to_remi(midi, event2idx):
    events = []
    ticks_per_beat = midi.ticks_per_beat
    ticks_per_bar = 4 * ticks_per_beat
    
    beat_resolution = 8
    quantization_unit = ticks_per_beat // beat_resolution
    
    prev_bar = -1
    prev_beat = -1
    
    for note in midi.instruments[0].notes:
        start = note.start
        quantized_start = round(start / quantization_unit) * quantization_unit
        
        bar = quantized_start // ticks_per_bar
        beat_pos = (quantized_start % ticks_per_bar) / ticks_per_beat
        beat = int(beat_pos)
        position = int(round((beat_pos - beat) * beat_resolution))
        
        if bar != prev_bar:
            events.append(f"Bar_{bar}")
            prev_bar = bar
            prev_beat = -1
        
        if beat != prev_beat:
            events.append(f"Beat_{beat}")
            prev_beat = beat
        
        events.append(f"Position_{position}")
        
        duration = note.end - note.start
        quantized_duration = round(duration / quantization_unit) * quantization_unit
        duration_unit = max(quantized_duration // (ticks_per_beat // 4), 1)
        
        velocity = note.velocity // 8 * 8
        
        events.extend([
            f"NoteOn_{note.pitch}",
            f"Duration_{duration_unit}",
            f"Velocity_{velocity}"
        ])
    
    return [e for e in events if e in event2idx]

def prepare_encoder_input(events, dset, config):
    pad_candidates = ['PAD_None', '<PAD>', 'Pad', '[PAD]']
    pad_token = next((c for c in pad_candidates if c in dset.event2idx), None)
    pad_idx = dset.event2idx[pad_token] if pad_token else 0

    valid_events = [e for e in events if e in dset.event2idx]
    bars = [i for i, e in enumerate(valid_events) if 'Bar_' in e]
    
    max_bars = config['generate']['max_bars']
    enc_seqlen = config['data']['enc_seqlen']
    enc_input_np = np.full((max_bars, enc_seqlen), pad_idx, dtype=np.int64)
    
    for i in range(min(len(bars), max_bars)):
        start = bars[i]
        end = bars[i+1] if i+1 < len(bars) else len(valid_events)
        enc_input_np[i, :end-start] = [dset.event2idx[e] for e in valid_events[start:end]]
    
    device = config['training']['device']
    enc_input = torch.from_numpy(enc_input_np).to(device)
    enc_padding_mask = torch.zeros_like(enc_input, dtype=torch.bool)
    
    return {
        'enc_input': enc_input,
        'enc_padding_mask': enc_padding_mask
    }

def calculate_polyphony(events: list, ticks_per_bar: int) -> np.ndarray:
    bar_stats = {}
    current_bar = 0
    active_notes = set()
    
    for event in events:
        if 'Bar_' in event:
            current_bar = int(event.split('_')[1])
        elif 'NoteOn_' in event:
            pitch = int(event.split('_')[1])
            active_notes.add(pitch)
            if current_bar not in bar_stats:
                bar_stats[current_bar] = []
            bar_stats[current_bar].append(len(active_notes))
        elif 'NoteOff_' in event:
            pitch = int(event.split('_')[1])
            active_notes.discard(pitch)
    
    polyph_levels = []
    for bar in sorted(bar_stats.keys()):
        max_notes = max(bar_stats[bar])
        polyph_levels.append(min(max_notes, 7))
    
    return np.array(polyph_levels)

def calculate_rhythm_density(events: list, ticks_per_bar: int, ticks_per_beat: int) -> np.ndarray:
    bar_counts = defaultdict(int)
    current_bar = 0
    
    for event in events:
        if 'Bar_' in event:
            current_bar = int(event.split('_')[1])
        elif 'Position_' in event:
            bar_counts[current_bar] += 1
    
    max_positions = ticks_per_bar // (ticks_per_beat // 4)
    return np.array([min(count//4, 7) for bar, count in sorted(bar_counts.items())])

def postprocess_events(events: list, vocab: dict) -> list:
    processed = []
    last_note = None
    
    for event in events:
        if 'NoteOn_' in event:
            if event == last_note:
                processed.append('Rest_1')
            else:
                processed.append(event)
                last_note = event
        else:
            processed.append(event)
    
    return [e for e in processed if e in vocab]

def generate_music(model, midi, remi_events, enc_data, dset, config, device):
    with torch.no_grad():
        latents = model.get_sampled_latent(
            enc_data['enc_input'].permute(1, 0),
            padding_mask=enc_data['enc_padding_mask']
        ).to(device)

    num_bars = latents.size(0)
    
    tracker = PositionTracker()
    
    orig_polyph = np.zeros(num_bars)
    orig_rhythm = np.zeros(num_bars)
    
    polyph_shifts = random_shift_attr_cls(
        num_bars,
        config['attributes'].get('max_polyph_shift', 2),
        config['attributes'].get('min_polyph_shift', -2)
    )
    rhythm_shifts = random_shift_attr_cls(
        num_bars,
        config['attributes'].get('max_rhythm_shift', 4),
        config['attributes'].get('min_rhythm_shift', -3)
    )
    
    polyph_cls = process_attributes(
        orig_polyph, polyph_shifts,
        config['attributes'].get('attr_clamp_range', (0, 7))
    )
    rfreq_cls = process_attributes(
        orig_rhythm, rhythm_shifts,
        config['attributes'].get('attr_clamp_range', (0, 7))
    )
    
    rfreq_cls = torch.from_numpy(rfreq_cls).long().to(device)
    polyph_cls = torch.from_numpy(polyph_cls).long().to(device)
    

    inp = torch.tensor([[dset.event2idx['Bar_None']]], device=device)
    generated = []
    max_seq_len = config['generate'].get('max_input_dec_seqlen', 512)
    
    for step in range(max_seq_len):
        current_bar = min(
            step // config['data']['dec_seqlen'],
            num_bars - 1
        )
      
        current_latent = latents[current_bar].unsqueeze(0)
        
        current_rfreq = rfreq_cls[current_bar].unsqueeze(0)
        current_polyph = polyph_cls[current_bar].unsqueeze(0)
        
        rfreq_emb = model.rfreq_attr_emb(current_rfreq)
        polyph_emb = model.polyph_attr_emb(current_polyph)
        
        dec_seg_emb = torch.cat([
            current_latent,
            rfreq_emb,
            polyph_emb
        ], dim=-1)
        
        dec_seg_emb = dec_seg_emb.unsqueeze(0)

        with torch.no_grad():
            logits = model.generate(inp, dec_seg_emb, keep_last_only=False)

        try:
            logits = logits[-1] / max(config['generate'].get('temperature', 1.0), 1e-8)
            
            logits = logits - logits.max()
            probs = torch.softmax(logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cum_probs = sorted_probs.cumsum(dim=-1)
            mask = (cum_probs <= config['generate'].get('nucleus_p', 0.9)) | (cum_probs == 0)
            
            if torch.all(~mask):
                mask[..., 0] = True
                
            filtered_probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
            sum_probs = filtered_probs.sum(dim=-1, keepdim=True)
            
            if torch.any(sum_probs == 0):
                filtered_probs = torch.ones_like(filtered_probs) / filtered_probs.size(-1)
            else:
                filtered_probs /= sum_probs

            next_token = sorted_indices.gather(1, torch.multinomial(filtered_probs, 1))

        except Exception as e:
            print(f"Sampling fallback: {str(e)}")
            next_token = torch.randint(0, dset.vocab_size, (1, 1), device=device)
        
        inp = torch.cat([inp, next_token.T], dim=0) 
        generated.append(next_token.item())

        if dset.idx2event[next_token.item()] == 'EOS_None':
            break
          
        tracker.update(dset.idx2event[next_token.item()])
        
        if 'Beat' in dset.idx2event[next_token.item()] and tracker.current_beat < tracker.last_beat:
            continue

    generated_events = [dset.idx2event[idx] for idx in generated if idx in dset.idx2event]
    
    processed_events = postprocess_events(generated_events, dset.event2idx)
    if not processed_events[0].startswith('Bar'):
        processed_events.insert(0, 'Bar_0')
    return processed_events

if __name__ == "__main__":
    main()
