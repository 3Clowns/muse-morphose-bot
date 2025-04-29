#!/usr/bin/env python3
import os
import tempfile
import requests
import telebot
from telebot import types

# === Config ===
BOT_TOKEN     = '7694863007:AAESOkqkCeyKkj4y5ic4hJWsu9GXRvWTgkE'
INFERENCE_URL = "https://your-server.com/generate_audio"
bot = telebot.TeleBot(BOT_TOKEN)

# In‐memory state: chat_id → { upload_path, rhythm, polyphony }
user_state = {}

# helper to build a 0–7 inline keyboard
def make_keyboard(prefix):
    kb = types.InlineKeyboardMarkup()
    row = []
    for i in range(8):
        btn = types.InlineKeyboardButton(str(i), callback_data=f"{prefix}:{i}")
        row.append(btn)
        if len(row) == 4:
            kb.row(*row)
            row = []
    if row:
        kb.row(*row)
    return kb

# /start
@bot.message_handler(commands=['start'])
def cmd_start(m):
    bot.reply_to(m,
        "👋 Send me an audio or MIDI file, then choose:\n"
        "1) Rhythmic Intensity (0–7)\n"
        "2) Polyphony (0–7)\n"
        "I’ll return a MuseMorphose-styled audio clip."
    )

# receive audio or document
@bot.message_handler(content_types=['audio','document','voice'])
def handle_file(m):
    f = m.audio or m.document or m.voice
    if not f:
        return bot.reply_to(m, "Please send an audio or MIDI file.")
    # get the file info
    file_info = bot.get_file(f.file_id)
    ext = os.path.splitext(file_info.file_path)[1] or f".{f.mime_type.split('/')[-1]}"
    # download bytes
    file_bytes = bot.download_file(file_info.file_path)
    # write to temp file
    fd, path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    with open(path, 'wb') as fp:
        fp.write(file_bytes)
    # store state
    user_state[m.chat.id] = {'upload_path': path}
    # ask for rhythmic intensity
    bot.send_message(
        m.chat.id,
        "Choose Rhythmic Intensity (0=sparse,7=busy):",
        reply_markup=make_keyboard("RHY")
    )
# callback for rhythm
@bot.callback_query_handler(func=lambda cb: cb.data.startswith("RHY:"))
def on_rhythm(cb):
    chat = cb.message.chat.id
    val = int(cb.data.split(":")[1])
    user_state.setdefault(chat, {})['rhythm'] = val
    bot.edit_message_text(f"Rhythmic Intensity = {val}\nNow pick Polyphony (0=mono,7=thick):",
                          chat, cb.message.message_id,
                          reply_markup=make_keyboard("POLY"))

# callback for polyphony → run inference
@bot.callback_query_handler(func=lambda cb: cb.data.startswith("POLY:"))
def on_poly(cb):
    chat = cb.message.chat.id
    val = int(cb.data.split(":")[1])
    st = user_state.get(chat, {})
    st['polyphony'] = val
    # clean up UI
    bot.edit_message_text(f"Polyphony = {val}\nGenerating… please wait",
                          chat, cb.message.message_id)
    # prepare request
    files = {'file': open(st['upload_path'], 'rb')}
    data  = {'rhythmic_intensity': st['rhythm'], 'polyphony': st['polyphony']}
    try:
        resp = requests.post(INFERENCE_URL, files=files, data=data, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        return bot.send_message(chat, f"❌ Generation error: {e}")
    # save and send
    fd, outp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with open(outp, 'wb') as fp:
        fp.write(resp.content)
    bot.send_audio(chat, audio=open(outp,'rb'),
                   caption=f"Here’s your MuseMorphose result 🎹\n"
                           f"Rhythm={st['rhythm']},Poly={st['polyphony']}")
    # cleanup
    os.remove(st['upload_path'])
    os.remove(outp)
    user_state.pop(chat, None)

# fallback /cancel
@bot.message_handler(commands=['cancel'])
def cmd_cancel(m):
    st = user_state.pop(m.chat.id, None)
    if st and 'upload_path' in st:
        try: os.remove(st['upload_path'])
        except: pass
    bot.reply_to(m, "Cancelled. Send another file anytime!")

if __name__ == "__main__":
    bot.infinity_polling()
