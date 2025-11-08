#!/usr/bin/env python3
"""
iwcm_master_render.py
Combined trainer + Lichess bot + live fine-tune + Flask dashboard.

Usage examples (local / Colab / Render):
  # Train from folder:
  python iwcm_master_render.py --prepare --pgn_dir ./pgns
  python iwcm_master_render.py --train --epochs 3 --batch 256

  # Run bot + server (Render):
  export LICHESS_TOKEN="your_bot_token"
  python iwcm_master_render.py --serve --bot

  # Or run everything (not recommended on Render, train in Colab first):
  python iwcm_master_render.py --train --pgn_dir ./pgns --epochs 2 --bot --serve

Files this script uses/produces:
  - model.h5
  - vocab.npz (keys: moves, token_to_id, id_to_token)
  - optional: pgns.zip or pgns/ folder
"""
# NOTE: keep this file single-file for Render deployment convenience.

import os
import sys
import time
import io
import zipfile
import json
import random
import tempfile
import threading
import argparse
import traceback
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional

# Basic dependencies
import numpy as np
import chess
import chess.pgn

# Try TensorFlow imports; provide meaningful error if missing
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as e:
    tf = None
    keras = None
    layers = None
    print("WARNING: TensorFlow not available. Training & model loading will fail until TF is installed.", file=sys.stderr)

# Try berserk for Lichess
try:
    import berserk
except Exception:
    berserk = None
    print("WARNING: berserk not installed; bot features disabled until 'pip install berserk'.")

# Flask for minimal dashboard
try:
    from flask import Flask, jsonify, render_template_string
except Exception:
    Flask = None
    print("WARNING: Flask not installed; serve option will fail.")

# ================
# CONFIG DEFAULTS
# ================
DEFAULT_MODEL = "model.h5"
DEFAULT_VOCAB = "vocab.npz"
DEFAULT_SEQ_LEN = 48           # context size for model
DEFAULT_EMBED_DIM = 192        # medium fidelity
DEFAULT_FF_DIM = 256
DEFAULT_HEADS = 3
DEFAULT_BATCH = 256
DEFAULT_EPOCHS = 1024
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_TOP_K = 5
FINE_TUNE_INTERVAL = 60 * 5    # seconds between incremental fine-tune runs (5 minutes)
FINE_TUNE_BATCH = 32
FINE_TUNE_EPOCHS = 256
MODEL_SAVE_INTERVAL = 60 * 2   # autosave model every 2 minutes if updated
SAMPLES_PER_FINE_TUNE = 2000   # cap of collected samples to fine-tune on in one run

# Repro
RANDOM_SEED = 1337
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Global runtime state
_live_game_samples_lock = threading.Lock()
_live_game_samples: List[List[str]] = []  # list of move-sequence lists (UCI)
_last_model_save = 0
_model_dirty = False

# ================
# UTILITIES
# ================
def safe_makedirs(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_vocab_canonical(moves: List[str], token_to_id: Dict[str,int], id_to_token: Dict[int,str], path: str):
    # Save using np.savez_compressed and include both dicts (pickled objects inside .npz)
    safe_makedirs(os.path.dirname(path) or ".")
    np.savez_compressed(path,
                        moves=np.array(moves, dtype=object),
                        token_to_id=token_to_id,
                        id_to_token=id_to_token)
    print(f"[vocab] Saved canonical vocab to {path} ({len(moves)} moves).")

def load_vocab_flexible(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    # Preferred names
    moves = None
    token_to_id = None
    id_to_token = None
    if "moves" in data:
        moves = data["moves"].tolist()
    if "token_to_id" in data:
        token_to_id = data["token_to_id"].tolist() if isinstance(data["token_to_id"], np.ndarray) else data["token_to_id"]
    if "id_to_token" in data:
        id_to_token = data["id_to_token"].tolist() if isinstance(data["id_to_token"], np.ndarray) else data["id_to_token"]
    # Fallback: first array-like
    if moves is None:
        for k in keys:
            v = data[k]
            if isinstance(v, np.ndarray) and v.dtype == object:
                moves = list(v.tolist())
                break
    # Last fallback: maybe it's a dict-only save (older)
    if token_to_id is None and moves is not None:
        token_to_id = {m:i for i,m in enumerate(moves)}
    if id_to_token is None and token_to_id is not None:
        id_to_token = {i:m for m,i in token_to_id.items()}
    if moves is None:
        raise ValueError("Could not find moves array in vocab file")
    print(f"[vocab] Loaded vocab from {path}: {len(moves)} moves.")
    return moves, token_to_id, id_to_token

# ================
# PGN HANDLING
# ================
def extract_pgns_from_folder(folder: str, limit_games: Optional[int]=None) -> List[str]:
    files = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(".pgn")]
    texts = []
    for fpath in sorted(files):
        with open(fpath, "r", errors="ignore") as fh:
            content = fh.read()
        pgn_io = io.StringIO(content)
        while True:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break
            texts.append(str(game))
            if limit_games and len(texts) >= limit_games:
                return texts
    print(f"[pgn] Extracted {len(texts)} games from folder {folder}")
    return texts

def extract_pgns_from_zip(zip_path: str, limit_games: Optional[int]=None) -> List[str]:
    texts = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.lower().endswith(".pgn"):
                with z.open(name) as fh:
                    try:
                        t = fh.read().decode("utf-8", errors="ignore")
                    except Exception:
                        t = fh.read().decode("latin-1", errors="ignore")
                pgn_io = io.StringIO(t)
                while True:
                    game = chess.pgn.read_game(pgn_io)
                    if game is None:
                        break
                    texts.append(str(game))
                    if limit_games and len(texts) >= limit_games:
                        return texts
    print(f"[pgn] Extracted {len(texts)} games from zip {zip_path}")
    return texts

def pgn_text_to_uci_sequence(pgn_text: str, max_moves: Optional[int]=None) -> List[str]:
    pgn_io = io.StringIO(pgn_text)
    try:
        game = chess.pgn.read_game(pgn_io)
    except Exception:
        return []
    if game is None:
        return []
    board = game.board()
    seq = []
    for move in game.mainline_moves():
        try:
            seq.append(move.uci())
            board.push(move)
        except Exception:
            # skip problematic move
            pass
        if max_moves and len(seq) >= max_moves:
            break
    return seq

def pgn_texts_to_uci_sequences(pgn_texts: List[str], max_moves: Optional[int]=None) -> List[List[str]]:
    seqs = []
    for t in pgn_texts:
        seq = pgn_text_to_uci_sequence(t, max_moves)
        if seq:
            seqs.append(seq)
    print(f"[pgn] Converted {len(seqs)} PGN texts to {len(seqs)} UCI sequences")
    return seqs

# ================
# TOKENIZER
# ================
class MoveTokenizer:
    def __init__(self):
        self.token_to_id: Dict[str,int] = {}
        self.id_to_token: Dict[int,str] = {}
        self.moves: List[str] = []

    def fit_on_sequences(self, sequences: List[List[str]], min_freq:int=1):
        freq = defaultdict(int)
        for seq in sequences:
            for mv in seq:
                freq[mv] += 1
        moves_sorted = sorted([m for m,c in freq.items() if c >= min_freq],
                              key=lambda m: (-freq[m], m))
        # Map moves to ids starting at 1. Reserve 0 for PAD/UNK.
        self.token_to_id = {m: i for i, m in enumerate(moves_sorted, start=1)}
        self.id_to_token = {i: m for m, i in self.token_to_id.items()}
        self.moves = moves_sorted
        print(f"[tok] Built tokenizer: vocab_size={len(self.token_to_id)} (+PAD/UNK)")

    def encode(self, seq: List[str], seq_len: int):
        ids = [self.token_to_id.get(m, 0) for m in seq[-seq_len:]]
        if len(ids) < seq_len:
            ids = [0] * (seq_len - len(ids)) + ids
        return np.array(ids, dtype=np.int32)

    def save(self, path: str):
        safe_makedirs(os.path.dirname(path) or ".")
        save_vocab_canonical(self.moves, self.token_to_id, self.id_to_token, path)

    @staticmethod
    def load(path: str):
        moves, t2i, i2t = load_vocab_flexible(path)
        tok = MoveTokenizer()
        tok.moves = moves
        tok.token_to_id = t2i
        tok.id_to_token = i2t
        return tok

# ================
# MODEL (small transformer-ish)
# ================
def build_transformer_model(vocab_size:int, seq_len:int=DEFAULT_SEQ_LEN, embed_dim:int=DEFAULT_EMBED_DIM,
                            num_heads:int=DEFAULT_HEADS, ff_dim:int=DEFAULT_FF_DIM, dropout=0.1):
    if tf is None:
        raise RuntimeError("TensorFlow not available; cannot build model.")
    inputs = keras.Input(shape=(seq_len,), dtype="int32", name="moves_input")
    x = layers.Embedding(input_dim=vocab_size + 1, output_dim=embed_dim, mask_zero=True, name="embed")(inputs)
    # single attention block
    att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads, name="mha")(x, x)
    x = layers.LayerNormalization(epsilon=1e-6, name="ln1")(x + att)
    ff = layers.Dense(ff_dim, activation="relu", name="ff1")(x)
    ff = layers.Dense(embed_dim, name="ff2")(ff)
    x = layers.LayerNormalization(epsilon=1e-6, name="ln2")(x + ff)
    x = layers.GlobalAveragePooling1D(name="pool")(x)
    outputs = layers.Dense(vocab_size + 1, activation="softmax", name="out")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="chess_imitation")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ================
# DATASET CREATION
# ================
def sequences_to_xy(seqs: List[List[str]], tokenizer: MoveTokenizer, seq_len:int, max_samples:Optional[int]=None):
    X = []
    y = []
    for seq in seqs:
        ids = [tokenizer.token_to_id.get(m, 0) for m in seq]
        for i in range(1, len(ids)):
            left = ids[max(0, i - seq_len):i]
            pad = [0] * (seq_len - len(left))
            X.append(pad + left)
            y.append(ids[i])
            if max_samples and len(X) >= max_samples:
                break
        if max_samples and len(X) >= max_samples:
            break
    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)
    print(f"[data] Built X:{X.shape} y:{y.shape}")
    return X, y

# ================
# TRAINER PIPELINE
# ================
def prepare_and_train(pgn_dir: Optional[str], pgn_zip: Optional[str], vocab_path: str, model_path: str,
                      seq_len:int=DEFAULT_SEQ_LEN, epochs:int=DEFAULT_EPOCHS, batch_size:int=DEFAULT_BATCH,
                      max_games:int=20000, quick:bool=False):
    # 1) collect pgn texts
    pgn_texts = []
    if pgn_dir and os.path.exists(pgn_dir):
        pgn_texts = extract_pgns_from_folder(pgn_dir, limit_games=max_games)
    elif pgn_zip and os.path.exists(pgn_zip):
        pgn_texts = extract_pgns_from_zip(pgn_zip, limit_games=max_games)
    else:
        raise RuntimeError("No PGNs found. Provide --pgn_dir or --pgn_zip")

    # 2) convert to sequences
    seqs = pgn_texts_to_uci_sequences(pgn_texts, max_moves=None)
    if not seqs:
        raise RuntimeError("No sequences extracted from provided PGNs.")

    # quick mode sampling
    if quick:
        sample_n = min(500, len(seqs))
        seqs = random.sample(seqs, sample_n)
        print(f"[train] Quick mode: reduced to {len(seqs)} sequences")

    # 3) build tokenizer & save vocab
    tokenizer = MoveTokenizer()
    tokenizer.fit_on_sequences(seqs, min_freq=1)
    tokenizer.save(vocab_path)

    # 4) dataset
    X, y = sequences_to_xy(seqs, tokenizer, seq_len, max_samples=None)

    # optionally subsample huge datasets for speed
    max_samples = 200000
    if X.shape[0] > max_samples:
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[idx]
        y = y[idx]
        print(f"[train] Subsampled to {X.shape[0]} training samples")

    # 5) build model
    model = build_transformer_model(vocab_size=len(tokenizer.token_to_id), seq_len=seq_len)
    print(model.summary())

    # 6) fit
    model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.05)

    # 7) save model
    model.save(model_path)
    print(f"[train] Saved model to {model_path}")
    return model, tokenizer

# ================
# UTILITY: encode history
# ================
def encode_history_for_inference(history: List[str], tokenizer: MoveTokenizer, seq_len:int):
    ids = [tokenizer.token_to_id.get(m, 0) for m in history[-seq_len:]]
    if len(ids) < seq_len:
        ids = [0] * (seq_len - len(ids)) + ids
    return np.array([ids], dtype=np.int32)

# ================
# PICK MOVE WITH MODEL
# ================
def pick_move_from_model(model: keras.Model, tokenizer: MoveTokenizer, board: chess.Board,
                         history: List[str], temperature: float = 1.0, top_k: int = DEFAULT_TOP_K):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    X = encode_history_for_inference(history, tokenizer, seq_len=DEFAULT_SEQ_LEN)
    preds = model.predict(X, verbose=0)[0]  # vector length = vocab+1
    # compute score per legal move
    scores = []
    legal_idx = []
    for mv in legal_moves:
        u = mv.uci()
        idx = tokenizer.token_to_id.get(u, None)
        if idx is None or idx >= preds.shape[0]:
            s = 1e-12
        else:
            s = float(preds[idx])
        scores.append(s)
        legal_idx.append((mv, idx))
    scores = np.array(scores, dtype=float)
    if np.isnan(scores.sum()) or scores.sum() <= 0:
        probs = np.ones_like(scores) / len(scores)
    else:
        logits = np.log(scores + 1e-12) / (temperature if temperature > 0 else 1.0)
        exps = np.exp(logits - np.max(logits))
        probs = exps / np.sum(exps)
        if top_k and top_k < len(probs):
            topk_positions = np.argsort(probs)[-top_k:]
            mask = np.zeros_like(probs)
            mask[topk_positions] = 1.0
            probs = probs * mask
            if probs.sum() <= 0:
                probs = np.ones_like(probs) / len(probs)
            else:
                probs = probs / probs.sum()
    choice = np.random.choice(len(legal_moves), p=probs)
    return legal_moves[choice]

# ================
# LIVE FINE-TUNE LOOP
# ================
def collect_finished_game_for_finetune(move_seq: List[str]):
    # Called by game handler when a game finishes (played by our bot); we push its move sequence to the buffer.
    with _live_game_samples_lock:
        _live_game_samples.append(move_seq.copy())
        # cap buffer to avoid memory explosion
        if len(_live_game_samples) > 10000:
            _live_game_samples.pop(0)
    print(f"[fine] Collected finished game for fine-tune, buffer size: {len(_live_game_samples)}")

def fine_tune_worker(model_path: str, vocab_path: str, seq_len:int=DEFAULT_SEQ_LEN,
                     interval:int=FINE_TUNE_INTERVAL, max_samples:int=SAMPLES_PER_FINE_TUNE):
    global _model_dirty, _last_model_save
    if tf is None:
        print("[fine] TensorFlow not available; skipping fine-tune worker.")
        return
    print("[fine] Fine-tune worker started, checking buffer every", interval, "seconds")
    while True:
        time.sleep(interval)
        with _live_game_samples_lock:
            local_samples = list(_live_game_samples)
            # After copying we clear global buffer (we will process them)
            _live_game_samples.clear()
        if not local_samples:
            print("[fine] No new live games to fine-tune on.")
            continue
        # Load tokenizer and model
        try:
            tokenizer = MoveTokenizer.load(vocab_path)
        except Exception as e:
            print("[fine] Failed to load vocab for fine-tune:", e)
            continue
        try:
            model = keras.models.load_model(model_path)
        except Exception as e:
            print("[fine] Failed to load model for fine-tune:", e)
            continue
        # Convert collected games to sequences and then to X,y
        # Use up to max_samples combined from multiple games
        seqs = local_samples
        X_list = []
        y_list = []
        for seq in seqs:
            ids = [tokenizer.token_to_id.get(m, 0) for m in seq]
            for i in range(1, len(ids)):
                left = ids[max(0, i - seq_len):i]
                pad = [0] * (seq_len - len(left))
                X_list.append(pad + left)
                y_list.append(ids[i])
                if len(X_list) >= max_samples:
                    break
            if len(X_list) >= max_samples:
                break
        if not X_list:
            print("[fine] No usable samples extracted from live games.")
            continue
        X = np.array(X_list, dtype=np.int32)
        y = np.array(y_list, dtype=np.int32)
        print(f"[fine] Fine-tuning on {len(X)} samples (from {len(seqs)} games)")
        try:
            # tiny fine-tune: small lr, few epochs
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                          loss="sparse_categorical_crossentropy")
            model.fit(X, y, batch_size=FINE_TUNE_BATCH, epochs=FINE_TUNE_EPOCHS, verbose=1)
            model.save(model_path)
            _model_dirty = False
            _last_model_save = time.time()
            print("[fine] Fine-tune complete and model saved.")
        except Exception as e:
            print("[fine] Error during fine-tune:", e)
            traceback.print_exc()

# ================
# LICHESS BOT
# ================
class ImitationLichessBot:
    def __init__(self, token: str, model_path: str, vocab_path: str,
                 seq_len:int=DEFAULT_SEQ_LEN, top_k:int=DEFAULT_TOP_K,
                 temperature:float=1.0, accept_variants:Tuple[str,...]=("standard","fromPosition")):
        if berserk is None:
            raise RuntimeError("berserk is not installed; please pip install berserk to run bot.")
        self.token = token
        self.session = berserk.TokenSession(token)
        self.client = berserk.Client(session=self.session, timeout=90)
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.seq_len = seq_len
        self.top_k = top_k
        self.temperature = temperature
        self.accept_variants = accept_variants
        self.running = True
        # load model & tokenizer
        self._reload_model_and_tokenizer()
        # stats
        self.active_games = {}  # game_id -> dict (moves, last_thought,...)
        self.threads = {}

    def _reload_model_and_tokenizer(self):
        # Loads model and tokenizer; called on init and possibly periodically
        if tf is None:
            raise RuntimeError("TensorFlow not available; bot cannot load model.")
        self.tokenizer = MoveTokenizer.load(self.vocab_path)
        try:
            self.model = keras.models.load_model(self.model_path)
            print("[bot] Model loaded.")
        except Exception as e:
            print("[bot] Failed to load model:", e)
            raise

    def _accept_or_decline(self, challenge):
        vid = challenge["id"]
        var = challenge["variant"]["key"]
        try:
            if var in self.accept_variants:
                self.client.bots.accept_challenge(vid)
                print(f"[bot] Accepted challenge {vid} variant={var}")
            else:
                self.client.bots.decline_challenge(vid)
                print(f"[bot] Declined challenge {vid} variant={var}")
        except Exception as e:
            print("[bot] Error accepting/declining challenge:", e)

    def _game_worker(self, game_id: str, my_color: chess.Color):
        print(f"[bot] Game thread started for {game_id} color={'white' if my_color==chess.WHITE else 'black'}")
        board = chess.Board()
        stream = None
        try:
            stream = self.client.bots.stream_game_state(game_id)
            for event in stream:
                try:
                    if event["type"] in ("gameFull", "gameState"):
                        state = event.get("state", event)
                        moves_str = state.get("moves", "")
                        moves = moves_str.split() if moves_str else []
                        # rebuild board
                        board = chess.Board()
                        for mv in moves:
                            try:
                                board.push_uci(mv)
                            except Exception:
                                pass
                        # update active metadata
                        self.active_games[game_id] = {"moves": moves.copy(), "last_thought": None, "result": None}
                        # if game over
                        if board.is_game_over():
                            res = state.get("status", "finished")
                            self.active_games[game_id]["result"] = res
                            print(f"[bot] Game {game_id} ended: {res}")
                            # collect finished game for fine-tune
                            collect_finished_game_for_finetune(moves.copy())
                            break
                        # our turn?
                        if board.turn == my_color:
                            # reload model occasionally to pick up incremental fine-tune
                            try:
                                # lightweight: reload model from disk if file changed (cheapish)
                                self.model = keras.models.load_model(self.model_path)
                            except Exception:
                                pass
                            print(f"[bot:{game_id}] Thinking... moves played: {len(moves)}")
                            mv = pick_move_from_model(self.model, self.tokenizer, board, moves,
                                                      temperature=self.temperature, top_k=self.top_k)
                            if mv is None:
                                print(f"[bot:{game_id}] No legal move (game over?)")
                                break
                            try:
                                self.client.bots.make_move(game_id, mv.uci())
                                print(f"[bot:{game_id}] Played {mv.uci()}")
                            except Exception as e:
                                print(f"[bot:{game_id}] Failed to make move: {e}")
                                # fallback: random legal
                                try:
                                    legal = list(board.legal_moves)
                                    if legal:
                                        fallback = next(iter(legal))
                                        self.client.bots.make_move(game_id, fallback.uci())
                                        print(f"[bot:{game_id}] Fallback played {fallback.uci()}")
                                except Exception:
                                    pass
                except Exception as e:
                    print(f"[bot:{game_id}] Error processing event: {e}")
        except Exception as e:
            print(f"[bot:{game_id}] Stream error: {e}")
        finally:
            # stream ended or game finished
            try:
                collect_finished_game_for_finetune(self.active_games.get(game_id, {}).get("moves", []))
            except Exception:
                pass
            print(f"[bot] Game thread exiting for {game_id}")
            # cleanup
            if game_id in self.active_games:
                del self.active_games[game_id]

    def run_event_loop(self):
        print("[bot] Starting incoming events stream")
        backoff = 1.0
        while self.running:
            try:
                for event in self.client.bots.stream_incoming_events():
                    if event["type"] == "challenge":
                        self._accept_or_decline(event["challenge"])
                    elif event["type"] == "gameStart":
                        game_id = event["game"]["id"]
                        color_str = event["game"]["color"]
                        my_color = chess.WHITE if color_str == "white" else chess.BLACK
                        th = threading.Thread(target=self._game_worker, args=(game_id, my_color), daemon=True)
                        th.start()
                        self.threads[game_id] = th
                backoff = 1.0
            except berserk.exceptions.ResponseError as re:
                print("[bot] Lichess API response error:", re)
                # handle 401, 429 etc
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
            except Exception as e:
                print("[bot] Unexpected exception in event loop:", e)
                traceback.print_exc()
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def stop(self):
        self.running = False

# ================
# FLASK DASHBOARD for Render
# ================
def run_flask_app(bot_obj: Optional[ImitationLichessBot], port:int=10000):
    if Flask is None:
        raise RuntimeError("Flask not installed.")
    app = Flask("iwcm_dashboard")
    @app.route("/")
    def home():
        return render_template_string("""
        <h1>Imitation Bot Dashboard</h1>
        <p>Status: <strong>Running</strong></p>
        <p>Model: {{model}}</p>
        <p>Vocab size: {{vsize}}</p>
        <p>Active games: <span id="active">loading...</span></p>
        <pre id="dump"></pre>
        <script>
          async function poll(){
            const res = await fetch('/status');
            const j = await res.json();
            document.getElementById('active').innerText = Object.keys(j).length;
            document.getElementById('dump').innerText = JSON.stringify(j, null, 2);
          }
          setInterval(poll, 2000);
          poll();
        </script>
        """, model=os.path.basename(DEFAULT_MODEL), vsize="(loaded at startup)")
    @app.route("/status")
    def status():
        if bot_obj:
            return jsonify(bot_obj.active_games)
        else:
            return jsonify({})
    print(f"[flask] Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port, threaded=True)

# ================
# MAIN CLI
# ================
def main():
    parser = argparse.ArgumentParser(description="Imitation Bot: train + serve + live fine-tune")
    parser.add_argument("--pgn_dir", default="pgns", help="folder with .pgn files")
    parser.add_argument("--pgn_zip", default="pgns.zip", help="zip of pgns (optional)")
    parser.add_argument("--vocab", default=DEFAULT_VOCAB, help="vocab.npz")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model.h5")
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--train", action="store_true", help="train from PGNs")
    parser.add_argument("--prepare", action="store_true", help="prepare vocab from PGNs")
    parser.add_argument("--bot", action="store_true", help="run lichess bot")
    parser.add_argument("--serve", action="store_true", help="run flask server (Render)")
    parser.add_argument("--fine_tune_worker", action="store_true", help="run fine-tune worker (background)")
    parser.add_argument("--quick", action="store_true", help="quick mode (small sample)")
    args = parser.parse_args()

    # Prep: unzip zip if present
    if args.pgn_zip and os.path.exists(args.pgn_zip) and not os.path.exists(args.pgn_dir):
        try:
            print("[main] Extracting pgns.zip to pgns/")
            safe_makedirs(args.pgn_dir)
            with zipfile.ZipFile(args.pgn_zip, "r") as z:
                z.extractall(args.pgn_dir)
            print("[main] Extraction complete.")
        except Exception as e:
            print("[main] Could not extract zip:", e)

    # Prepare only: build vocab
    if args.prepare:
        pgn_texts = []
        if os.path.exists(args.pgn_dir):
            pgn_texts = extract_pgns_from_folder(args.pgn_dir, limit_games=50000)
        elif os.path.exists(args.pgn_zip):
            pgn_texts = extract_pgns_from_zip(args.pgn_zip, limit_games=50000)
        else:
            print("[main] No PGNs found for --prepare.")
            sys.exit(1)
        seqs = pgn_texts_to_uci_sequences(pgn_texts)
        tok = MoveTokenizer()
        tok.fit_on_sequences(seqs)
        tok.save(args.vocab)
        print("[main] Vocab prepared and saved to", args.vocab)
        # if only prepare requested, exit
        if not args.train and not args.bot and not args.serve:
            return

    model_obj = None
    tokenizer = None

    # Train if requested (or if no model exists)
    need_train = args.train or (not os.path.exists(args.model))
    if need_train:
        print("[main] Starting training pipeline...")
        model_obj, tokenizer = prepare_and_train(pgn_dir=args.pgn_dir if os.path.exists(args.pgn_dir) else None,
                                                 pgn_zip=args.pgn_zip if os.path.exists(args.pgn_zip) else None,
                                                 vocab_path=args.vocab, model_path=args.model,
                                                 seq_len=args.seq_len, epochs=args.epochs, batch_size=args.batch,
                                                 quick=args.quick)
    else:
        # Load model & vocab
        if tf is None:
            print("[main] TensorFlow not available; cannot load existing model.")
        else:
            try:
                tokenizer = MoveTokenizer.load(args.vocab)
                model_obj = keras.models.load_model(args.model)
                print("[main] Loaded model and vocab.")
            except Exception as e:
                print("[main] Failed to load existing model/vocab:", e)

    # If bot requested, start bot thread
    bot_obj = None
    if args.bot:
        token = os.environ.get("LICHESS_TOKEN") or os.environ.get("Lichess_token") or os.environ.get("LICHESS_API_TOKEN")
        if not token:
            print("[main] Missing LICHESS_TOKEN environment variable; bot cannot start.")
        else:
            bot_obj = ImitationLichessBot(token=token, model_path=args.model, vocab_path=args.vocab,
                                          seq_len=args.seq_len)
            bot_thread = threading.Thread(target=bot_obj.run_event_loop, daemon=True)
            bot_thread.start()
            print("[main] Bot event loop started in background.")

    # Fine-tune worker (collects live games & periodically fine-tunes the model)
    if args.fine_tune_worker:
        if tf is None:
            print("[main] TF not available; skipping fine-tune worker.")
        else:
            fine_thread = threading.Thread(target=fine_tune_worker, args=(args.model, args.vocab), daemon=True)
            fine_thread.start()
            print("[main] Fine-tune worker started in background.")

    # Serve flask app (blocking)
    if args.serve:
        run_flask_app(bot_obj, port=int(os.environ.get("PORT", 10000)))
    else:
        # If not serving, keep main thread alive if bot/fine workers running
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("Interrupted; exiting.")
            if bot_obj:
                bot_obj.stop()
            sys.exit(0)

if __name__ == "__main__":
    main()
