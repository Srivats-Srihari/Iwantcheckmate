#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py - Lichess player-imitation bot (single-file deployment)
Author: generated for you
Description:
  - Loads your model.h5 and vocab.npz (if present)
  - Connects to Lichess via berserk, accepts standard/fromPosition challenges
  - Spawns per-game handler threads that reconstruct board state from moves
  - Runs model inference (if MODEL available) using several encoders
  - Chooses legal move from model logits or falls back to statistical imitation
  - Dashboard and admin endpoints (Flask)
  - Verbose logging and diagnostics
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
import random
import threading
import traceback
from collections import Counter, defaultdict, deque
from typing import Dict, Any, List, Optional, Tuple

# third-party libs
try:
    import numpy as np
except Exception as e:
    print("ERROR: numpy is required. Install with `pip install numpy`.")
    raise

try:
    import berserk
except Exception:
    print("ERROR: berserk is required. Install with `pip install berserk`.")
    raise

try:
    import chess
    import chess.pgn
except Exception:
    print("ERROR: python-chess is required. Install with `pip install python-chess`.")
    raise

# optional TensorFlow/Keras model support
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as tf_load_model
except Exception:
    tf = None
    tf_load_model = None

# flask for dashboard & admin
try:
    from flask import Flask, jsonify, request, send_file, render_template_string, abort
except Exception:
    print("ERROR: Flask is required. Install with `pip install Flask`.")
    raise

# ============================
# Configuration (env variables)
# ============================
# Lichess token (required)
LICHESS_TOKEN_ENV = os.environ.get("LICHESS_TOKEN_ENV", "Lichess_token")
LICHESS_TOKEN = os.environ.get(LICHESS_TOKEN_ENV) or os.environ.get("LICHESS_TOKEN")

if not LICHESS_TOKEN:
    # we won't crash here so the code can still run for testing without Lichess connection,
    # but you must set this env var to actually accept challenges.
    print("WARNING: Lichess token not found in env var", LICHESS_TOKEN_ENV)

# Paths
MODEL_PATH = os.environ.get("MODEL_PATH", "model.h5")
VOCAB_PATH = os.environ.get("VOCAB_PATH", "vocab.npz")
STORAGE_DIR = os.environ.get("STORAGE_DIR", "saved_games")

# Inference and imitation tuning
ARGMAX = os.environ.get("ARGMAX", "false").lower() in ("1", "true", "yes")
TOP_K = int(os.environ.get("TOP_K", "32"))
TEMPERATURE = float(os.environ.get("TEMP", "1.0"))

# Debug/testing toggles
DEBUG_FORCE_RANDOM = os.environ.get("DEBUG_FORCE_RANDOM", "false").lower() in ("1", "true", "yes")
DEBUG_FORCE_OPENING = os.environ.get("DEBUG_FORCE_OPENING", "")  # e.g. "e2e4 e7e5"
DEBUG_LOG_PROBS = os.environ.get("DEBUG_LOG_PROBS", "true").lower() in ("1", "true", "yes")

# Streaming + retry/backoff policy
BACKOFF_BASE = float(os.environ.get("BACKOFF_BASE", "1.0"))
BACKOFF_MAX = float(os.environ.get("BACKOFF_MAX", "60.0"))
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "6"))

# misc
PORT = int(os.environ.get("PORT", "10000"))
MOVE_DELAY = float(os.environ.get("MOVE_DELAY", "0.10"))
HEARTBEAT_TIMEOUT = int(os.environ.get("HEARTBEAT_TIMEOUT", "180"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "128"))

# ensure dirs
os.makedirs(STORAGE_DIR, exist_ok=True)

# =============
# Logging helper
# =============
def log(*args, **kwargs):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(stamp, *args, **kwargs)
    sys.stdout.flush()

# ========================
# Global application state
# ========================
MODEL = None
MODEL_INPUT_SHAPE = None
IDX2MOVE: List[str] = []
MOVE2IDX: Dict[str, int] = {}
VOCAB_SIZE = 0
MOVE_FREQ: Counter = Counter()

# games dictionary: game_id -> info dict
GAMES: Dict[str, Dict[str, Any]] = {}
GAMES_LOCK = threading.Lock()

# per-game thread references (so we can monitor/join if needed)
GAME_THREADS: Dict[str, threading.Thread] = {}
GAME_THREADS_LOCK = threading.Lock()

# berserk global client (created when token present)
_GLOBAL_CLIENT: Optional[berserk.Client] = None
_GLOBAL_CLIENT_LOCK = threading.Lock()

# event thread
_event_thread: Optional[threading.Thread] = None
_event_thread_lock = threading.Lock()
_stop_flag = threading.Event()

# ============================
# Utility: exponential backoff
# ============================
def retry_call(func, *args, max_attempts: int = MAX_ATTEMPTS, base_wait: float = BACKOFF_BASE, **kwargs):
    wait = base_wait
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            s = str(e)
            log(f"[retry] attempt {attempt}/{max_attempts} failed: {s}")
            if "429" in s or "Too Many Requests" in s:
                log(f"[retry] rate limited — sleeping {wait}s")
                time.sleep(wait)
                wait = min(wait * 2.0, BACKOFF_MAX)
                continue
            if attempt == max_attempts:
                log("[retry] reached max attempts — raising")
                raise
            time.sleep(min(wait, BACKOFF_MAX))
            wait = min(wait * 2.0, BACKOFF_MAX)
    raise RuntimeError("retry_call exhausted")

# =====================================
# Create and return a berserk client
# =====================================
def create_global_client():
    global _GLOBAL_CLIENT
    with _GLOBAL_CLIENT_LOCK:
        if _GLOBAL_CLIENT is None:
            if not LICHESS_TOKEN:
                raise RuntimeError("Lichess token not set. Set env var " + LICHESS_TOKEN_ENV)
            session = berserk.TokenSession(LICHESS_TOKEN)
            _GLOBAL_CLIENT = berserk.Client(session=session)
            log("Created global berserk client")
    return _GLOBAL_CLIENT

# =================================================
# Vocab loader: load vocab.npz and construct maps
# =================================================
def load_vocab_npz(path: str = VOCAB_PATH):
    """
    Loads a vocab.npz file with a 'moves' key (UCI strings like 'e2e4').
    It is robust to common variations:
      - keys: 'moves', 'vocab', 'idx2move', 'move2idx', etc.
    On success populates IDX2MOVE, MOVE2IDX, VOCAB_SIZE, MOVE_FREQ.
    """
    global IDX2MOVE, MOVE2IDX, VOCAB_SIZE, MOVE_FREQ
    if not os.path.exists(path):
        log(f"[vocab] file not found: {path}")
        IDX2MOVE, MOVE2IDX, VOCAB_SIZE = [], {}, 0
        MOVE_FREQ = Counter()
        return

    data = np.load(path, allow_pickle=True)
    keys = list(data.files)
    log("[vocab] npz keys:", keys)

    moves = None
    if "moves" in data.files:
        moves = data["moves"]
    elif "vocab" in data.files:
        moves = data["vocab"]
    elif "idx2move" in data.files:
        moves = data["idx2move"]
    else:
        # fallback: pick the first array-like that looks like tokens
        for k in keys:
            arr = data[k]
            try:
                cand = [str(x) for x in arr.tolist()]
                # heuristic: tokens are short strings
                if len(cand) > 50 and all(1 <= len(s) <= 8 for s in cand[:50]):
                    moves = arr
                    log(f"[vocab] fallback using key {k}")
                    break
            except Exception:
                continue

    if moves is None:
        log("[vocab] failed to find move list in npz")
        IDX2MOVE, MOVE2IDX, VOCAB_SIZE = [], {}, 0
        MOVE_FREQ = Counter()
        return

    # convert to python list of strings
    try:
        IDX2MOVE = [str(x) for x in moves.tolist()]
    except Exception:
        try:
            IDX2MOVE = [str(x) for x in moves]
        except Exception:
            IDX2MOVE = []

    MOVE2IDX = {m: i for i, m in enumerate(IDX2MOVE)}
    VOCAB_SIZE = len(IDX2MOVE)
    # frequency: if moves came from a large flattened list, use counts
    MOVE_FREQ = Counter(IDX2MOVE)

    log(f"[vocab] loaded {VOCAB_SIZE} tokens")

# =================================================
# Model loader: load Keras model.h5 if TF is present
# =================================================
def load_keras_model(path: str = MODEL_PATH):
    """
    Loads Keras model using tf.keras.models.load_model.
    Sets MODEL global and tries to infer input shape.
    """
    global MODEL, MODEL_INPUT_SHAPE
    if tf_load_model is None:
        log("[model] TensorFlow/Keras not available in this runtime. Skipping model load.")
        MODEL = None
        MODEL_INPUT_SHAPE = None
        return

    if not os.path.exists(path):
        log(f"[model] model file not found: {path}")
        MODEL = None
        MODEL_INPUT_SHAPE = None
        return

    try:
        log(f"[model] loading model from {path} ...")
        MODEL = tf_load_model(path)
        # infer input shape
        try:
            inp = MODEL.inputs[0].shape
            if hasattr(inp, "as_list"):
                dims = inp.as_list()
            else:
                dims = tuple(inp)
            MODEL_INPUT_SHAPE = tuple(dims)
        except Exception:
            MODEL_INPUT_SHAPE = None
        log(f"[model] loaded. input_shape={MODEL_INPUT_SHAPE}")
    except Exception as e:
        log("[model] load failed:", e)
        MODEL = None
        MODEL_INPUT_SHAPE = None

# ============================
# Initialize model & vocab
# ============================
load_vocab_npz(VOCAB_PATH)
load_keras_model(MODEL_PATH)

# ------------------------------
# Encoders (sequence, planes, one-hot)
# ------------------------------
PIECE_TO_PLANE = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
}

def encode_sequence(board: chess.Board, seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    Returns shape (1, seq_len) int32 array where tokens are idx+1 (0 reserved for padding).
    """
    moves = [m.uci() for m in board.move_stack]
    toks = []
    for mv in moves[-seq_len:]:
        idx = MOVE2IDX.get(mv)
        toks.append((idx + 1) if idx is not None else 0)
    if len(toks) < seq_len:
        toks = [0] * (seq_len - len(toks)) + toks
    return np.array(toks, dtype=np.int32).reshape(1, seq_len)

def encode_planes(board: chess.Board) -> np.ndarray:
    """
    Returns shape (1, 8*8*12) float32 array representing piece planes.
    board.piece_map() uses square indices; we fill rows by rank 0..7.
    """
    arr = np.zeros((8, 8, 12), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        r = chess.square_rank(sq)
        c = chess.square_file(sq)
        plane = PIECE_TO_PLANE.get(piece.symbol())
        if plane is not None:
            arr[r, c, plane] = 1.0
    return arr.reshape(1, -1).astype(np.float32)

def encode_onehot_vec(board: chess.Board) -> np.ndarray:
    """
    Returns (1, VOCAB_SIZE) vector where indices corresponding to moves in history are incremented.
    Normalized to sum 1 if nonzero.
    """
    vec = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
    for mv in [m.uci() for m in board.move_stack]:
        idx = MOVE2IDX.get(mv)
        if idx is not None:
            vec[0, idx] += 1.0
    s = vec.sum()
    if s > 0:
        vec /= s
    return vec

# ------------------------------
# Model inference wrapper
# ------------------------------
def normalize_probs(arr: np.ndarray) -> np.ndarray:
    """Turn logits or scores into a probability vector."""
    a = np.array(arr, dtype=np.float64).flatten()
    if a.size == 0:
        raise ValueError("empty model output")
    # if already non-negative and sums approx 1, keep it
    if (a >= 0).all() and abs(a.sum() - 1.0) < 1e-6:
        return a
    # use softmax
    ex = np.exp(a - np.max(a))
    return ex / (ex.sum() + 1e-12)

def infer_move_probs(board: chess.Board) -> np.ndarray:
    """
    Try to infer model output probabilities using sequence, planes, or one-hot encoders.
    Returns a length-VOCAB_SIZE array summing to 1.
    Raises on failure.
    """
    if MODEL is None:
        raise RuntimeError("No MODEL loaded")

    errors = []
    # sequence
    try:
        seq = encode_sequence(board)
        out = MODEL.predict(seq, verbose=0)
        raw = np.array(out[0]).flatten()
        if raw.size == VOCAB_SIZE:
            return normalize_probs(raw)
        else:
            errors.append(("seq_len_mismatch", raw.size))
    except Exception as e:
        errors.append(("seq", str(e)))

    # planes
    try:
        planes = encode_planes(board)
        out = MODEL.predict(planes, verbose=0)
        raw = np.array(out[0]).flatten()
        if raw.size == VOCAB_SIZE:
            return normalize_probs(raw)
        else:
            errors.append(("planes_len_mismatch", raw.size))
    except Exception as e:
        errors.append(("planes", str(e)))

    # one-hot
    try:
        vec = encode_onehot_vec(board)
        out = MODEL.predict(vec, verbose=0)
        raw = np.array(out[0]).flatten()
        if raw.size == VOCAB_SIZE:
            return normalize_probs(raw)
        else:
            errors.append(("onehot_len_mismatch", raw.size))
    except Exception as e:
        errors.append(("onehot", str(e)))

    log("[infer] failed encoders/len mismatches:", errors)
    raise RuntimeError("model inference failed; see logs")

# ------------------------------
# Choose a legal move given probs
# ------------------------------
def softmax_with_temp(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    if temp <= 0:
        temp = 1e-6
    a = x / temp
    a = a - np.max(a)
    e = np.exp(a)
    return e / (e.sum() + 1e-12)

def choose_legal_move_from_probs(board: chess.Board, probs: np.ndarray, argmax: bool = ARGMAX, top_k: int = TOP_K, temp: float = TEMPERATURE) -> Optional[str]:
    legal_moves = [m.uci() for m in board.legal_moves]
    if not legal_moves:
        return None

    items = []
    for mv in legal_moves:
        idx = MOVE2IDX.get(mv)
        if idx is None or idx < 0 or idx >= probs.size:
            continue
        items.append((mv, float(probs[idx]), idx))
    if not items:
        return None

    items.sort(key=lambda x: -x[1])  # descending by score

    ucis = [it[0] for it in items]
    scores = np.array([it[1] for it in items], dtype=np.float64)
    if argmax:
        return ucis[0]

    k = min(top_k, len(scores))
    top_scores = scores[:k]
    # interpret as logits if needed
    if (top_scores < 0).any() or top_scores.sum() <= 1e-8:
        logits = top_scores - np.max(top_scores)
        p = np.exp(logits / (temp if temp > 0 else 1))
        p = p / (p.sum() + 1e-12)
    else:
        p = softmax_with_temp(top_scores, temp)
    try:
        idx_choice = np.random.choice(range(k), p=p)
    except Exception:
        idx_choice = 0
    return ucis[idx_choice]

# ------------------------------
# Statistical imitation fallback
# ------------------------------
def get_board_phase(board: chess.Board) -> str:
    """Simple heuristic: opening/midgame/endgame based on number of pieces."""
    pieces = len(board.piece_map())
    if pieces >= 26:
        return "opening"
    elif pieces >= 12:
        return "midgame"
    else:
        return "endgame"

def sample_weighted_imitation(board: chess.Board) -> Tuple[str, Dict[str, Any]]:
    """
    Sample a move based on move frequency in player's corpus, with some
    board-phase bias and a tiny random exploration weight to avoid total determinism.
    Returns (chosen_move_uci, thought_info)
    """
    legal = [m.uci() for m in board.legal_moves]
    if not legal:
        raise RuntimeError("No legal moves")

    phase = get_board_phase(board)
    phase_bias = {"opening": 1.5, "midgame": 1.0, "endgame": 0.8}.get(phase, 1.0)

    weights = []
    for mv in legal:
        freq = MOVE_FREQ.get(mv, 0)
        # base weight: freq+1 so unseen moves aren't zero
        base = (freq + 1) ** phase_bias
        # small random jitter to break ties
        weight = base + random.random() * 0.01
        weights.append(weight)

    total = sum(weights)
    probs = [w / total for w in weights]

    chosen = random.choices(legal, weights=probs, k=1)[0]
    thought = {
        "method": "weighted-imitation",
        "phase": phase,
        "chosen": chosen,
        "chosen_freq": MOVE_FREQ.get(chosen, 0),
        "legal_known": sum(1 for mv in legal if MOVE_FREQ.get(mv, 0) > 0),
        "probs_sampled_len": len(probs),
    }
    return chosen, thought

# ------------------------------
# Per-game handler
# ------------------------------
def handle_game(game_id: str, my_color: bool):
    """
    This function streams the game state via berserk and reacts to moves.
    my_color: chess.WHITE (True) if the bot is white else chess.BLACK (False)
    """
    try:
        local_session = berserk.TokenSession(LICHESS_TOKEN)
        local_client = berserk.Client(session=local_session)
    except Exception as e:
        log("[handle_game] Failed to create berserk client:", e)
        local_client = None

    log(f"[{game_id}] handler start; my_color={'WHITE' if my_color else 'BLACK'}")

    board = chess.Board()
    # initialize game meta
    with GAMES_LOCK:
        GAMES[game_id] = {
            "moves": [],
            "white": None,
            "black": None,
            "last_thought": None,
            "result": None,
            "started_at": time.time(),
            "updated_at": time.time(),
        }

    # stream the game state
    try:
        # stream_game_state yields gameFull and then updates
        stream = local_client.bots.stream_game_state(game_id) if local_client else []
        for event in stream:
            try:
                etype = event.get("type")
                if etype not in ("gameFull", "gameState"):
                    continue

                state = event.get("state", event)
                moves_str = state.get("moves", "")
                moves = moves_str.split() if moves_str else []
                # rebuild board
                board = chess.Board()
                for mv in moves:
                    try:
                        board.push_uci(mv)
                    except Exception:
                        # sometimes variants or odd SAN slip; we ignore bad moves
                        log(f"[{game_id}] failed to push historic move {mv}")

                # update metadata
                white_meta = state.get("white")
                black_meta = state.get("black")
                white_id = None
                black_id = None
                if isinstance(white_meta, dict):
                    white_id = white_meta.get("id") or white_meta.get("name")
                if isinstance(black_meta, dict):
                    black_id = black_meta.get("id") or black_meta.get("name")

                with GAMES_LOCK:
                    g = GAMES.get(game_id)
                    if g:
                        g["moves"] = moves
                        g["white"] = white_id
                        g["black"] = black_id
                        g["updated_at"] = time.time()

                # terminal state
                status = state.get("status")
                if status in ("mate", "resign", "timeout", "stalemate", "draw"):
                    winner = state.get("winner")
                    result = "1-0" if winner == "white" else ("0-1" if winner == "black" else "1/2-1/2")
                    with GAMES_LOCK:
                        GAMES[game_id]["result"] = result
                    # save pgn
                    try:
                        save_game_pgn(game_id, white_id, black_id, moves, result)
                    except Exception:
                        pass
                    log(f"[{game_id}] finished: {result}")
                    break

                # check turn: board.turn True means white to move
                is_my_turn = (board.turn == my_color)

                if is_my_turn and not board.is_game_over():
                    log(f"[{game_id}] it's our turn. moves so far: {len(moves)}")
                    # debug forced behaviors
                    if DEBUG_FORCE_RANDOM:
                        chosen = random.choice([m.uci() for m in board.legal_moves])
                        thought = {"method": "debug-random"}
                        log(f"[{game_id}] DEBUG_FORCE_RANDOM picked {chosen}")
                    elif DEBUG_FORCE_OPENING:
                        opening_list = DEBUG_FORCE_OPENING.strip().split()
                        idx = len(moves)
                        if idx < len(opening_list):
                            cand = opening_list[idx]
                            if cand in [m.uci() for m in board.legal_moves]:
                                chosen = cand
                                thought = {"method": "debug-opening", "chosen": chosen}
                                log(f"[{game_id}] DEBUG_FORCE_OPENING playing {chosen}")
                            else:
                                # fall through to normal selection
                                chosen = None
                                thought = None
                        else:
                            chosen = None
                            thought = None
                    else:
                        chosen = None
                        thought = None

                    # If we haven't chosen yet, prefer model inference if available
                    if chosen is None:
                        if MODEL is not None and VOCAB_SIZE > 0:
                            try:
                                probs = infer_move_probs(board)
                                if DEBUG_LOG_PROBS:
                                    topk = np.argsort(probs)[::-1][:16]
                                    tops = [(int(i), IDX2MOVE[i] if i < len(IDX2MOVE) else None, float(probs[i])) for i in topk]
                                    log(f"[{game_id}] model top predictions: {tops}")
                                chosen = choose_legal_move_from_probs(board, probs, argmax=ARGMAX, top_k=TOP_K, temp=TEMPERATURE)
                                thought = {"method": "model", "argmax": ARGMAX, "top_k": TOP_K}
                                if chosen is None:
                                    log(f"[{game_id}] model produced no legal-known moves; falling back to weighted imitation")
                                    # fall back
                                    chosen, thought = sample_weighted_imitation(board)
                            except Exception as e:
                                log(f"[{game_id}] model inference exception: {e}")
                                traceback.print_exc()
                                # fallback
                                chosen, thought = sample_weighted_imitation(board)
                        else:
                            # no model — use weighted imitation
                            chosen, thought = sample_weighted_imitation(board)

                    # final safety: if chosen still None, pick a legal move
                    if not chosen:
                        legal = [m.uci() for m in board.legal_moves]
                        chosen = random.choice(legal)
                        thought = thought or {"method": "fallback-random"}

                    # update thought + store + send move
                    with GAMES_LOCK:
                        GAMES[game_id]["last_thought"] = thought
                        GAMES[game_id]["moves"] = moves + [chosen]
                        GAMES[game_id]["updated_at"] = time.time()

                    # send move with retry/backoff
                    try:
                        retry_call(local_client.bots.make_move, game_id, chosen)
                        log(f"[{game_id}] played {chosen} (method={thought.get('method') if isinstance(thought, dict) else thought})")
                    except Exception:
                        log(f"[{game_id}] failed to send chosen move {chosen}; will try direct fallback")
                        try:
                            fallback = random.choice([m.uci() for m in board.legal_moves])
                            retry_call(local_client.bots.make_move, game_id, fallback)
                            log(f"[{game_id}] played fallback {fallback}")
                        except Exception:
                            log(f"[{game_id}] fallback also failed")

                    # pacing delay
                    time.sleep(MOVE_DELAY)

                else:
                    # not our turn; continue waiting
                    pass

            except Exception:
                log(f"[{game_id}] exception while processing event")
                traceback.print_exc()

    except Exception:
        log(f"[{game_id}] game stream failed")
        traceback.print_exc()
    finally:
        log(f"[{game_id}] handler exiting")

# ------------------------------
# Utility: save game as PGN
# ------------------------------
def save_game_pgn(game_id: str, white: Optional[str], black: Optional[str], moves: List[str], result: Optional[str] = None) -> str:
    try:
        game = chess.pgn.Game()
        game.headers["Event"] = "ImitationGame"
        game.headers["Site"] = "Lichess"
        game.headers["White"] = white or "white"
        game.headers["Black"] = black or "black"
        game.headers["Result"] = result or "*"
        node = game
        for uci in moves:
            try:
                mv = chess.Move.from_uci(uci)
                node = node.add_variation(mv)
            except Exception:
                # skip unparseable move
                pass
        path = os.path.join(STORAGE_DIR, f"{game_id}.pgn")
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(game))
        log(f"[pgn] saved {path}")
        return path
    except Exception:
        log("[pgn] save failed")
        traceback.print_exc()
        raise

# ------------------------------
# Incoming events loop (main)
# ------------------------------
def main_event_loop():
    """
    Streams incoming events (challenges & gameStart). Accepts standard/fromPosition challenges.
    """
    client = create_global_client()
    log("[events] starting incoming events stream")
    for event in client.bots.stream_incoming_events():
        try:
            etype = event.get("type")
            log("[events] incoming event type:", etype)
            if etype == "challenge":
                chal = event.get("challenge", {})
                cid = chal.get("id")
                variant = chal.get("variant", {}).get("key")
                log(f"[events] challenge id={cid} variant={variant}")
                if variant in ("standard", "fromPosition"):
                    try:
                        retry_call(client.bots.accept_challenge, cid)
                        log(f"[events] accepted challenge {cid}")
                    except Exception:
                        log(f"[events] accept failed for {cid}")
                        traceback.print_exc()
                else:
                    try:
                        retry_call(client.bots.decline_challenge, cid)
                        log(f"[events] declined challenge {cid}")
                    except Exception:
                        log(f"[events] decline failed for {cid}")
                        traceback.print_exc()

            elif etype == "gameStart":
                gid = event.get("game", {}).get("id")
                color_str = event.get("game", {}).get("color")
                my_color = chess.WHITE if color_str == "white" else chess.BLACK
                log(f"[events] gameStart id={gid} color={color_str}")
                th = threading.Thread(target=handle_game, args=(gid, my_color), daemon=True)
                with GAME_THREADS_LOCK:
                    GAME_THREADS[gid] = th
                th.start()

        except Exception:
            log("[events] exception in main event loop")
            traceback.print_exc()

# ------------------------------
# Supervisor: restart event thread if it dies
# ------------------------------
def start_event_thread():
    global _event_thread
    with _event_thread_lock:
        if _event_thread and _event_thread.is_alive():
            log("[supervisor] event thread already running")
            return
        _stop_flag.clear()
        def worker():
            backoff = BACKOFF_BASE
            while not _stop_flag.is_set():
                try:
                    create_global_client()
                    main_event_loop()
                    log("[supervisor] event stream ended; reconnecting after", backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, BACKOFF_MAX)
                except Exception:
                    log("[supervisor] event thread exception; sleeping", backoff)
                    traceback.print_exc()
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, BACKOFF_MAX)
        _event_thread = threading.Thread(target=worker, daemon=True)
        _event_thread.start()
        log("[supervisor] event thread started")

def stop_event_thread():
    _stop_flag.set()
    with _event_thread_lock:
        if _event_thread and _event_thread.is_alive():
            _event_thread.join(timeout=2.0)
            log("[supervisor] event thread stopped")

# ------------------------------
# Heartbeat monitor
# ------------------------------
def start_heartbeat_monitor():
    def hb():
        while True:
            try:
                alive = False
                with _event_thread_lock:
                    if _event_thread and _event_thread.is_alive():
                        alive = True
                if not alive:
                    log("[heartbeat] event thread not alive, restarting")
                    start_event_thread()
                time.sleep(max(5, HEARTBEAT_TIMEOUT // 6))
            except Exception:
                traceback.print_exc()
                time.sleep(5)
    t = threading.Thread(target=hb, daemon=True)
    t.start()

# ------------------------------
# Flask admin & dashboard
# ------------------------------
app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <title>Imitation Bot Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; background: #0b0b0b; color: #e6f0ff; }
    .game { border: 1px solid #37474f; padding: 10px; margin: 8px; border-radius: 6px; background: #071018; }
    h1 { color: #8ee0a9; }
    pre { white-space: pre-wrap; }
  </style>
  <meta http-equiv="refresh" content="5">
</head>
<body>
  <h1>Imitation Bot Dashboard</h1>
  <p>Model loaded: {{ model_loaded }}, vocab size: {{ vocab_size }}, active games: {{ active_games }}</p>
  {% for gid, g in games.items() %}
  <div class="game">
    <b>Game:</b> {{ gid }} <br/>
    <b>White:</b> {{ g.white }}  <b>Black:</b> {{ g.black }} <br/>
    <b>Moves ({{ g.moves|length }}):</b> <pre>{{ " ".join(g.moves) }}</pre>
    <b>Last thought:</b> <pre>{{ g.last_thought }}</pre>
    <b>Result:</b> {{ g.result }} <br/>
    <form method="post" action="/admin/save/{{ gid }}">
      <button type="submit">Save PGN</button>
    </form>
  </div>
  {% endfor %}
</body>
</html>
"""

@app.route("/")
def index():
    with GAMES_LOCK:
        summary = {gid: {
            "moves": g.get("moves", []),
            "last_thought": g.get("last_thought"),
            "white": g.get("white"),
            "black": g.get("black"),
            "result": g.get("result")
        } for gid, g in GAMES.items()}
        active = len(GAMES)
    return render_template_string(INDEX_HTML, games=summary, model_loaded=(MODEL is not None), vocab_size=VOCAB_SIZE, active_games=active)

@app.route("/debug")
def debug_json():
    with GAMES_LOCK:
        return jsonify(GAMES)

@app.route("/admin/reload", methods=["POST"])
def admin_reload():
    try:
        load_vocab_npz(VOCAB_PATH)
        load_keras_model(MODEL_PATH)
        return jsonify({"status": "reloaded", "vocab_size": VOCAB_SIZE, "model_loaded": MODEL is not None})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/admin/stats")
def admin_stats():
    with GAMES_LOCK:
        active = len(GAMES)
    return jsonify({"model_loaded": MODEL is not None, "vocab_size": VOCAB_SIZE, "active_games": active, "move_freq_sample": MOVE_FREQ.most_common(10)})

@app.route("/admin/save/<game_id>", methods=["POST"])
def admin_save(game_id):
    with GAMES_LOCK:
        g = GAMES.get(game_id)
        if not g:
            return jsonify({"error": "game not found"}), 404
        white = g.get("white")
        black = g.get("black")
        moves = g.get("moves", [])
        result = g.get("result")
    path = save_game_pgn(game_id, white, black, moves, result)
    return jsonify({"saved": path})

@app.route("/pgn/<game_id>")
def get_pgn(game_id):
    path = os.path.join(STORAGE_DIR, f"{game_id}.pgn")
    if not os.path.exists(path):
        return jsonify({"error": "pgn not found"}), 404
    return send_file(path, as_attachment=True, download_name=f"{game_id}.pgn")

# ------------------------------
# CLI helper: load a model/vocab from paths and report diagnostics
# ------------------------------
def diagnostic_model_check():
    log("[diag] running quick model diagnostics")
    try:
        if VOCAB_SIZE == 0:
            log("[diag] vocab not loaded or empty")
        else:
            log(f"[diag] vocab size: {VOCAB_SIZE}. sample: {IDX2MOVE[:20]}")
    except Exception:
        traceback.print_exc()

    if MODEL is None:
        log("[diag] model is not loaded")
        return

    try:
        # test on starting position
        board = chess.Board()
        encs = []
        try:
            seq = encode_sequence(board)
            encs.append(("sequence", seq))
        except Exception:
            pass
        try:
            planes = encode_planes(board)
            encs.append(("planes", planes))
        except Exception:
            pass
        try:
            onehot = encode_onehot_vec(board)
            encs.append(("onehot", onehot))
        except Exception:
            pass

        for name, inp in encs:
            try:
                out = MODEL.predict(inp, verbose=0)
                arr = np.array(out).flatten()
                log(f"[diag] encoder {name} -> model output length = {arr.size}")
                top = np.argsort(arr)[::-1][:20]
                top_moves = [(int(i), IDX2MOVE[i] if i < len(IDX2MOVE) else None, float(arr[i])) for i in top]
                log("[diag] top moves:", top_moves)
                known = [IDX2MOVE[i] for i in top if i < len(IDX2MOVE) and IDX2MOVE[i] in [m.uci() for m in board.legal_moves]]
                log("[diag] known legal in top20:", known)
            except Exception:
                log(f"[diag] model predict failed for encoder {name}")
                traceback.print_exc()

    except Exception:
        log("[diag] model diagnostics failed")
        traceback.print_exc()

# ------------------------------
# Boot and run
# ------------------------------
def boot():
    # ensure client
    try:
        create_global_client()
    except Exception:
        log("[boot] global client creation failed (retry will happen in thread)")

    # diagnostic info
    log("Booting imitation bot")
    log("MODEL path:", MODEL_PATH, "MODEL loaded:", MODEL is not None, "VOCAB path:", VOCAB_PATH, "VOCAB size:", VOCAB_SIZE)
    diagnostic_model_check()
    start_event_thread()
    start_heartbeat_monitor()

if __name__ == "__main__":
    boot()
    log("Starting Flask admin on port", PORT)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
