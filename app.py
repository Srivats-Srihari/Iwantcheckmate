#!/usr/bin/env python3
"""
app.py - Player-imitation Lichess bot (full deployment-ready file)

Features:
 - Uses per-game berserk client and threaded handler (robust against stream stalls)
 - Loads Keras model (model.h5) and vocab (vocab.npz) with flexible key parsing
 - Tries multiple encoders (sequence tokens, board planes, one-hot move-vector)
 - Restricts model output to legal moves; samples or argmax as configured
 - Safe fallbacks (random legal move) on inference failure
 - Debug flag to force random moves for testing
 - Admin endpoints and reload ability
 - Rate-limit-aware retry/backoff wrapper
 - Detailed logging for diagnostics
"""

from __future__ import annotations

import os
import sys
import time
import math
import json
import random
import logging
import threading
import traceback
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import berserk
import chess
import chess.pgn
from flask import Flask, jsonify, request, send_file

# Try TensorFlow import; if missing, model functionality still available but disabled.
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception:
    tf = None
    load_model = None

# ----------------------
# Configuration (env)
# ----------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LICHESS_TOKEN_ENV = os.environ.get("LICHESS_TOKEN_ENV", "Lichess_token")
LICHESS_TOKEN = os.environ.get(LICHESS_TOKEN_ENV) or os.environ.get("LICHESS_TOKEN")

MODEL_PATH = os.environ.get("MODEL_PATH", "model.h5")
VOCAB_PATH = os.environ.get("VOCAB_PATH", "vocab.npz")
MODEL_URL = os.environ.get("MODEL_URL", None)
VOCAB_URL = os.environ.get("VOCAB_URL", None)

# Inference parameters
ARGMAX = os.environ.get("ARGMAX", "false").lower() in ("1", "true", "yes")
TOP_K = int(os.environ.get("TOP_K", "32"))
TEMPERATURE = float(os.environ.get("TEMP", "1.0"))

# Debugging and test flags
DEBUG_FORCE_RANDOM = os.environ.get("DEBUG_FORCE_RANDOM", "false").lower() in ("1", "true", "yes")
DEBUG_FORCE_OPENING = os.environ.get("DEBUG_FORCE_OPENING", "")  # e.g. "e2e4 e7e5"

# Network + retry/backoff
BACKOFF_BASE = float(os.environ.get("BACKOFF_BASE", "1.0"))
BACKOFF_MAX = float(os.environ.get("BACKOFF_MAX", "60.0"))
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "6"))

# Misc
MOVE_DELAY = float(os.environ.get("MOVE_DELAY", "0.30"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "128"))
STORAGE_DIR = os.environ.get("STORAGE_DIR", "saved_games")
PORT = int(os.environ.get("PORT", "10000"))
HEARTBEAT_TIMEOUT = int(os.environ.get("HEARTBEAT_TIMEOUT", "180"))

# Ensure dirs
os.makedirs(STORAGE_DIR, exist_ok=True)

# ----------------------
# Logging
# ----------------------
logger = logging.getLogger("imit-bot")
logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(handler)

# ----------------------
# Globals: Model & Vocab
# ----------------------
MODEL = None
MODEL_INPUT_SHAPE = None
IDX2MOVE: List[str] = []
MOVE2IDX: Dict[str, int] = {}
VOCAB_SIZE = 0

# ----------------------
# Flask app + Stores
# ----------------------
app = Flask(__name__, static_folder=None)
GAMES: Dict[str, Dict[str, Any]] = {}
GAMES_LOCK = threading.Lock()

# ----------------------
# Helper: optional download
# ----------------------
def download_if_needed(url: Optional[str], dest: str):
    if not url:
        return
    if os.path.exists(dest):
        logger.info("File exists, skipping download: %s", dest)
        return
    logger.info("Downloading %s -> %s", url, dest)
    import requests
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    tmp = dest + ".tmp"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)
    os.replace(tmp, dest)
    logger.info("Finished download: %s", dest)

# ----------------------
# Load Vocab (robust)
# ----------------------
def load_vocab_npz(path: str):
    global IDX2MOVE, MOVE2IDX, VOCAB_SIZE
    if not os.path.exists(path):
        raise FileNotFoundError(f"vocab not found: {path}")
    data = np.load(path, allow_pickle=True)
    keys = list(data.files)
    logger.info("Loading vocab.npz keys: %s", keys)

    moves = None
    move2idx = None
    idx2move = None

    if "moves" in keys:
        moves = data["moves"]
    if "move2idx" in keys:
        move2idx = data["move2idx"]
    if "idx2move" in keys:
        idx2move = data["idx2move"]

    # fallback names
    if moves is None and "vocab" in keys:
        moves = data["vocab"]
    if moves is None and len(keys) == 1:
        moves = data[keys[0]]

    # Build IDX2MOVE / MOVE2IDX
    if idx2move is not None:
        try:
            IDX2MOVE = [str(x) for x in idx2move.tolist()]
        except Exception:
            IDX2MOVE = [str(x) for x in idx2move]
    elif moves is not None:
        try:
            IDX2MOVE = [str(x) for x in moves.tolist()]
        except Exception:
            IDX2MOVE = [str(x) for x in moves]
    elif move2idx is not None:
        try:
            m2i = dict(move2idx.tolist())
        except Exception:
            m2i = dict(move2idx)
        maxidx = max(m2i.values())
        IDX2MOVE = [None] * (maxidx + 1)
        for m, i in m2i.items():
            IDX2MOVE[i] = str(m)
    else:
        # heuristic: pick a key with array-like of moves
        for k in keys:
            arr = data[k]
            try:
                cand = [str(x) for x in arr]
                # heuristics: moves are short strings like 'e2e4' or 'Nf3'
                if all(1 <= len(s) <= 6 for s in cand):
                    IDX2MOVE = cand
                    logger.warning("Fallback: using key '%s' as move list", k)
                    break
            except Exception:
                continue

    if not IDX2MOVE:
        raise ValueError("Could not extract moves list from vocab.npz; keys: %s" % keys)

    MOVE2IDX = {m: i for i, m in enumerate(IDX2MOVE)}
    VOCAB_SIZE = len(IDX2MOVE)
    logger.info("Vocab loaded: %d moves", VOCAB_SIZE)

# ----------------------
# Load Keras model (if TF available)
# ----------------------
def load_keras_model(path: str):
    global MODEL, MODEL_INPUT_SHAPE
    if load_model is None:
        raise RuntimeError("TensorFlow/Keras not available in this environment.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    logger.info("Loading Keras model from %s", path)
    MODEL = load_model(path)
    try:
        inp = MODEL.inputs[0].shape
        if hasattr(inp, "as_list"):
            dims = inp.as_list()
        else:
            dims = list(inp)
        MODEL_INPUT_SHAPE = tuple(dims)
        logger.info("Model input shape: %s", MODEL_INPUT_SHAPE)
    except Exception:
        logger.warning("Unable to infer model input shape; encoders will try safe defaults.")

# ----------------------
# Public reload function
# ----------------------
def reload_model_and_vocab():
    download_if_needed(MODEL_URL, MODEL_PATH)
    download_if_needed(VOCAB_URL, VOCAB_PATH)
    load_vocab_npz(VOCAB_PATH)
    try:
        load_keras_model(MODEL_PATH)
    except Exception as e:
        logger.exception("Failed to load Keras model: %s", e)
        # keep MODEL as None if load fails

# Try initial load but do not crash
try:
    reload_model_and_vocab()
except Exception:
    logger.exception("Initial model/vocab load failed (use /admin/reload after fixing).")

# ----------------------
# Encoders (multiple strategies)
# ----------------------

# 1) Sequence-token encoder (last SEQ_LEN moves -> integer tokens)
def encode_sequence(board: chess.Board, seq_len: int = SEQ_LEN) -> np.ndarray:
    moves = [m.uci() for m in board.move_stack]
    toks = []
    for mv in moves[-seq_len:]:
        idx = MOVE2IDX.get(mv)
        # we reserve 0 for padding; store idx+1 for real moves so 0 used for pad
        toks.append((idx + 1) if idx is not None else 0)
    if len(toks) < seq_len:
        toks = [0] * (seq_len - len(toks)) + toks
    return np.array(toks, dtype=np.int32).reshape(1, seq_len)

# 2) Board plane encoder (8x8x12 -> 768)
PIECE_TO_PLANE = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
}
def encode_planes(board: chess.Board) -> np.ndarray:
    arr = np.zeros((8, 8, 12), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        r = chess.square_rank(sq)
        c = chess.square_file(sq)
        plane = PIECE_TO_PLANE.get(piece.symbol())
        if plane is not None:
            arr[r, c, plane] = 1.0
    return arr.reshape(1, -1).astype(np.float32)

# 3) One-hot move-vector summarizer
def encode_onehot_vec(board: chess.Board) -> np.ndarray:
    vec = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
    for mv in [m.uci() for m in board.move_stack]:
        idx = MOVE2IDX.get(mv)
        if idx is not None:
            vec[0, idx] += 1.0
    s = vec.sum()
    if s > 0:
        vec /= s
    return vec

# ----------------------
# Model inference: try encoders until success
# ----------------------
def normalize_probs(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Empty model output")
    if (arr >= 0).all() and abs(arr.sum() - 1.0) < 1e-3:
        return arr
    ex = np.exp(arr - np.max(arr))
    return ex / (ex.sum() + 1e-12)

def infer_move_probs(board: chess.Board) -> np.ndarray:
    """
    Attempt to run the model with different encoders in order:
      1) sequence tokens
      2) board planes
      3) one-hot move vector
    Returns probs length VOCAB_SIZE (sums to 1).
    Raises on total failure.
    """
    if MODEL is None:
        raise RuntimeError("Model is not loaded")

    errors = []
    # 1: sequence tokens
    try:
        seq = encode_sequence(board)
        out = MODEL.predict(seq, verbose=0)
        probs = np.array(out[0], dtype=np.float64)
        return normalize_probs(probs)
    except Exception as e:
        errors.append(("seq", str(e)))

    # 2: board planes
    try:
        planes = encode_planes(board)
        # if model expects 768-dim and our planes are 768, this should work.
        out = MODEL.predict(planes, verbose=0)
        probs = np.array(out[0], dtype=np.float64)
        return normalize_probs(probs)
    except Exception as e:
        errors.append(("planes", str(e)))

    # 3: one-hot vector
    try:
        v = encode_onehot_vec(board)
        out = MODEL.predict(v, verbose=0)
        probs = np.array(out[0], dtype=np.float64)
        return normalize_probs(probs)
    except Exception as e:
        errors.append(("onehot", str(e)))

    logger.warning("Model inference failed for all encoders: %s", errors)
    raise RuntimeError("Model inference failed for all encoders; see logs for details")

# ----------------------
# Choose legal move from model probs
# ----------------------
def softmax_with_temp(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    if temp <= 0:
        temp = 1e-6
    a = x / temp
    a = a - np.max(a)
    e = np.exp(a)
    return e / (e.sum() + 1e-12)

def choose_legal_move(board: chess.Board, probs: np.ndarray,
                      argmax: bool = ARGMAX, top_k: int = TOP_K, temp: float = TEMPERATURE) -> Optional[str]:
    legal_moves = [m.uci() for m in board.legal_moves]
    if not legal_moves:
        return None

    # Map legal moves to model indices & scores
    items: List[Tuple[str, float, int]] = []
    for mv in legal_moves:
        idx = MOVE2IDX.get(mv)
        if idx is None or idx < 0 or idx >= len(probs):
            continue
        items.append((mv, float(probs[idx]), idx))
    if not items:
        # model doesn't know legal moves, fallback to random legal
        return random.choice(legal_moves)

    items.sort(key=lambda x: -x[1])  # descending by score
    ucis = [it[0] for it in items]
    scores = np.array([it[1] for it in items], dtype=np.float64)

    if argmax:
        return ucis[0]

    k = min(top_k, len(scores))
    top_scores = scores[:k]
    # treat as logits if they are not normalized
    if (top_scores < 0).any() or top_scores.sum() <= 1e-8:
        logits = top_scores - np.max(top_scores)
        p = np.exp(logits / (temp if temp > 0 else 1e-6))
        p = p / (p.sum() + 1e-12)
    else:
        p = softmax_with_temp(top_scores, temp)
    idx_choice = np.random.choice(range(k), p=p)
    return ucis[idx_choice]

# ----------------------
# Berserk wrapper and per-game client
# ----------------------
_global_client: Optional[berserk.Client] = None
_global_client_lock = threading.Lock()

def create_global_client():
    global _global_client
    with _global_client_lock:
        if _global_client is None:
            if not LICHESS_TOKEN:
                raise RuntimeError(f"Lichess token not set in env var {LICHESS_TOKEN_ENV}")
            session = berserk.TokenSession(LICHESS_TOKEN)
            _global_client = berserk.Client(session=session)
            logger.info("Created global berserk client")
    return _global_client

def retry_call(func, *args, max_attempts: int = MAX_ATTEMPTS, base_wait: float = BACKOFF_BASE, **kwargs):
    wait = base_wait
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            logger.warning("Call failed (attempt %d/%d): %s", attempt, max_attempts, msg)
            # If rate-limited, backoff aggressively
            if "429" in msg or "Too Many Requests" in msg:
                logger.warning("Rate-limited: sleeping %.1fs", wait)
                time.sleep(wait)
                wait = min(wait * 2.0, BACKOFF_MAX)
                continue
            # For other transient network errors, backoff mildly and retry
            time.sleep(min(wait, BACKOFF_MAX))
            wait = min(wait * 2.0, BACKOFF_MAX)
    raise RuntimeError("Max retry attempts exceeded for call")

# ----------------------
# Save PGN utility
# ----------------------
def save_game_pgn(game_id: str, white: str, black: str, moves: List[str], result: Optional[str] = None) -> str:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    path = os.path.join(STORAGE_DIR, f"{game_id}.pgn")
    game = chess.pgn.Game()
    game.headers["Event"] = "ImitationGame"
    game.headers["White"] = white or "white"
    game.headers["Black"] = black or "black"
    game.headers["Result"] = result or "*"
    node = game
    for uci in moves:
        try:
            move = chess.Move.from_uci(uci)
            node = node.add_variation(move)
        except Exception:
            pass
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(game))
    logger.info("Saved PGN: %s", path)
    return path

# ----------------------
# Core: per-game handler (robust)
# ----------------------
def handle_game(game_id: str, my_color: bool):
    """
    game_id: lichess game id
    my_color: chess.WHITE (True) if we're white, False if black
    """
    logger.info("[%s] handler starting (color=%s). model_loaded=%s vocab=%d",
                game_id, "WHITE" if my_color else "BLACK", MODEL is not None, VOCAB_SIZE)

    # per-thread bersek client to avoid cross-thread session issues
    local_session = berserk.TokenSession(LICHESS_TOKEN)
    local_client = berserk.Client(session=local_session)

    board = chess.Board()
    start_ts = time.time()

    # Initialize store entry
    with GAMES_LOCK:
        GAMES[game_id] = {
            "moves": [],
            "white": None,
            "black": None,
            "last_thought": None,
            "result": None,
            "started_at": start_ts,
            "updated_at": start_ts
        }

    try:
        # stream_game_state yields gameFull once and then gameState updates
        for event in local_client.bots.stream_game_state(game_id):
            try:
                etype = event.get("type")
                if etype not in ("gameFull", "gameState"):
                    # ignore other events in this stream
                    continue

                state = event.get("state", event)
                moves_str = state.get("moves", "")
                moves = moves_str.split() if moves_str else []

                # rebuild board from scratch (robust)
                board = chess.Board()
                for mv in moves:
                    try:
                        board.push_uci(mv)
                    except Exception:
                        logger.debug("[%s] failed to push historic move %s", game_id, mv)

                # store basic metadata (white/black ids)
                white_id = None
                black_id = None
                if isinstance(state.get("white"), dict):
                    white_id = state["white"].get("id") or state["white"].get("name")
                if isinstance(state.get("black"), dict):
                    black_id = state["black"].get("id") or state["black"].get("name")

                with GAMES_LOCK:
                    g = GAMES.get(game_id)
                    if g:
                        g["moves"] = moves
                        g["white"] = white_id
                        g["black"] = black_id
                        g["updated_at"] = time.time()

                # terminal check
                status = state.get("status")
                if status in ("mate", "resign", "timeout", "draw", "stalemate", "outoftime"):
                    winner = state.get("winner")
                    result = "1-0" if winner == "white" else ("0-1" if winner == "black" else "1/2-1/2")
                    with GAMES_LOCK:
                        GAMES[game_id]["result"] = result
                    save_game_pgn(game_id, white_id, black_id, moves, result)
                    logger.info("[%s] finished: %s", game_id, result)
                    break

                # Reliable turn check: use my_color and the reconstructed board
                is_my_turn = (board.turn == my_color)

                if is_my_turn and not board.is_game_over():
                    logger.info("[%s] It's our turn. Moves so far: %s", game_id, " ".join(moves))
                    # If debug forcing random moves, do so
                    if DEBUG_FORCE_RANDOM:
                        chosen_uci = random.choice([m.uci() for m in board.legal_moves])
                        reason = "debug-random"
                        thought = {"chosen": chosen_uci, "reason": reason}
                        with GAMES_LOCK:
                            GAMES[game_id]["last_thought"] = thought
                        try:
                            retry_call(local_client.bots.make_move, game_id, chosen_uci)
                            logger.info("[%s] (debug) played %s", game_id, chosen_uci)
                        except Exception:
                            logger.exception("[%s] (debug) failed to play %s", game_id, chosen_uci)
                        time.sleep(MOVE_DELAY)
                        continue

                    # If we have a debug opening override string, play successive opening moves in order
                    if DEBUG_FORCE_OPENING:
                        opening_moves = DEBUG_FORCE_OPENING.split()
                        # only play if next opening move matches a legal move
                        next_idx = len(moves)
                        if next_idx < len(opening_moves):
                            candidate = opening_moves[next_idx]
                            if candidate in [m.uci() for m in board.legal_moves]:
                                chosen_uci = candidate
                                reason = "debug-opening"
                                thought = {"chosen": chosen_uci, "reason": reason}
                                with GAMES_LOCK:
                                    GAMES[game_id]["last_thought"] = thought
                                try:
                                    retry_call(local_client.bots.make_move, game_id, chosen_uci)
                                    logger.info("[%s] (debug-opening) played %s", game_id, chosen_uci)
                                except Exception:
                                    logger.exception("[%s] (debug-opening) failed to play %s", game_id, chosen_uci)
                                time.sleep(MOVE_DELAY)
                                continue
                            # else fall through to regular model logic

                    # Attempt model inference (protected)
                    chosen_uci = None
                    reason = None
                    try:
                        if MODEL is None or VOCAB_SIZE == 0:
                            raise RuntimeError("Model or vocab not loaded")
                        probs = infer_move_probs(board)
                        chosen_uci = choose_legal_move(board, probs, argmax=ARGMAX, top_k=TOP_K, temp=TEMPERATURE)
                        reason = "model-argmax" if ARGMAX else "model-sampled"
                    except Exception as e:
                        # log full stacktrace for debugging
                        logger.exception("[%s] Model inference failed: %s", game_id, e)
                        chosen_uci = None
                        reason = "model-failed"

                    if not chosen_uci:
                        # fallback to random legal move
                        chosen_uci = random.choice([m.uci() for m in board.legal_moves])
                        if reason is None:
                            reason = "fallback-random"
                        else:
                            reason = reason + "|fallback-random"

                    # build thought for dashboard
                    thought = {"chosen": chosen_uci, "reason": reason}
                    try:
                        # include simple top-k slices (best-effort; if probs unavailable skip)
                        if 'probs' in locals() and probs is not None:
                            topk = min(8, len(probs))
                            idxs = np.argsort(probs)[::-1][:topk]
                            thought["top"] = [(IDX2MOVE[i] if i < len(IDX2MOVE) else None, float(probs[i])) for i in idxs]
                    except Exception:
                        pass

                    with GAMES_LOCK:
                        GAMES[game_id]["last_thought"] = thought

                    # send the move (with retry/backoff)
                    try:
                        retry_call(local_client.bots.make_move, game_id, chosen_uci)
                        logger.info("[%s] Played %s (%s)", game_id, chosen_uci, reason)
                    except Exception:
                        logger.exception("[%s] Failed to send move %s; trying random fallback", game_id, chosen_uci)
                        fallback = random.choice([m.uci() for m in board.legal_moves])
                        try:
                            retry_call(local_client.bots.make_move, game_id, fallback)
                            logger.info("[%s] Played fallback %s", game_id, fallback)
                        except Exception:
                            logger.exception("[%s] Fallback move also failed", game_id)

                    # small pacing delay
                    time.sleep(MOVE_DELAY)
                else:
                    logger.debug("[%s] Not our turn (is_my_turn=%s) or game is over", game_id, is_my_turn)

            except Exception:
                logger.exception("[%s] Exception while processing game stream event", game_id)

    except Exception:
        logger.exception("[%s] Unhandled exception in game handler stream", game_id)
    finally:
        logger.info("[%s] Handler terminating", game_id)

# ----------------------
# Main incoming events loop (robust)
# ----------------------
def main_event_loop():
    cl = create_global_client()
    logger.info("Starting incoming events stream")
    for event in cl.bots.stream_incoming_events():
        try:
            etype = event.get("type")
            logger.info("Incoming event: %s", etype)
            if etype == "challenge":
                chal = event.get("challenge", {})
                cid = chal.get("id")
                variant = chal.get("variant", {}).get("key")
                logger.info("Challenge id=%s variant=%s", cid, variant)
                if variant in ("standard", "fromPosition"):
                    try:
                        retry_call(cl.bots.accept_challenge, cid)
                        logger.info("Accepted challenge %s", cid)
                    except Exception:
                        logger.exception("Failed to accept challenge %s", cid)
                else:
                    try:
                        retry_call(cl.bots.decline_challenge, cid)
                        logger.info("Declined challenge %s", cid)
                    except Exception:
                        logger.exception("Failed to decline challenge %s", cid)
            elif etype == "gameStart":
                gid = event.get("game", {}).get("id")
                color_str = event.get("game", {}).get("color")
                my_color = chess.WHITE if color_str == "white" else chess.BLACK
                logger.info("GameStart: id=%s color=%s", gid, color_str)
                # start handler thread
                t = threading.Thread(target=handle_game, args=(gid, my_color), daemon=True)
                t.start()
        except Exception:
            logger.exception("Error processing incoming event")

# ----------------------
# Thread supervisor
# ----------------------
_event_thread: Optional[threading.Thread] = None
_event_thread_lock = threading.Lock()
_stop_flag = threading.Event()

def start_event_thread():
    global _event_thread
    with _event_thread_lock:
        if _event_thread and _event_thread.is_alive():
            logger.info("Event thread already running")
            return
        _stop_flag.clear()
        def worker():
            backoff = BACKOFF_BASE
            while not _stop_flag.is_set():
                try:
                    if _global_client is None:
                        create_global_client()
                    main_event_loop()
                    logger.warning("Incoming events stream ended; reconnecting after %.1fs", backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, BACKOFF_MAX)
                except Exception:
                    logger.exception("Event thread exception; backing off %.1fs", backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, BACKOFF_MAX)
        _event_thread = threading.Thread(target=worker, daemon=True)
        _event_thread.start()
        logger.info("Event thread started")

def stop_event_thread():
    _stop_flag.set()
    with _event_thread_lock:
        if _event_thread and _event_thread.is_alive():
            logger.info("Stopping event thread")
            _event_thread.join(timeout=2.0)

# ----------------------
# Flask endpoints (debug/admin)
# ----------------------
@app.route("/")
def index():
    with GAMES_LOCK:
        summary = {gid: {"moves": g["moves"], "last_thought": g["last_thought"], "result": g["result"], "white": g["white"], "black": g["black"]} for gid, g in GAMES.items()}
    return jsonify(summary)

@app.route("/debug")
def debug():
    with GAMES_LOCK:
        return jsonify(GAMES)

@app.route("/admin/reload", methods=["POST"])
def admin_reload():
    try:
        reload_model_and_vocab()
        return jsonify({"status": "reloaded", "vocab_size": VOCAB_SIZE, "model_loaded": MODEL is not None})
    except Exception as e:
        logger.exception("Reload failed")
        return jsonify({"error": str(e)}), 500

@app.route("/admin/stats")
def admin_stats():
    with GAMES_LOCK:
        active = len(GAMES)
    return jsonify({"model_loaded": MODEL is not None, "vocab_size": VOCAB_SIZE, "active_games": active})

@app.route("/admin/save/<game_id>", methods=["POST"])
def admin_save_game(game_id):
    with GAMES_LOCK:
        g = GAMES.get(game_id)
        if not g:
            return jsonify({"error": "game not found"}), 404
        path = save_game_pgn(game_id, g.get("white"), g.get("black"), g.get("moves", []), g.get("result"))
    return jsonify({"saved": path})

@app.route("/pgn/<game_id>")
def serve_pgn(game_id):
    path = os.path.join(STORAGE_DIR, f"{game_id}.pgn")
    if not os.path.exists(path):
        return jsonify({"error": "pgn not found"}), 404
    return send_file(path, as_attachment=True, download_name=f"{game_id}.pgn")

# ----------------------
# Boot sequence
# ----------------------
def boot():
    logger.info("Booting imitation bot")
    # attempt to load model/vocab already done at import time, but ensure global client
    try:
        create_global_client()
    except Exception:
        logger.exception("Creating global client failed at boot (will be retried in event thread)")
    start_event_thread()

# Heartbeat supervisor thread to restart event thread if it dies
def start_heartbeat():
    def hb_worker():
        while True:
            try:
                alive = False
                with _event_thread_lock:
                    if _event_thread:
                        alive = _event_thread.is_alive()
                if not alive:
                    logger.warning("Event thread not alive; restarting")
                    start_event_thread()
                time.sleep(max(5, HEARTBEAT_TIMEOUT // 6))
            except Exception:
                logger.exception("Heartbeat failed; sleeping briefly")
                time.sleep(5)
    t = threading.Thread(target=hb_worker, daemon=True)
    t.start()

# ----------------------
# Run as script
# ----------------------
if __name__ == "__main__":
    boot()
    start_heartbeat()
    logger.info("Starting Flask on port %d", PORT)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
