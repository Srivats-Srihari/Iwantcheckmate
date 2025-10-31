#!/usr/bin/env python3
"""
app.py - Lichess player-imitation bot.
- Loads model.h5 and vocab.npz (flexible keys).
- Tries sequence-token, plane, and one-hot encoders.
- Uses model outputs to pick legal moves (argmax or sampled).
- Robust per-game streaming and reconnect/backoff logic.
- Flask endpoints: / (status), /debug, /admin/reload
"""

import os
import sys
import time
import math
import json
import random
import logging
import threading
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import berserk
import chess
from flask import Flask, jsonify, request

# try tensorflow import
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception as e:
    tf = None
    load_model = None

# ---------- CONFIG ----------
LOG = logging.getLogger("imit-bot")
LOG.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
LOG.addHandler(h)

LICHESS_TOKEN_ENV = "Lichess_token"
LICHESS_TOKEN = os.environ.get(LICHESS_TOKEN_ENV)
if not LICHESS_TOKEN:
    LOG.warning("No Lichess token found in env var '%s'. Set it before starting.", LICHESS_TOKEN_ENV)

MODEL_PATH = os.environ.get("MODEL_PATH", "model.h5")
VOCAB_PATH = os.environ.get("VOCAB_PATH", "vocab.npz")

# inference behaviour
ARGMAX = os.environ.get("ARGMAX", "false").lower() in ("1", "true", "yes")
TOP_K = int(os.environ.get("TOP_K", "32"))
TEMPERATURE = float(os.environ.get("TEMP", "1.0"))

# other
MOVE_DELAY = float(os.environ.get("MOVE_DELAY", "0.25"))
BACKOFF_BASE = float(os.environ.get("BACKOFF_BASE", "1.0"))
BACKOFF_MAX = float(os.environ.get("BACKOFF_MAX", "60.0"))
HEARTBEAT_TIMEOUT = int(os.environ.get("HEARTBEAT_TIMEOUT", "180"))

# ---------- GLOBALS ----------
app = Flask(__name__)
_client_lock = threading.Lock()
global_client = None  # will hold berserk client
MODEL = None
VOCAB_MOVES: List[str] = []
MOVE2IDX: Dict[str, int] = {}
IDX2MOVE: List[str] = []
MODEL_INPUT_SHAPE = None
# We'll record per-game info for dashboard
GAMES: Dict[str, Dict[str, Any]] = {}
GAMES_LOCK = threading.Lock()

# ---------- UTIL: download helper (optional) ----------
def maybe_download(url: Optional[str], dest: str):
    if not url:
        return
    if os.path.exists(dest):
        LOG.info("File exists: %s", dest)
        return
    LOG.info("Downloading %s -> %s", url, dest)
    import requests
    r = requests.get(url, stream=True)
    r.raise_for_status()
    tmp = dest + ".tmp"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)
    os.replace(tmp, dest)
    LOG.info("Downloaded %s", dest)

# ---------- LOAD VOCAB (flexible) ----------
def load_vocab(path: str) -> None:
    global VOCAB_MOVES, MOVE2IDX, IDX2MOVE
    if not os.path.exists(path):
        raise FileNotFoundError(f"vocab file not found: {path}")
    data = np.load(path, allow_pickle=True)
    keys = list(data.files)
    LOG.info("vocab.npz keys: %s", keys)

    # Try common keys
    moves = None
    move2idx = None
    idx2move = None

    if "moves" in keys:
        moves = data["moves"]
    if "move2idx" in keys:
        move2idx = data["move2idx"]
    if "idx2move" in keys:
        idx2move = data["idx2move"]

    # alt names
    if moves is None and "vocab" in keys:
        moves = data["vocab"]
    if moves is None and len(keys) == 1:
        moves = data[keys[0]]

    if idx2move is not None:
        try:
            IDX2MOVE = [str(x) for x in idx2move.tolist()]
        except Exception:
            IDX2MOVE = [str(x) for x in idx2move]
        MOVE2IDX = {m: i for i, m in enumerate(IDX2MOVE)}
    elif moves is not None:
        try:
            VOCAB_MOVES = [str(x) for x in moves.tolist()]
        except Exception:
            VOCAB_MOVES = [str(x) for x in moves]
        MOVE2IDX = {m: i for i, m in enumerate(VOCAB_MOVES)}
        IDX2MOVE = VOCAB_MOVES.copy()
    elif move2idx is not None:
        try:
            m2i = dict(move2idx.tolist())
        except Exception:
            m2i = dict(move2idx)
        # invert sorted by index
        IDX2MOVE = [None] * (max(m2i.values()) + 1)
        for m, idx in m2i.items():
            IDX2MOVE[idx] = str(m)
        MOVE2IDX = {m: i for i, m in enumerate(IDX2MOVE)}
        VOCAB_MOVES = IDX2MOVE.copy()
    else:
        # try heuristics
        for k in keys:
            arr = data[k]
            try:
                cand = [str(x) for x in arr]
                if all(0 < len(s) <= 6 for s in cand):  # 'e2e4' style
                    VOCAB_MOVES = cand
                    MOVE2IDX = {m: i for i, m in enumerate(VOCAB_MOVES)}
                    IDX2MOVE = VOCAB_MOVES.copy()
                    LOG.warning("Fallback: using key %s for moves", k)
                    break
            except Exception:
                continue

    if not (MOVE2IDX and IDX2MOVE):
        raise ValueError("Could not parse vocab.npz into moves and indices. Keys: %s" % keys)

    LOG.info("Vocab loaded: %d moves", len(IDX2MOVE))

# ---------- LOAD MODEL ----------
def load_keras_model(path: str):
    global MODEL, MODEL_INPUT_SHAPE
    if load_model is None:
        raise RuntimeError("TensorFlow/Keras not available in environment.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    LOG.info("Loading Keras model from %s", path)
    MODEL = load_model(path)
    # Try to infer input shape
    try:
        inp = MODEL.inputs[0].shape
        if hasattr(inp, "as_list"):
            dims = inp.as_list()
        else:
            dims = list(inp)
        MODEL_INPUT_SHAPE = tuple(dims)
        LOG.info("Model input shape: %s", MODEL_INPUT_SHAPE)
    except Exception:
        LOG.warning("Could not infer model input shape; will try multiple encoders.")

# ---------- AUTO-RELOAD convenience ----------
def reload_model_and_vocab():
    global MODEL, MOVE2IDX, IDX2MOVE, VOCAB_MOVES
    # optional remote download (if you set MODEL_URL / VOCAB_URL env)
    maybe_download(os.environ.get("MODEL_URL"), MODEL_PATH)
    maybe_download(os.environ.get("VOCAB_URL"), VOCAB_PATH)

    load_vocab(VOCAB_PATH)
    load_keras_model(MODEL_PATH)
    LOG.info("Reload complete.")

# Try initial load (don't crash whole process if fails)
try:
    reload_model_and_vocab()
except Exception:
    LOG.exception("Initial model/vocab load failed. Use /admin/reload to reload after fixing files.")

# ---------- ENCODERS ----------
# 1) Sequence-token encoder: last N moves -> indices padded to seq_len (integers)
SEQ_LEN = int(os.environ.get("SEQ_LEN", "128"))

def encode_sequence_tokens(board: chess.Board, seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    Create integer token sequence of last moves, padded left with zeros.
    Format: indices of moves (MOVE2IDX). Unknown moves => -1 replaced with 0.
    Returns shape (1, seq_len) dtype=int32
    """
    # get moves in UCI order
    moves = [m.uci() for m in board.move_stack]
    # take last seq_len moves
    toks = []
    for mv in moves[-seq_len:]:
        idx = MOVE2IDX.get(mv)
        if idx is None:
            toks.append(0)
        else:
            toks.append(idx + 1)  # reserve 0 for padding
    # pad to left
    if len(toks) < seq_len:
        toks = [0] * (seq_len - len(toks)) + toks
    arr = np.array(toks, dtype=np.int32).reshape(1, seq_len)
    return arr

# 2) Plane encoder: classic 8x8x12 flattened -> 768 dims (0/1)
PIECE_TO_PLANE = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
}

def encode_board_planes(board: chess.Board) -> np.ndarray:
    arr = np.zeros((8, 8, 12), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        r = chess.square_rank(sq)
        c = chess.square_file(sq)
        plane = PIECE_TO_PLANE.get(piece.symbol())
        if plane is not None:
            arr[r, c, plane] = 1.0
    flat = arr.reshape(1, -1).astype(np.float32)
    return flat

# 3) One-hot move vector (vocab-size)
def encode_onehot_movevec(board: chess.Board) -> np.ndarray:
    # naive summary-vector of position: freq of moves present in history
    vec = np.zeros((1, len(IDX2MOVE)), dtype=np.float32)
    for mv in [m.uci() for m in board.move_stack]:
        idx = MOVE2IDX.get(mv)
        if idx is not None:
            vec[0, idx] += 1.0
    # optionally normalize
    s = vec.sum()
    if s > 0:
        vec /= s
    return vec

# ---------- INFERENCE: Try encoders until one works ----------
def infer_probs_from_model(board: chess.Board) -> np.ndarray:
    """
    Returns a 1-D numpy array of length len(IDX2MOVE) with probabilities.
    Strategy:
      - Try model.predict with sequence-token shape (1, SEQ_LEN) (integers) if model accepts ints.
      - Try model.predict with plane encoding (1, 768)
      - Try one-hot move vector (1, vocab_size)
    """
    if MODEL is None:
        raise RuntimeError("Model not loaded")

    errors = []
    # 1) sequence tokens
    try:
        seq = encode_sequence_tokens(board)
        out = MODEL.predict(seq, verbose=0)
        probs = np.array(out[0], dtype=np.float64)
        probs = normalize_probs(probs)
        LOG.debug("Used sequence-token encoder")
        return probs
    except Exception as e:
        errors.append(("seq", str(e)))

    # 2) plane encoder
    try:
        plane = encode_board_planes(board)
        out = MODEL.predict(plane, verbose=0)
        probs = np.array(out[0], dtype=np.float64)
        probs = normalize_probs(probs)
        LOG.debug("Used plane encoder")
        return probs
    except Exception as e:
        errors.append(("plane", str(e)))

    # 3) one-hot move-vector
    try:
        v = encode_onehot_movevec(board)
        out = MODEL.predict(v, verbose=0)
        probs = np.array(out[0], dtype=np.float64)
        probs = normalize_probs(probs)
        LOG.debug("Used one-hot encoder")
        return probs
    except Exception as e:
        errors.append(("onehot", str(e)))

    LOG.warning("All encoders failed to run model. Errors: %s", errors)
    raise RuntimeError("Model inference failed for all encoders")

def normalize_probs(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Empty model output")
    # if output already sums to ~1 and non-negative, accept it
    if (arr >= 0).all() and (abs(arr.sum() - 1.0) < 1e-3):
        return arr
    # otherwise treat as logits -> softmax
    ex = np.exp(arr - np.max(arr))
    probs = ex / (ex.sum() + 1e-12)
    return probs

# ---------- CHOOSE LEGAL MOVE ----------
def choose_move_from_probs(board: chess.Board, probs: np.ndarray,
                           argmax: bool = ARGMAX, top_k: int = TOP_K, temp: float = TEMPERATURE) -> Optional[str]:
    legal = [m.uci() for m in board.legal_moves]
    if not legal:
        return None

    # map legal moves to indices known to model
    items = []
    for mv in legal:
        idx = MOVE2IDX.get(mv)
        if idx is not None and 0 <= idx < len(probs):
            items.append((mv, float(probs[idx]), idx))
    if not items:
        # model doesn't know legal moves: fallback to sampling by heuristics (random)
        return random.choice(legal)

    # sort by score desc
    items.sort(key=lambda x: -x[1])
    ucis = [it[0] for it in items]
    scores = np.array([it[1] for it in items], dtype=np.float64)

    if argmax:
        return ucis[0]

    # top-k sampling among known legal moves
    k = min(top_k, len(scores))
    top_scores = scores[:k]
    # temperature softmax
    if (top_scores < 0).any() or top_scores.sum() <= 1e-8:
        # treat as logits
        logits = top_scores - np.max(top_scores)
        ex = np.exp(logits / (temp if temp > 0 else 1e-6))
        p = ex / (ex.sum() + 1e-12)
    else:
        p = softmax(top_scores, temp)
    choice = np.random.choice(range(k), p=p)
    return ucis[choice]

def softmax(x: np.ndarray, temp: float) -> np.ndarray:
    if temp <= 0:
        temp = 1e-6
    a = x / temp
    a = a - np.max(a)
    e = np.exp(a)
    return e / (e.sum() + 1e-12)

# ---------- Berserk wrapper with retry/backoff ----------
def create_global_client():
    global global_client
    with _client_lock:
        if global_client is None:
            session = berserk.TokenSession(LICHESS_TOKEN)
            global_client = berserk.Client(session=session)
            LOG.info("Created global Berserk client")
    return global_client

def retry_berserk_call(func, *args, max_attempts: int = 6, base_wait: float = BACKOFF_BASE, **kwargs):
    wait = base_wait
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Berserk raises requests exceptions for HTTP errors
            msg = str(e)
            LOG.warning("Call failed (attempt %d/%d): %s", attempt, max_attempts, msg)
            # if rate-limited, backoff
            if "429" in msg or "Too Many Requests" in msg:
                LOG.warning("Rate-limited; sleeping %.1fs", wait)
                time.sleep(wait)
                wait = min(wait * 2, BACKOFF_MAX)
                continue
            else:
                # small backoff and retry for transient errors
                time.sleep(min(wait, BACKOFF_MAX))
                wait = min(wait * 2, BACKOFF_MAX)
    raise RuntimeError("Max retries exceeded for berserk call")

# ---------- GAME HANDLER (per-game client) ----------
def handle_game(game_id: str, my_color: chess.Color):
    LOG.info("[%s] handler starting, color=%s", game_id, "WHITE" if my_color else "BLACK")
    # create a per-thread berserk client (avoids session issues across threads)
    local_session = berserk.TokenSession(LICHESS_TOKEN)
    local_client = berserk.Client(session=local_session)

    board = chess.Board()
    # init in store
    with GAMES_LOCK:
        GAMES[game_id] = {"moves": [], "white": None, "black": None, "last_thought": None, "result": None, "ts": time.time()}

    try:
        for event in local_client.bots.stream_game_state(game_id):
            # event types: gameFull (initial), gameState (updates)
            etype = event.get("type")
            if etype not in ("gameFull", "gameState"):
                continue
            state = event.get("state", event)
            moves_str = state.get("moves", "")
            moves = moves_str.split() if moves_str else []
            # rebuild board
            board = chess.Board()
            for u in moves:
                try:
                    board.push_uci(u)
                except Exception:
                    LOG.debug("[%s] failed pushing move %s", game_id, u)
            # store white/black ids if present
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
                    g["ts"] = time.time()

            # check terminal
            status = state.get("status")
            if status in ("mate", "resign", "timeout", "draw", "stalemate", "outoftime"):
                winner = state.get("winner")
                result = "1-0" if winner == "white" else ("0-1" if winner == "black" else "1/2-1/2")
                with GAMES_LOCK:
                    GAMES[game_id]["result"] = result
                LOG.info("[%s] finished: %s", game_id, result)
                break

            # check if it's our turn
            # fetch our account id (cached from global client)
            acct = create_global_client().account.get()
            our_id = acct.get("id")
            # board.turn True means white to move
            is_my_turn = False
            if board.turn:
                # white to move
                is_my_turn = (state.get("white", {}).get("id") == our_id) or (state.get("white", {}).get("name") == our_id)
            else:
                is_my_turn = (state.get("black", {}).get("id") == our_id) or (state.get("black", {}).get("name") == our_id)

            if is_my_turn and not board.is_game_over():
                LOG.info("[%s] It's our turn. Moves so far: %s", game_id, " ".join(moves))
                # infer probs
                try:
                    probs = infer_probs_from_model(board)
                    chosen_uci = choose_move_from_probs(board, probs, argmax=ARGMAX, top_k=TOP_K, temp=TEMPERATURE)
                    if not chosen_uci:
                        # fallback
                        chosen_uci = random.choice([m.uci() for m in board.legal_moves])
                        reason = "fallback-random"
                    else:
                        reason = "model-sampled" if not ARGMAX else "model-argmax"

                    thought = {"chosen": chosen_uci, "reason": reason, "top": []}
                    # compute top 5 for thought
                    try:
                        top_idx = np.argsort(probs)[::-1][:8]
                        for i in top_idx:
                            mv = IDX2MOVE[i] if i < len(IDX2MOVE) else None
                            score = float(probs[i]) if i < len(probs) else 0.0
                            thought["top"].append((mv, score))
                    except Exception:
                        pass

                    with GAMES_LOCK:
                        GAMES[game_id]["last_thought"] = thought

                    # send move with retry wrapper
                    try:
                        retry_berserk_call(local_client.bots.make_move, game_id, chosen_uci)
                        LOG.info("[%s] played %s (%s)", game_id, chosen_uci, reason)
                    except Exception:
                        LOG.exception("[%s] failed to play %s; fallback to random", game_id, chosen_uci)
                        fb = random.choice([m.uci() for m in board.legal_moves])
                        try:
                            retry_berserk_call(local_client.bots.make_move, game_id, fb)
                            LOG.info("[%s] played fallback %s", game_id, fb)
                        except Exception:
                            LOG.exception("[%s] fallback also failed", game_id)
                except Exception:
                    LOG.exception("[%s] model inference failed, playing random", game_id)
                    fb = random.choice([m.uci() for m in board.legal_moves])
                    try:
                        retry_berserk_call(local_client.bots.make_move, game_id, fb)
                    except Exception:
                        LOG.exception("[%s] fallback play failed", game_id)

                time.sleep(MOVE_DELAY)
            else:
                LOG.debug("[%s] Not our turn", game_id)
    except Exception:
        LOG.exception("[%s] exception in game handler", game_id)
    finally:
        LOG.info("[%s] handler exiting", game_id)

# ---------- MAIN EVENT LOOP ----------
def main_event_loop():
    cl = create_global_client()
    LOG.info("Starting incoming events stream")
    for event in cl.bots.stream_incoming_events():
        try:
            etype = event.get("type")
            LOG.info("Event: %s", etype)
            if etype == "challenge":
                chal = event.get("challenge", {})
                cid = chal.get("id")
                variant = chal.get("variant", {}).get("key")
                LOG.info("Challenge %s variant=%s", cid, variant)
                if variant in ("standard", "fromPosition"):
                    try:
                        retry_berserk_call(cl.bots.accept_challenge, cid)
                        LOG.info("Accepted %s", cid)
                    except Exception:
                        LOG.exception("Failed to accept challenge %s", cid)
                else:
                    try:
                        retry_berserk_call(cl.bots.decline_challenge, cid)
                        LOG.info("Declined %s", cid)
                    except Exception:
                        LOG.exception("Failed to decline %s", cid)
            elif etype == "gameStart":
                gid = event.get("game", {}).get("id")
                color_str = event.get("game", {}).get("color")
                my_color = chess.WHITE if color_str == "white" else chess.BLACK
                LOG.info("GameStart %s color=%s", gid, color_str)
                t = threading.Thread(target=handle_game, args=(gid, my_color), daemon=True)
                t.start()
        except Exception:
            LOG.exception("Error in main event loop processing event")

# ---------- THREAD MANAGEMENT ----------
_event_thread = None
_event_thread_lock = threading.Lock()
_stop_flag = threading.Event()

def start_event_thread():
    global _event_thread
    with _event_thread_lock:
        if _event_thread and _event_thread.is_alive():
            LOG.info("Event thread already running")
            return
        _stop_flag.clear()
        def worker():
            backoff = BACKOFF_BASE
            while not _stop_flag.is_set():
                try:
                    if global_client is None:
                        create_global_client()
                    main_event_loop()
                    LOG.warning("Event loop ended; reconnecting after %.1fs", backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, BACKOFF_MAX)
                except Exception:
                    LOG.exception("Event thread exception; backing off %.1fs", backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, BACKOFF_MAX)
        _event_thread = threading.Thread(target=worker, daemon=True)
        _event_thread.start()
        LOG.info("Started event thread")

def stop_event_thread():
    _stop_flag.set()
    with _event_thread_lock:
        if _event_thread and _event_thread.is_alive():
            LOG.info("Stopping event thread")
            _event_thread.join(timeout=2.0)

# ---------- FLASK DEBUG / ADMIN ----------
@app.route("/")
def index():
    with GAMES_LOCK:
        return jsonify({gid: {"moves": g["moves"], "last_thought": g["last_thought"], "result": g["result"], "white": g["white"], "black": g["black"]} for gid, g in GAMES.items()})

@app.route("/debug")
def debug():
    with GAMES_LOCK:
        return jsonify(GAMES)

@app.route("/admin/reload", methods=["POST"])
def admin_reload():
    try:
        reload_model_and_vocab()
        return jsonify({"status": "ok", "vocab_size": len(IDX2MOVE)})
    except Exception as e:
        LOG.exception("reload error")
        return jsonify({"error": str(e)}), 500

@app.route("/admin/stats")
def admin_stats():
    return jsonify({
        "model_loaded": MODEL is not None,
        "vocab_size": len(IDX2MOVE),
        "active_games": len(GAMES)
    })

# ---------- BOOT ----------
def boot():
    # create client
    try:
        create_global_client()
    except Exception:
        LOG.exception("Could not create global client at boot (will retry in thread)")

    start_event_thread()

# ---------- helpers ----------
def create_global_client():
    global global_client
    with _client_lock:
        if global_client is None:
            if not LICHESS_TOKEN:
                raise RuntimeError(f"Lichess token not set in env var {LICHESS_TOKEN_ENV}")
            session = berserk.TokenSession(LICHESS_TOKEN)
            global_client = berserk.Client(session=session)
            LOG.info("Global berserk client created")
    return global_client

# ---------- run ----------
if __name__ == "__main__":
    LOG.info("Booting player-imitation bot")
    boot()
    # heartbeat to ensure event thread alive
    def heartbeat():
        while True:
            try:
                alive = False
                with _event_thread_lock:
                    if _event_thread:
                        alive = _event_thread.is_alive()
                if not alive:
                    LOG.warning("Event thread not alive; restarting")
                    start_event_thread()
                time.sleep(max(5, HEARTBEAT_TIMEOUT // 6))
            except Exception:
                LOG.exception("Heartbeat error")
                time.sleep(5)
    t = threading.Thread(target=heartbeat, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 10000))
    LOG.info("Starting Flask on port %d", port)
    app.run(host="0.0.0.0", port=port, threaded=True)
