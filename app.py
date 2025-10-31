#!/usr/bin/env python3
"""
app.py - Integrated Lichess bot using user's 'worst survivable' handler but neural inference core.

Usage:
  - Place model.h5 and vocab.npz in repo root.
  - Set environment variable Lichess_token to your berserk token.
  - Run: python app.py  (or use gunicorn for deployment)
"""

import os
import sys
import time
import json
import math
import random
import logging
import threading
import traceback
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import requests
import berserk
import chess
import chess.pgn
from flask import Flask, jsonify, request, render_template_string

# try tensorflow (deferred import to allow fast failure with clear message)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception as e:
    tf = None
    load_model = None

# ---------------------------
# Config / environment
# ---------------------------
STORAGE_DIR = os.environ.get("LOGDIR", "games")
os.makedirs(STORAGE_DIR, exist_ok=True)

  # preserved to match your code
LICHESS_TOKEN = os.environ.get('Lichess_token')
if not LICHESS_TOKEN:
    logging.warning("No Lichess token found in env var '%s'. Bot will not start until you set it.", LICHESS_TOKEN_ENV_NAME)

MODEL_PATH = os.environ.get("MODEL_PATH", "model.h5")
VOCAB_PATH = os.environ.get("VOCAB_PATH", "vocab.npz")
MODEL_URL = os.environ.get("MODEL_URL", None)  # optional remote URLs
VOCAB_URL = os.environ.get("VOCAB_URL", None)

# sampling params
DEFAULT_ARGMAX = os.environ.get("DEFAULT_ARGMAX", "false").lower() in ("1", "true", "yes")
DEFAULT_TOPK = int(os.environ.get("DEFAULT_TOPK", "40"))
DEFAULT_TEMP = float(os.environ.get("DEFAULT_TEMP", "1.0"))

# retry/backoff config
BASE_BACKOFF = float(os.environ.get("BASE_BACKOFF", "1.0"))
MAX_BACKOFF = float(os.environ.get("MAX_BACKOFF", "60.0"))

# move delay after playing
MOVE_DELAY_SECONDS = float(os.environ.get("MOVE_DELAY_SECONDS", "0.4"))

# heartbeat timeout (seconds without events to attempt reconnect)
HEARTBEAT_TIMEOUT = int(os.environ.get("HEARTBEAT_TIMEOUT", "180"))

# Logging setup
logger = logging.getLogger("person-bot")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s"))
logger.addHandler(ch)

# Flask app for keep-alive and debug
app = Flask(__name__)

# ---------------------------
# Helper: download file if URL given
# ---------------------------
def download_file_if_needed(url: Optional[str], path: str):
    if not url:
        return
    if os.path.exists(path):
        logger.info("File %s exists locally, skipping download from %s", path, url)
        return
    logger.info("Downloading %s -> %s", url, path)
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)
    os.replace(tmp, path)
    logger.info("Downloaded %s", path)

# ---------------------------
# Model + Vocab loader
# ---------------------------
class ModelBundle:
    def __init__(self, model_path=MODEL_PATH, vocab_path=VOCAB_PATH):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.model = None
        self.move_list: List[str] = []
        self.move_to_idx: Dict[str, int] = {}
        self.input_dim = None

    def maybe_download(self):
        download_file_if_needed(MODEL_URL, self.model_path)
        download_file_if_needed(VOCAB_URL, self.vocab_path)

    def load_vocab(self):
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"vocab not found: {self.vocab_path}")
        logger.info("Loading vocab from %s", self.vocab_path)
        data = np.load(self.vocab_path, allow_pickle=True)
        keys = list(data.files)
        logger.info("vocab.npz keys: %s", keys)

        # Try a few likely key names
        moves = None
        move2idx = None
        idx2move = None

        if "moves" in keys:
            moves = data["moves"]
        if "idx2move" in keys:
            idx2move = data["idx2move"]
        if "move2idx" in keys:
            move2idx = data["move2idx"]
        # alternative names
        if moves is None and "vocab" in keys:
            moves = data["vocab"]
        if moves is None and len(keys) == 1:
            moves = data[keys[0]]

        # If idx2move provided as mapping array/object
        if idx2move is not None:
            try:
                self.move_list = [str(x) for x in idx2move.tolist()]
            except Exception:
                self.move_list = [str(x) for x in idx2move]
        elif moves is not None:
            self.move_list = [str(x) for x in moves]
        elif move2idx is not None:
            # move2idx might be a dict-like
            try:
                mv2i = dict(move2idx.tolist()) if hasattr(move2idx, "tolist") else dict(move2idx)
            except Exception:
                mv2i = dict(move2idx)
            # invert
            inv = sorted(mv2i.items(), key=lambda kv: kv[1])
            self.move_list = [kv[0] for kv in inv]
        else:
            # last resort, try to parse any arrays in file
            for k in keys:
                arr = data[k]
                try:
                    cand = [str(x) for x in arr]
                    if all(len(s) <= 6 for s in cand):  # move heuristics like 'e2e4' or 'g1f3'
                        self.move_list = cand
                        logger.warning("Fallback: using key %s for moves", k)
                        break
                except Exception:
                    continue

        if not self.move_list:
            raise ValueError("Could not parse move list from vocab.npz; keys: %s" % keys)

        # build move_to_idx
        self.move_to_idx = {m: i for i, m in enumerate(self.move_list)}
        logger.info("Loaded vocab size: %d moves", len(self.move_list))

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"model not found: {self.model_path}")
        if load_model is None:
            raise RuntimeError("TensorFlow not available (unable to import load_model).")
        logger.info("Loading Keras model from %s", self.model_path)
        self.model = load_model(self.model_path)
        # try to infer input_dim
        try:
            inp_shape = self.model.inputs[0].shape
            if hasattr(inp_shape, "as_list"):
                dims = inp_shape.as_list()[1:]
            else:
                dims = list(inp_shape)[1:]
            dim = 1
            for d in dims:
                dim *= (d or 1)
            self.input_dim = int(dim)
            logger.info("Inferred model input_dim=%d (dims=%s)", self.input_dim, dims)
        except Exception:
            logger.exception("Failed to infer model input dim; leave as None")

    def reload(self):
        logger.info("Reloading model bundle")
        self.maybe_download()
        self.load_vocab()
        self.load_model()
        logger.info("Model bundle reload complete")

# instantiate
MODEL_BUNDLE = ModelBundle()

try:
    MODEL_BUNDLE.reload()
except Exception:
    logger.exception("Initial model/vocab load failed; you can POST to /admin/reload after fixing files")

# ---------------------------
# Utility: encode board for model
# ---------------------------
# Many models have different input modalities. We will use a safe, generic encoding:
#  - Standard 8x8x12 binary planes flattened to 768 dims (if model expects that).
#  - If model input dim differs, we will pad/truncate the flattened array to match.
PIECE_TO_PLANE = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
}

def fen_to_planes(fen: str):
    board = chess.Board(fen)
    arr = np.zeros((8, 8, 12), dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        r = chess.square_rank(sq)
        c = chess.square_file(sq)
        plane = PIECE_TO_PLANE.get(piece.symbol())
        if plane is not None:
            arr[r, c, plane] = 1
    return arr

def fen_to_input(fen: str):
    flat = fen_to_planes(fen).reshape(-1).astype(np.float32)
    if MODEL_BUNDLE.input_dim:
        if flat.size < MODEL_BUNDLE.input_dim:
            pad = np.zeros(MODEL_BUNDLE.input_dim - flat.size, dtype=np.float32)
            flat = np.concatenate([flat, pad])
        elif flat.size > MODEL_BUNDLE.input_dim:
            flat = flat[:MODEL_BUNDLE.input_dim]
    # final shape (1, dim)
    return flat.reshape(1, -1)

# ---------------------------
# Model inference & move selection
# ---------------------------
def model_predict_probs_for_fen(fen: str) -> np.ndarray:
    if MODEL_BUNDLE.model is None:
        raise RuntimeError("Model not loaded")
    x = fen_to_input(fen)
    preds = MODEL_BUNDLE.model.predict(x, verbose=0)[0]
    probs = np.array(preds, dtype=np.float64)
    # Try softmax-stabilized if it's logits
    if (probs < 0).any() or probs.sum() <= 1e-8:
        # assume logits -> softmax
        ex = np.exp(probs - np.max(probs))
        probs = ex / (ex.sum() + 1e-12)
    return probs

def choose_move_from_model(board: chess.Board, probs: np.ndarray,
                           argmax: bool = DEFAULT_ARGMAX,
                           top_k: int = DEFAULT_TOPK,
                           temp: float = DEFAULT_TEMP) -> Optional[str]:
    legal_moves = [m.uci() for m in board.legal_moves]
    if not legal_moves:
        return None

    legal_scores: List[Tuple[str, float]] = []
    for u in legal_moves:
        idx = MODEL_BUNDLE.move_to_idx.get(u)
        if idx is None:
            continue
        if idx < 0 or idx >= len(probs):
            continue
        legal_scores.append((u, float(probs[idx])))

    if not legal_scores:
        # model doesn't know any legal moves => fallback to random legal
        return random.choice(legal_moves)

    # sort descending
    legal_scores.sort(key=lambda x: -x[1])
    ucis = [p[0] for p in legal_scores]
    scores = np.array([p[1] for p in legal_scores], dtype=np.float64)

    if argmax:
        return ucis[0]

    # top-k sampling
    k = min(top_k, len(scores))
    top_scores = scores[:k]
    # temperature softmax
    if top_scores.sum() <= 0:
        # treat as logits
        top_scores = top_scores - np.max(top_scores)
        ex = np.exp(top_scores / (temp if temp > 0 else 1e-6))
        sample_p = ex / (ex.sum() + 1e-12)
    else:
        sample_p = softmax(top_scores, temp)
    choice = np.random.choice(range(k), p=sample_p)
    return ucis[choice]

def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    if temp <= 0:
        temp = 1e-6
    a = x / temp
    a = a - np.max(a)
    e = np.exp(a)
    return e / (e.sum() + 1e-12)

# ---------------------------
# Lichess client wrapper + retry/backoff
# ---------------------------
session = None
client = None

def create_client():
    global session, client
    if not LICHESS_TOKEN:
        raise RuntimeError("Environment Lichess_token not set")
    session = berserk.TokenSession(LICHESS_TOKEN)
    client = berserk.Client(session=session)
    logger.info("Created berserk client")
    return client

def retry_call(func, *args, max_attempts=6, base_wait=BASE_BACKOFF, **kwargs):
    wait = base_wait
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.HTTPError as he:
            code = getattr(he.response, "status_code", None)
            logger.warning("HTTP error %s on attempt %d/%d for %s", code, attempt, max_attempts, func)
            if code == 429:
                logger.warning("Rate-limited (429). Sleeping %.1fs before retry", wait)
                time.sleep(wait)
                wait = min(wait * 2.0, MAX_BACKOFF)
                continue
            else:
                raise
        except Exception as e:
            logger.warning("Exception on attempt %d/%d: %s", attempt, max_attempts, e)
            time.sleep(wait)
            wait = min(wait * 2.0, MAX_BACKOFF)
    raise RuntimeError("Max retry attempts exceeded for call")

# ---------------------------
# Store and dashboard
# ---------------------------
GAME_STORE: Dict[str, dict] = {}  # game_id -> {moves:[], white:, black:, last_thought: {}}
GAME_STORE_LOCK = threading.Lock()

def save_game_entry(game_id: str, white: str, black: str, moves: List[str], result: Optional[str] = None):
    os.makedirs(STORAGE_DIR, exist_ok=True)
    path = os.path.join(STORAGE_DIR, f"{game_id}.pgn")
    game = chess.pgn.Game()
    game.headers["Event"] = "Model-played game"
    game.headers["White"] = white or "white"
    game.headers["Black"] = black or "black"
    game.headers["Result"] = result or "*"
    node = game
    for uci in moves:
        try:
            node = node.add_variation(chess.Move.from_uci(uci))
        except Exception:
            pass
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(game))
    logger.info("Saved PGN %s", path)
    return path

def update_game_store(game_id: str, **kwargs):
    with GAME_STORE_LOCK:
        g = GAME_STORE.get(game_id)
        if not g:
            g = {"moves": [], "white": None, "black": None, "last_thought": None, "result": None, "ts": time.time()}
            GAME_STORE[game_id] = g
        for k, v in kwargs.items():
            g[k] = v
        g["ts"] = time.time()
        return g

# ---------------------------
# Core - adapted handle_game using model
# ---------------------------
def handle_game(game_id: str, my_color: chess.Color):
    logger.info("[%s] Game handler started. Color: %s", game_id, "WHITE" if my_color else "BLACK")
    board = chess.Board()
    try:
        # stream_game_state yields events for this game
        for event in client.bots.stream_game_state(game_id):
            logger.debug("[%s] Event: %s", game_id, event.get("type"))
            state = event.get("state", event)  # sometimes nested
            moves_str = state.get("moves", "")
            moves = moves_str.split() if moves_str else []
            # reconstruct board
            board = chess.Board()
            for m in moves:
                try:
                    board.push_uci(m)
                except Exception:
                    logger.debug("[%s] failed to push historical move: %s", game_id, m)
            # update store
            white = state.get("white", {}).get("name") or state.get("white", {}).get("id")
            black = state.get("black", {}).get("name") or state.get("black", {}).get("id")
            update_game_store(game_id, moves=moves, white=white, black=black)

            # check for finished
            status = state.get("status")
            if status in ("mate", "resign", "timeout", "draw", "outoftime", "stalemate"):
                winner = state.get("winner")
                result = "1-0" if winner == "white" else ("0-1" if winner == "black" else "1/2-1/2")
                update_game_store(game_id, result=result)
                save_game_entry(game_id, white, black, moves, result=result)
                logger.info("[%s] Game finished: status=%s result=%s", game_id, status, result)
                break

            # is it our turn?
            is_my_turn = False
            if board.turn:
                is_my_turn = state.get("white", {}).get("id") == client.account.get()["id"]
            else:
                is_my_turn = state.get("black", {}).get("id") == client.account.get()["id"]

            if is_my_turn and not board.is_game_over():
                logger.info("[%s] It's our turn (moves so far: %s)", game_id, ' '.join(moves))
                # Model inference
                try:
                    probs = model_predict_probs_for_fen(board.fen())
                    chosen = choose_move_from_model(board, probs, argmax=DEFAULT_ARGMAX, top_k=DEFAULT_TOPK, temp=DEFAULT_TEMP)
                    if not chosen:
                        chosen = random.choice([m.uci() for m in board.legal_moves])
                        reason = "fallback-random-no-known-legal"
                    else:
                        reason = "model-chosen"
                    # log thought
                    topk = 8
                    idxs = np.argsort(probs)[::-1][:topk]
                    top_moves = []
                    top_scores = []
                    for i in idxs:
                        mv = MODEL_BUNDLE.move_list[i] if i < len(MODEL_BUNDLE.move_list) else None
                        top_moves.append(mv)
                        top_scores.append(float(probs[i]))
                    thought = {"chosen": chosen, "reason": reason, "top_moves": top_moves, "top_scores": top_scores}
                    update_game_store(game_id, last_thought=thought)
                    # make move with retry protection
                    try:
                        retry_call(client.bots.make_move, game_id, chosen)
                        logger.info("[%s] Played %s (%s)", game_id, chosen, reason)
                    except Exception:
                        logger.exception("[%s] Failed to play chosen move %s; falling back to random", game_id, chosen)
                        fallback = random.choice([m.uci() for m in board.legal_moves])
                        try:
                            retry_call(client.bots.make_move, game_id, fallback)
                            logger.info("[%s] Played fallback %s", game_id, fallback)
                        except Exception:
                            logger.exception("[%s] Fallback move also failed", game_id)
                except Exception:
                    logger.exception("[%s] Model inference or move sending failed, picking random", game_id)
                    fallback = random.choice([m.uci() for m in board.legal_moves])
                    try:
                        retry_call(client.bots.make_move, game_id, fallback)
                    except Exception:
                        logger.exception("[%s] fallback move also failed", game_id)
                # slight delay so we don't hammer (and to give event stream time to update)
                time.sleep(MOVE_DELAY_SECONDS)
            else:
                logger.debug("[%s] Not our turn", game_id)

    except Exception:
        logger.exception("[%s] Exception in game handler stream", game_id)
    finally:
        logger.info("[%s] Game handler terminating", game_id)

# ---------------------------
# Main incoming event loop (keeps original logic)
# ---------------------------
def main_event_loop():
    # create client if missing
    global client
    if client is None:
        create_client()
    logger.info("Starting incoming events stream")
    try:
        for event in client.bots.stream_incoming_events():
            logger.info("Incoming event: %s", event.get("type"))
            if event.get("type") == "challenge":
                chal = event.get("challenge", {})
                variant = chal.get("variant", {}).get("key")
                cid = chal.get("id")
                if variant in ("standard", "fromPosition"):
                    try:
                        retry_call(client.bots.accept_challenge, cid)
                        logger.info("Accepted challenge %s (variant=%s)", cid, variant)
                    except Exception:
                        logger.exception("Failed to accept challenge %s", cid)
                else:
                    try:
                        retry_call(client.bots.decline_challenge, cid)
                        logger.info("Declined challenge %s (variant=%s)", cid, variant)
                    except Exception:
                        logger.exception("Failed to decline challenge %s", cid)
            elif event.get("type") == "gameStart":
                gid = event.get("game", {}).get("id")
                color_str = event.get("game", {}).get("color")
                my_color = chess.WHITE if color_str == "white" else chess.BLACK
                logger.info("GameStart: %s color=%s", gid, color_str)
                t = threading.Thread(target=handle_game, args=(gid, my_color), daemon=True)
                t.start()
    except requests.exceptions.HTTPError as he:
        logger.exception("HTTP error in incoming events stream: %s", he)
        # upstream may have returned 429 or similar; caller should manage reconnects
    except Exception:
        logger.exception("Unexpected error in incoming events stream")
    finally:
        logger.warning("Incoming events stream ended; main_event_loop terminating")

# We'll run the main_event_loop in a resilient thread that restarts on disconnects
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
        def _worker():
            backoff = 1.0
            while not _stop_flag.is_set():
                try:
                    if client is None:
                        create_client()
                    main_event_loop()
                    # if loop ended (e.g., connection closed), wait and reconnect
                    logger.warning("Stream closed, reconnecting after backoff %.1fs", backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, MAX_BACKOFF)
                except Exception:
                    logger.exception("Event thread encountered exception, backing off %s s", backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, MAX_BACKOFF)
        _event_thread = threading.Thread(target=_worker, daemon=True)
        _event_thread.start()
        logger.info("Event thread started")

def stop_event_thread():
    _stop_flag.set()
    with _event_thread_lock:
        if _event_thread and _event_thread.is_alive():
            logger.info("Stopping event thread")
            _event_thread.join(timeout=2.0)

# ---------------------------
# Flask endpoints for keepalive/debug/admin
# ---------------------------
@app.route("/")
def home():
    return "✅ Person-model bot running. Use /debug to inspect games."

@app.route("/debug", methods=["GET"])
def debug():
    with GAME_STORE_LOCK:
        # convert keys to safe JSON
        out = {gid: {"moves": g["moves"], "white": g["white"], "black": g["black"], "last_thought": g["last_thought"], "result": g.get("result")} for gid, g in GAME_STORE.items()}
    return jsonify(out)

@app.route("/admin/reload", methods=["POST"])
def admin_reload():
    try:
        MODEL_BUNDLE.reload()
        return jsonify({"status": "reloaded", "vocab_size": len(MODEL_BUNDLE.move_list)})
    except Exception:
        logger.exception("Reload failed")
        return jsonify({"error": "reload failed"}), 500

@app.route("/admin/stats", methods=["GET"])
def admin_stats():
    try:
        # minimal stats
        with GAME_STORE_LOCK:
            active = len(GAME_STORE)
        return jsonify({
            "model_loaded": MODEL_BUNDLE.model is not None,
            "vocab_size": len(MODEL_BUNDLE.move_list),
            "active_games": active
        })
    except Exception:
        return jsonify({"error": "server error"}), 500

# ---------------------------
# Boot sequence
# ---------------------------
def boot():
    # ensure client & model bundles are loaded; start event thread
    try:
        MODEL_BUNDLE.reload()
    except Exception:
        logger.exception("Model bundle reload failed at boot (you can call /admin/reload)")

    try:
        create_client()
    except Exception:
        logger.exception("Could not create Lichess client at boot (will retry in thread)")

    start_event_thread()

if __name__ == "__main__":
    # Run bot + Flask
    boot()
    # also run a background heartbeat that restarts event thread if nothing for a while
    def heartbeat_monitor():
        while True:
            try:
                # if event thread not alive, start
                with _event_thread_lock:
                    alive = _event_thread.is_alive() if _event_thread else False
                if not alive:
                    logger.warning("Event thread not alive — restarting")
                    start_event_thread()
                time.sleep(max(5, HEARTBEAT_TIMEOUT // 6))
            except Exception:
                logger.exception("Heartbeat monitor error")
                time.sleep(5)
    t_hb = threading.Thread(target=heartbeat_monitor, daemon=True)
    t_hb.start()

    port = int(os.environ.get("PORT", 10000))
    logger.info("Starting Flask keepalive server on port %d", port)
    app.run(host="0.0.0.0", port=port, threaded=True)
