#!/usr/bin/env python3
# app.py
# IwantCheckmate — full deployment bot + dashboard
#
# Features:
#  - Loads model.h5 and vocab.npz (key 'moves')
#  - Robust shape detection for TF models (handles tuple shapes)
#  - Single persistent incoming-event stream to avoid 429s
#  - Per-game stream_game_state handlers
#  - Retry/backoff wrapper for API calls
#  - Heartbeat that restarts connection when stalled
#  - Web dashboard that displays active games, moves, and model "thoughts"
#  - Admin API for reload, stats, set params
#  - Saves PGNs to games/
#
# Env vars:
#  LICHESS_TOKEN (required)
#  MODEL_PATH (default model.h5)
#  VOCAB_PATH (default vocab.npz)
#  MODEL_URL, VOCAB_URL (optional)
#  HEALTH_SECRET (optional) - required by admin endpoints if set
#  DEFAULT_ARGMAX (true/false)
#  DEFAULT_TOPK (int)
#  DEFAULT_TEMP (float)
#  MOVE_DELAY_SECONDS
#  HEARTBEAT_TIMEOUT
#  LOGDIR
#
# Start with: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 3600
# -----------------------------------------------------------------------------

import os
import sys
import time
import json
import math
import random
import logging
import threading
import traceback
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

# third-party libs
import numpy as np
import requests
from tqdm import tqdm
import chess
import chess.pgn
import berserk
from flask import Flask, jsonify, request, render_template_string, abort

# -----------------------------------------------------------------------------
# Configuration from environment
# -----------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "model.h5")
VOCAB_PATH = os.environ.get("VOCAB_PATH", "vocab.npz")
MODEL_URL = os.environ.get("MODEL_URL", None)
VOCAB_URL = os.environ.get("VOCAB_URL", None)
LICHESS_TOKEN = os.environ.get("LICHESS_TOKEN", None)
HEALTH_SECRET = os.environ.get("HEALTH_SECRET", None)

DEFAULT_ARGMAX = os.environ.get("DEFAULT_ARGMAX", "true").lower() in ("1", "true", "yes")
DEFAULT_TOPK = int(os.environ.get("DEFAULT_TOPK", "40"))
DEFAULT_TEMP = float(os.environ.get("DEFAULT_TEMP", "1.0"))

MOVE_DELAY_SECONDS = float(os.environ.get("MOVE_DELAY_SECONDS", "0.6"))
CHALLENGE_ACCEPT_DELAY = float(os.environ.get("CHALLENGE_ACCEPT_DELAY", "0.15"))
HEARTBEAT_TIMEOUT = int(os.environ.get("HEARTBEAT_TIMEOUT", "180"))  # seconds without events -> restart
LOG_DIR = os.environ.get("LOGDIR", "games")

# Ensure directories
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("iwcm")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
logger.addHandler(ch)

# -----------------------------------------------------------------------------
# Utility: download file with progress (if MODEL_URL/VOCAB_URL provided)
# -----------------------------------------------------------------------------
def download_file(url: str, out_path: str, timeout: int = 300):
    if not url:
        raise ValueError("No url provided")
    if os.path.exists(out_path):
        logger.info("download_file: %s exists, skipping", out_path)
        return
    logger.info("Downloading %s -> %s", url, out_path)
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        if total == 0:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        else:
            with tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(out_path)) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    os.replace(tmp, out_path)
    logger.info("Downloaded %s (%d bytes)", out_path, os.path.getsize(out_path))

# -----------------------------------------------------------------------------
# Model / Vocab bundle (reloadable)
# -----------------------------------------------------------------------------
@dataclass
class ModelBundle:
    model_path: str = MODEL_PATH
    vocab_path: str = VOCAB_PATH
    model: Optional[Any] = None
    move_list: List[str] = field(default_factory=list)
    move_to_idx: Dict[str, int] = field(default_factory=dict)
    input_dim: int = 768  # default flatten 8x8x12

    def maybe_download(self):
        if MODEL_URL and not os.path.exists(self.model_path):
            try:
                download_file(MODEL_URL, self.model_path)
            except Exception:
                logger.exception("Failed to download model")
        if VOCAB_URL and not os.path.exists(self.vocab_path):
            try:
                download_file(VOCAB_URL, self.vocab_path)
            except Exception:
                logger.exception("Failed to download vocab")

    def load_vocab(self):
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"Vocab missing: {self.vocab_path}")
        logger.info("Loading vocab from %s", self.vocab_path)
        data = np.load(self.vocab_path, allow_pickle=True)
        logger.info("Vocab keys: %s", list(data.files))
        # per inspection earlier, key 'moves' exists in your file
        if "moves" in data.files:
            moves_arr = data["moves"]
            # ensure strings
            move_list = [str(x) for x in moves_arr]
        else:
            # fallback to first key
            keys = list(data.files)
            if not keys:
                raise ValueError("Empty vocab.npz")
            logger.warning("Using first key '%s' from vocab.npz", keys[0])
            moves_arr = data[keys[0]]
            move_list = [str(x) for x in moves_arr]
        self.move_list = move_list
        self.move_to_idx = {m: i for i, m in enumerate(self.move_list)}
        logger.info("Loaded vocab size %d", len(self.move_list))

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model missing: {self.model_path}")
        logger.info("Loading Keras model from %s", self.model_path)
        # local import to reduce cold-start time if TF not needed
        from tensorflow.keras.models import load_model
        self.model = load_model(self.model_path)
        # robustly infer input_dim
        try:
            # model.inputs[0].shape may be a tuple or TensorShape
            inp0 = self.model.inputs[0].shape
            if hasattr(inp0, "as_list"):
                dims = inp0.as_list()[1:]
            else:
                # tuple case
                dims = list(inp0)[1:]
            dim = 1
            for d in dims:
                dim *= (d or 1)
            self.input_dim = int(dim)
            logger.info("Inferred model input dim %d (shape dims=%s)", self.input_dim, dims)
        except Exception:
            logger.exception("Failed to infer input dim, using default %d", self.input_dim)

    def reload(self):
        logger.info("Reloading model bundle")
        self.maybe_download()
        self.load_vocab()
        self.load_model()
        logger.info("Reload complete")

# single global bundle
MODEL_BUNDLE = ModelBundle()
try:
    MODEL_BUNDLE.reload()
except Exception:
    logger.exception("Initial model load failed; call /admin/reload after fixing files")

# -----------------------------------------------------------------------------
# Board encoding utilities (must match your training encoding)
# 8x8x12 binary planes mapping pieces to planes.
# -----------------------------------------------------------------------------
PIECE_TO_PLANE = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
}

def fen_to_planes(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    arr = np.zeros((8, 8, 12), dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        r = chess.square_rank(sq)
        c = chess.square_file(sq)
        plane = PIECE_TO_PLANE.get(piece.symbol())
        if plane is not None:
            arr[r, c, plane] = 1
    return arr

def fen_to_flat(fen: str) -> np.ndarray:
    planes = fen_to_planes(fen)
    flat = planes.reshape(-1).astype(np.float32)
    if flat.size != MODEL_BUNDLE.input_dim:
        logger.warning("Flattened input size %d != model expected %d; padding/truncating", flat.size, MODEL_BUNDLE.input_dim)
        if flat.size < MODEL_BUNDLE.input_dim:
            pad = np.zeros(MODEL_BUNDLE.input_dim - flat.size, dtype=np.float32)
            flat = np.concatenate([flat, pad])
        else:
            flat = flat[:MODEL_BUNDLE.input_dim]
    return flat.reshape(1, -1)

# -----------------------------------------------------------------------------
# Model inference helpers and move selection
# -----------------------------------------------------------------------------
def predict_prob_vector(fen: str) -> np.ndarray:
    if MODEL_BUNDLE.model is None:
        raise RuntimeError("Model not loaded")
    x = fen_to_flat(fen)
    preds = MODEL_BUNDLE.model.predict(x, verbose=0)[0]
    return np.array(preds, dtype=np.float64)

def softmax_temp(x: np.ndarray, temp: float) -> np.ndarray:
    if temp <= 0:
        temp = 1e-6
    a = (x.astype(np.float64) / temp)
    a -= np.max(a)
    e = np.exp(a)
    return e / (e.sum() + 1e-12)

def choose_move_from_probs(board: chess.Board, probs: np.ndarray, argmax: bool, top_k: int, temp: float) -> Optional[str]:
    legal = [m.uci() for m in board.legal_moves]
    if not legal:
        return None
    legal_pairs = []
    for u in legal:
        idx = MODEL_BUNDLE.move_to_idx.get(u)
        if idx is not None and 0 <= idx < len(probs):
            legal_pairs.append((u, float(probs[idx])))
    if not legal_pairs:
        # no legal move in vocab -> random legal
        return random.choice(legal)
    legal_pairs.sort(key=lambda x: -x[1])
    ucis = [p[0] for p in legal_pairs]
    vals = np.array([p[1] for p in legal_pairs], dtype=np.float64)
    if argmax:
        return ucis[0]
    k = min(top_k, len(vals))
    top_vals = vals[:k]
    if top_vals.sum() <= 0:
        sample_probs = np.ones_like(top_vals) / len(top_vals)
    else:
        sample_probs = softmax_temp(top_vals, temp)
    choice = np.random.choice(range(k), p=sample_probs)
    return ucis[choice]

# -----------------------------------------------------------------------------
# PGN save helper
# -----------------------------------------------------------------------------
def save_pgn(game_id: str, white: str, black: str, moves: List[str], result: Optional[str], outdir: str = LOG_DIR) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{game_id}.pgn")
    game = chess.pgn.Game()
    game.headers["Event"] = "Lichess Bot Game"
    game.headers["White"] = white
    game.headers["Black"] = black
    game.headers["Result"] = result if result else "*"
    node = game
    for uci in moves:
        try:
            mv = chess.Move.from_uci(uci)
            node = node.add_variation(mv)
        except Exception:
            logger.debug("Skipping invalid move while saving PGN: %s", uci)
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(game))
    logger.info("Saved PGN %s", path)
    return path

# -----------------------------------------------------------------------------
# Safe request wrapper for Lichess (retry + exponential backoff for 429)
# -----------------------------------------------------------------------------
def retry_request(func, *args, max_retries: int = 6, initial_wait: float = 1.0, **kwargs):
    wait = initial_wait
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.HTTPError as he:
            code = getattr(he.response, "status_code", None)
            if code == 429:
                logger.warning("HTTP 429 rate limit; sleeping %.1f s (attempt %d/%d)", wait, attempt+1, max_retries)
                time.sleep(wait)
                wait *= 2
                continue
            else:
                logger.exception("HTTP error (not 429)")
                raise
        except Exception as e:
            logger.warning("Request exception: %s; retrying after %.1f s (attempt %d/%d)", e, wait, attempt+1, max_retries)
            time.sleep(wait)
            wait *= 2
    raise RuntimeError("Max retries exceeded for request")

# -----------------------------------------------------------------------------
# Bot state store: tracking active games, moves, and model thoughts for the dashboard
# -----------------------------------------------------------------------------
@dataclass
class GameState:
    game_id: str
    white: str = ""
    black: str = ""
    moves: List[str] = field(default_factory=list)
    result: Optional[str] = None
    last_update_ts: float = field(default_factory=time.time)
    thoughts: Dict[str, Any] = field(default_factory=dict)  # store last probs/top moves etc.

# central in-memory store
GAME_STORE: Dict[str, GameState] = {}
GAME_STORE_LOCK = threading.Lock()

def update_game_state(game_id: str, **kwargs):
    with GAME_STORE_LOCK:
        gs = GAME_STORE.get(game_id)
        if not gs:
            gs = GameState(game_id=game_id)
            GAME_STORE[game_id] = gs
        for k, v in kwargs.items():
            setattr(gs, k, v)
        gs.last_update_ts = time.time()
        return gs

def remove_game_state(game_id: str):
    with GAME_STORE_LOCK:
        if game_id in GAME_STORE:
            del GAME_STORE[game_id]

# -----------------------------------------------------------------------------
# Lichess bot core class
# -----------------------------------------------------------------------------
class LichessBot:
    def __init__(self, token: str, argmax: bool = DEFAULT_ARGMAX, top_k: int = DEFAULT_TOPK, temp: float = DEFAULT_TEMP):
        self.token = token
        self.client = None
        self.argmax = argmax
        self.top_k = top_k
        self.temp = temp
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.active_threads: Dict[str, threading.Thread] = {}
        self._last_event_ts: float = time.time()
        self._account_info: Optional[dict] = None

    def connect(self):
        if not self.token:
            raise RuntimeError("LICHESS_TOKEN not set")
        logger.info("Connecting to Lichess API")
        session = berserk.TokenSession(self.token)
        # allow longer default timeout on the client internals
        # berserk.Client does not take explicit timeout param, but requests.Session inside can be tuned if desired
        client = berserk.Client(session=session)
        # test account
        try:
            me = retry_request(client.account.get)
            self._account_info = me
            logger.info("Connected as %s (id=%s)", me.get("username"), me.get("id"))
        except Exception:
            logger.exception("Failed to fetch account info")
            raise
        self.client = client
        return me

    def get_my_id(self) -> Optional[str]:
        if self._account_info:
            return self._account_info.get("id")
        try:
            me = retry_request(self.client.account.get)
            self._account_info = me
            return me.get("id")
        except Exception:
            return None

    def accept_challenge(self, chal_id: str):
        try:
            retry_request(self.client.bots.accept_challenge, chal_id)
            logger.info("Accepted challenge %s", chal_id)
        except Exception:
            logger.exception("accept_challenge failed for %s", chal_id)

    def decline_challenge(self, chal_id: str):
        try:
            retry_request(self.client.bots.decline_challenge, chal_id)
            logger.info("Declined challenge %s", chal_id)
        except Exception:
            logger.exception("decline_challenge failed for %s", chal_id)

    def make_move(self, game_id: str, move_uci: str):
        try:
            retry_request(self.client.bots.make_move, game_id, move_uci)
            logger.info("[%s] Played move %s", game_id, move_uci)
        except Exception:
            logger.exception("[%s] make_move failed for %s", game_id, move_uci)
            raise

    def _process_incoming_event(self, event: dict):
        etype = event.get("type")
        if etype == "challenge":
            chal = event.get("challenge", {})
            variant = chal.get("variant", {}).get("key")
            challenger = chal.get("challenger", {}).get("name", "<unknown>")
            chal_id = chal.get("id")
            logger.info("Challenge from %s variant=%s id=%s", challenger, variant, chal_id)
            if variant in ("standard", "fromPosition"):
                # brief delay to avoid rapid accept spam
                time.sleep(CHALLENGE_ACCEPT_DELAY)
                try:
                    self.accept_challenge(chal_id)
                except Exception:
                    logger.exception("Accept failed; trying to decline")
                    try:
                        self.decline_challenge(chal_id)
                    except Exception:
                        logger.exception("Decline failed too")
            else:
                try:
                    self.decline_challenge(chal_id)
                except Exception:
                    logger.exception("Decline unsupported variant failed")
        elif etype == "gameStart":
            game_id = event.get("game", {}).get("id")
            logger.info("Game started: %s", game_id)
            th = threading.Thread(target=self._game_thread, args=(game_id,), daemon=True)
            th.start()
            with self._lock:
                self.active_threads[game_id] = th
        else:
            logger.debug("Unhandled incoming event type: %s", etype)

    def start_incoming_event_stream(self):
        if self.client is None:
            self.connect()
        logger.info("Starting incoming events stream")
        # keep the stream open; reconnect with backoff on exceptions
        backoff = 1.0
        while self.running:
            try:
                for event in self.client.bots.stream_incoming_events():
                    # update heartbeat timestamp
                    self._last_event_ts = time.time()
                    try:
                        self._process_incoming_event(event)
                    except Exception:
                        logger.exception("Error processing incoming event")
                # stream ended—reconnect after a small pause
                logger.warning("Incoming events stream closed; reconnecting after short sleep")
                time.sleep(2.0)
                backoff = 1.0
            except requests.exceptions.HTTPError as he:
                code = getattr(he.response, "status_code", None)
                logger.exception("HTTPError in incoming stream: %s", code)
                # if 429, wait longer
                if code == 429:
                    logger.warning("Got HTTP 429 incoming stream; sleeping longer (%ds)", int(backoff))
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 300.0)
                else:
                    time.sleep(min(backoff, 30.0))
                    backoff = min(backoff * 2.0, 300.0)
            except Exception:
                logger.exception("Exception in incoming event stream; reconnecting with delay")
                time.sleep(min(backoff, 30.0))
                backoff = min(backoff * 2.0, 300.0)

    def start(self):
        if self.running:
            logger.info("Bot already running")
            return
        self.running = True
        def _run():
            while self.running:
                try:
                    self.connect()
                    self.start_incoming_event_stream()
                except Exception:
                    logger.exception("Exception in bot main loop; sleeping before retry")
                    time.sleep(5.0)
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        logger.info("Bot main thread started")

    def stop(self):
        logger.info("Stopping bot")
        self.running = False
        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
        except Exception:
            pass

    def _game_thread(self, game_id: str):
        """
        Handles a single game's stream_game_state loop.
        Updates GAME_STORE with moves & thoughts so the dashboard can show them.
        """
        logger.info("[%s] Starting game handler thread", game_id)
        try:
            for state in self.client.bots.stream_game_state(game_id):
                self._last_event_ts = time.time()
                try:
                    moves_str = state.get("moves", "").strip()
                    moves = moves_str.split() if moves_str else []
                    board = chess.Board()
                    for mv in moves:
                        try:
                            board.push_uci(mv)
                        except Exception:
                            logger.debug("[%s] skipping invalid move when reconstructing board: %s", game_id, mv)
                    white_name = state.get("white", {}).get("name") or state.get("white", {}).get("id")
                    black_name = state.get("black", {}).get("name") or state.get("black", {}).get("id")
                    update_game_state(game_id, white=white_name or "white", black=black_name or "black", moves=moves)
                    # if finished -> save pgn and break
                    status = state.get("status")
                    if status in ("mate", "resign", "timeout", "draw", "outoftime", "stalemate"):
                        winner = state.get("winner")
                        result = "1-0" if winner == "white" else ("0-1" if winner == "black" else "1/2-1/2")
                        save_pgn(game_id, white_name or "white", black_name or "black", moves, result)
                        update_game_state(game_id, result=result)
                        logger.info("[%s] Game ended status=%s result=%s", game_id, status, result)
                        break
                    # Determine if it's our turn
                    my_id = self.get_my_id()
                    if my_id is None:
                        logger.warning("[%s] Could not determine my_id; continuing", game_id)
                        time.sleep(0.5)
                        continue
                    # White to move? board.turn == True means white to move
                    if board.turn:
                        is_my_turn = (state.get("white", {}).get("id") == my_id)
                    else:
                        is_my_turn = (state.get("black", {}).get("id") == my_id)
                    if is_my_turn:
                        # Model inference and play
                        try:
                            probs = predict_prob_vector(board.fen())
                        except Exception:
                            logger.exception("[%s] Model predict failed", game_id)
                            # fallback random
                            fallback = random.choice([m.uci() for m in board.legal_moves])
                            try:
                                self.make_move(game_id, fallback)
                                update_game_state(game_id, moves=moves + [fallback])
                                logger.info("[%s] fallback random played %s", game_id, fallback)
                            except Exception:
                                logger.exception("[%s] fallback move failed", game_id)
                            time.sleep(MOVE_DELAY_SECONDS)
                            continue
                        # compute top thoughts for dashboard
                        topk = min(12, len(probs))
                        top_idx = np.argsort(probs)[::-1][:topk]
                        top_moves = [MODEL_BUNDLE.move_list[i] if i < len(MODEL_BUNDLE.move_list) else None for i in top_idx]
                        top_scores = [float(probs[i]) for i in top_idx]
                        thoughts = {"top_moves": top_moves, "top_scores": top_scores, "timestamp": time.time()}
                        update_game_state(game_id, thoughts=thoughts)
                        # choose move
                        chosen = choose_move_from_probs(board, probs, argmax=self.argmax, top_k=self.top_k, temp=self.temp)
                        if chosen is None:
                            logger.warning("[%s] No chosen move from model; selecting random legal", game_id)
                            chosen = random.choice([m.uci() for m in board.legal_moves])
                        # Ensure legal
                        if chosen not in [m.uci() for m in board.legal_moves]:
                            logger.warning("[%s] Chosen move %s not legal; picking random legal", game_id, chosen)
                            chosen = random.choice([m.uci() for m in board.legal_moves])
                        # Play move
                        try:
                            self.make_move(game_id, chosen)
                            moves_after = moves + [chosen]
                            update_game_state(game_id, moves=moves_after)
                        except Exception:
                            logger.exception("[%s] Failed to make chosen move %s", game_id, chosen)
                        # slight delay
                        time.sleep(MOVE_DELAY_SECONDS)
                except Exception:
                    logger.exception("[%s] Inner game loop exception", game_id)
        except Exception:
            logger.exception("[%s] stream_game_state exception", game_id)
        finally:
            logger.info("[%s] Game handler thread terminating", game_id)
            # cleanup store (keep PGNs though)
            # remove_game_state(game_id)  # optional: retain for dashboard
            with self._lock:
                if game_id in self.active_threads:
                    del self.active_threads[game_id]

# -----------------------------------------------------------------------------
# Heartbeat: monitor the bot main loop and restart connection if no events
# -----------------------------------------------------------------------------
def heartbeat_monitor(bot: LichessBot, timeout: int = HEARTBEAT_TIMEOUT):
    logger.info("Starting heartbeat monitor with timeout %d s", timeout)
    while True:
        try:
            last = getattr(bot, "_last_event_ts", None)
            if last is None:
                last = time.time()
            elapsed = time.time() - last
            if elapsed > timeout:
                logger.warning("No events for %d s (> %d). Restarting bot connection.", int(elapsed), timeout)
                try:
                    bot.stop()
                    time.sleep(1.0)
                except Exception:
                    pass
                try:
                    bot.start()
                except Exception:
                    logger.exception("Failed to restart bot")
            time.sleep(max(5, timeout // 6))
        except Exception:
            logger.exception("Heartbeat monitor exception")
            time.sleep(5)

# -----------------------------------------------------------------------------
# Flask app & dashboard
# -----------------------------------------------------------------------------
app = Flask(__name__)

# Simple template for dashboard (polls JSON endpoints every 2s)
DASHBOARD_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>IwantCheckmate Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; background: #0f1720; color: #e6eef6; }
    .container { width: 95%; margin: 16px auto; }
    .card { background: #111827; border-radius: 8px; padding: 12px; margin-bottom: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.5); }
    h1 { font-size: 20px; margin: 0 0 8px 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }
    pre { background: #0b1220; padding: 8px; border-radius: 6px; overflow-x: auto; }
    .moves { font-family: monospace; white-space: pre-wrap; }
    .thought { font-family: monospace; }
    .muted { color: #9aa8b2; }
    .btn { background:#0ea5a3; color:#022b2a; padding:8px 10px; border-radius:6px; text-decoration:none; margin-right:8px; }
    .small { font-size:12px; color:#9aa8b2; }
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>IwantCheckmate — Dashboard</h1>
      <div class="small">Bot status: <span id="status">loading...</span> — <a class="btn" href="/" target="_blank">Home</a></div>
    </div>

    <div id="games"></div>

    <div class="card">
      <h2>Last Events (debug)</h2>
      <pre id="events" class="muted">Starting...</pre>
    </div>
  </div>

<script>
async function fetchJSON(url){
  try{
    let r = await fetch(url);
    if(!r.ok) return null;
    return await r.json();
  }catch(e){
    return null;
  }
}

function renderGameCard(g){
  const id = g.game_id;
  const moves = g.moves.join(' ');
  const result = g.result || 'ongoing';
  const name = `${g.white} vs ${g.black}`;
  const thoughts = g.thoughts || {};
  const top_moves = thoughts.top_moves || [];
  const top_scores = thoughts.top_scores || [];
  let thoughtHtml = '';
  for(let i=0;i<top_moves.length;i++){
    thoughtHtml += `${i+1}. ${top_moves[i] || '-'} — ${ (top_scores[i]||0).toFixed(5) } \n`;
  }
  return `
    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <b>${name}</b><div class="small">id: ${id}</div>
        </div>
        <div class="small">result: ${result}</div>
      </div>
      <hr/>
      <div><b>Moves:</b><pre class="moves">${moves || '(none yet)'}</pre></div>
      <div><b>Model thoughts (top ${top_moves.length}):</b>
        <pre class="thought">${thoughtHtml}</pre>
      </div>
    </div>
  `;
}

async function refresh(){
  const status = await fetchJSON('/api/status');
  document.getElementById('status').innerText = status ? (status.model_loaded ? 'OK' : 'model missing') : 'offline';
  const games = await fetchJSON('/api/games');
  const ev = await fetchJSON('/api/events');
  let html = '';
  if(games && games.length){
    for(let g of games){
      html += renderGameCard(g);
    }
  } else {
    html += `<div class="card"><div class="small">No active games</div></div>`;
  }
  document.getElementById('games').innerHTML = html;
  document.getElementById('events').innerText = ev ? ev.join("\\n") : "No events yet";
}

setInterval(refresh, 2000);
refresh();
</script>
</body>
</html>
"""

# Simple landing
@app.route("/", methods=["GET"])
def home():
    return "<h3>IwantCheckmate Bot — deployed. Visit /dashboard for live info.</h3>"

@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template_string(DASHBOARD_HTML)

# API: status, games, per-game, events, thoughts
@app.route("/api/status", methods=["GET"])
def api_status():
    try:
        return jsonify({
            "model_loaded": MODEL_BUNDLE.model is not None,
            "vocab_size": len(MODEL_BUNDLE.move_list),
            "argmax": BOT.argmax if BOT else DEFAULT_ARGMAX,
            "top_k": BOT.top_k if BOT else DEFAULT_TOPK,
            "temp": BOT.temp if BOT else DEFAULT_TEMP,
        })
    except Exception:
        return jsonify({"error": "server error"}), 500

@app.route("/api/games", methods=["GET"])
def api_games():
    with GAME_STORE_LOCK:
        out = []
        for gid, gs in GAME_STORE.items():
            out.append({
                "game_id": gs.game_id,
                "white": gs.white,
                "black": gs.black,
                "moves": gs.moves,
                "result": gs.result,
                "last_update_ts": gs.last_update_ts,
            })
    # sort by last update desc
    out.sort(key=lambda x: -x["last_update_ts"])
    return jsonify(out)

@app.route("/api/game/<game_id>", methods=["GET"])
def api_game(game_id):
    with GAME_STORE_LOCK:
        gs = GAME_STORE.get(game_id)
        if not gs:
            return jsonify({"error": "not found"}), 404
        return jsonify({
            "game_id": gs.game_id,
            "white": gs.white,
            "black": gs.black,
            "moves": gs.moves,
            "result": gs.result,
            "last_update_ts": gs.last_update_ts,
            "thoughts": gs.thoughts
        })

# simple events debug (keeps last N events)
EVENT_BUFFER: List[str] = []
EVENT_BUFFER_LOCK = threading.Lock()
def push_event_debug(msg: str):
    with EVENT_BUFFER_LOCK:
        EVENT_BUFFER.insert(0, f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}")
        if len(EVENT_BUFFER) > 200:
            EVENT_BUFFER.pop()

@app.route("/api/events", methods=["GET"])
def api_events():
    with EVENT_BUFFER_LOCK:
        return jsonify(EVENT_BUFFER[:50])

@app.route("/api/thoughts/<game_id>", methods=["GET"])
def api_thoughts(game_id):
    with GAME_STORE_LOCK:
        gs = GAME_STORE.get(game_id)
        if not gs:
            return jsonify({"error": "not found"}), 404
        return jsonify(gs.thoughts or {})

# Admin endpoints
def _check_secret(payload: dict) -> bool:
    if not HEALTH_SECRET:
        return True
    if not payload:
        return False
    return payload.get("secret") == HEALTH_SECRET

@app.route("/admin/reload", methods=["POST"])
def admin_reload():
    try:
        payload = request.get_json(force=True)
        if not _check_secret(payload):
            return jsonify({"error": "bad secret"}), 403
        MODEL_BUNDLE.reload()
        return jsonify({"status": "reloaded", "vocab_size": len(MODEL_BUNDLE.move_list)})
    except Exception:
        logger.exception("admin reload failed")
        return jsonify({"error": "reload failed"}), 500

@app.route("/admin/stats", methods=["GET"])
def admin_stats():
    try:
        stats = {
            "model_loaded": MODEL_BUNDLE.model is not None,
            "vocab_size": len(MODEL_BUNDLE.move_list),
            "active_games": list(BOT.active_threads.keys()) if BOT else [],
            "argmax": BOT.argmax if BOT else DEFAULT_ARGMAX,
            "top_k": BOT.top_k if BOT else DEFAULT_TOPK,
            "temp": BOT.temp if BOT else DEFAULT_TEMP,
        }
        return jsonify(stats)
    except Exception:
        logger.exception("admin stats failed")
        return jsonify({"error": "server error"}), 500

@app.route("/admin/set_params", methods=["POST"])
def admin_set_params():
    try:
        payload = request.get_json(force=True)
        if not _check_secret(payload):
            return jsonify({"error": "bad secret"}), 403
        if BOT is None:
            return jsonify({"error": "bot not running"}), 400
        if "argmax" in payload:
            BOT.argmax = bool(payload["argmax"])
        if "top_k" in payload:
            BOT.top_k = int(payload["top_k"])
        if "temp" in payload:
            BOT.temp = float(payload["temp"])
        return jsonify({"status": "ok", "argmax": BOT.argmax, "top_k": BOT.top_k, "temp": BOT.temp})
    except Exception:
        logger.exception("admin set_params failed")
        return jsonify({"error": "server error"}), 500

# Prediction endpoint for local testing
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        payload = request.get_json(force=True)
        fen = payload.get("fen")
        if not fen:
            return jsonify({"error": "missing fen"}), 400
        argmax = payload.get("argmax", DEFAULT_ARGMAX)
        top_k = int(payload.get("top_k", DEFAULT_TOPK))
        temp = float(payload.get("temp", DEFAULT_TEMP))
        try:
            board = chess.Board(fen)
        except Exception as e:
            return jsonify({"error": f"invalid fen: {e}"}), 400
        probs = predict_prob_vector(fen)
        topn = min(50, len(probs))
        idxs = np.argsort(probs)[::-1][:topn]
        top_moves = [MODEL_BUNDLE.move_list[i] if i < len(MODEL_BUNDLE.move_list) else None for i in idxs]
        top_scores = [float(probs[i]) for i in idxs]
        chosen = choose_move_from_probs(board, probs, argmax=argmax, top_k=top_k, temp=temp)
        return jsonify({
            "chosen_move": chosen,
            "top_moves": top_moves,
            "top_scores": top_scores
        })
    except Exception:
        logger.exception("predict failed")
        return jsonify({"error": "server error"}), 500

# -----------------------------------------------------------------------------
# Instantiate and start the bot and heartbeat monitor
# -----------------------------------------------------------------------------
BOT: Optional[LichessBot] = None
try:
    if not LICHESS_TOKEN:
        logger.warning("LICHESS_TOKEN not provided. Bot will not start until you set this env var.")
    else:
        BOT = LichessBot(token=LICHESS_TOKEN)
        BOT.start()
        push_event_debug("Bot started and connecting")
        # heartbeat thread
        hb = threading.Thread(target=heartbeat_monitor, args=(BOT,), daemon=True)
        hb.start()
except Exception:
    logger.exception("Failed to instantiate/start BOT")

# Hook into event pushes for debugging (small wrapper)
_orig_process = None
if BOT is not None:
    # wrap _process_incoming_event to push debug messages
    try:
        _orig_process = BOT._process_incoming_event
        def _wrapped_proc(ev):
            try:
                etype = ev.get("type")
                if etype == "challenge":
                    chal = ev.get("challenge", {})
                    challenger = chal.get("challenger", {}).get("name", "<unknown>")
                    push_event_debug(f"Challenge from {challenger} variant={chal.get('variant', {}).get('key')}")
                elif etype == "gameStart":
                    gid = ev.get("game", {}).get("id")
                    push_event_debug(f"GameStart {gid}")
                else:
                    push_event_debug(f"Event {etype}")
            except Exception:
                pass
            return _orig_process(ev)
        BOT._process_incoming_event = _wrapped_proc
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Local-run support
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IwantCheckmate Bot (app.py)")
    parser.add_argument("--no-start", action="store_true", help="Don't auto-start bot (for debugging)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Local port")
    args = parser.parse_args()
    if args.no_start:
        logger.info("Starting Flask server only (bot not started)")
    else:
        if BOT is None and LICHESS_TOKEN:
            BOT = LichessBot(token=LICHESS_TOKEN)
            BOT.start()
    logger.info("Starting Flask app")
    app.run(host="0.0.0.0", port=args.port, threaded=True)
