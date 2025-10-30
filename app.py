#!/usr/bin/env python3
"""
app.py - IwantCheckmate Lichess Bot (deployment-ready)

Expect:
  - model.h5  (Keras model expecting flattened 8*8*12 = 768 input)
  - vocab.npz (contains key 'moves' -> np.array of UCI move strings)

Environment variables:
  LICHESS_TOKEN   - required bot token (keep secret)
  MODEL_PATH      - optional (default: model.h5)
  VOCAB_PATH      - optional (default: vocab.npz)
  MODEL_URL       - optional remote URL to download model if absent
  VOCAB_URL       - optional remote URL to download vocab if absent
  HEALTH_SECRET   - optional secret needed for admin endpoints
  DEFAULT_ARGMAX  - "true"/"false" (default true)
  DEFAULT_TOPK    - integer (default 40)
  DEFAULT_TEMP    - float (default 1.0)
  LOGDIR          - where to save PGNs (default "games")
  MOVE_DELAY_SECONDS - seconds to sleep after making a move (default 0.6)
  HEARTBEAT_TIMEOUT  - seconds before restarting stream if no events (default 180)
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
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import requests
from tqdm import tqdm
import chess
import chess.pgn
import berserk
from flask import Flask, jsonify, request

# --- Configuration from environment ---
MODEL_PATH = os.environ.get("MODEL_PATH", "model.h5")
VOCAB_PATH = os.environ.get("VOCAB_PATH", "vocab.npz")
MODEL_URL = os.environ.get("MODEL_URL", None)
VOCAB_URL = os.environ.get("VOCAB_URL", None)
LICHESS_TOKEN = os.environ.get("LICHESS_TOKEN", None)
HEALTH_SECRET = os.environ.get("HEALTH_SECRET", None)
LOG_DIR = os.environ.get("LOGDIR", "games")
DEFAULT_ARGMAX = os.environ.get("DEFAULT_ARGMAX", "true").lower() in ("1", "true", "yes")
DEFAULT_TOPK = int(os.environ.get("DEFAULT_TOPK", "40"))
DEFAULT_TEMP = float(os.environ.get("DEFAULT_TEMP", "1.0"))
MOVE_DELAY_SECONDS = float(os.environ.get("MOVE_DELAY_SECONDS", "0.6"))
HEARTBEAT_TIMEOUT = int(os.environ.get("HEARTBEAT_TIMEOUT", "180"))  # seconds without event -> reconnect

# Ensure log dir exists
os.makedirs(LOG_DIR, exist_ok=True)

# --- Logging ---
logger = logging.getLogger("iwcm-bot")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
logger.addHandler(ch)

# --- Utility: download file with progress ---
def download_file(url: str, out_path: str, timeout: int = 120):
    if not url:
        raise ValueError("Empty URL")
    if os.path.exists(out_path):
        logger.info("download_file: %s exists, skipping download", out_path)
        return
    logger.info("Downloading %s -> %s", url, out_path)
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        if total == 0:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        else:
            with tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(out_path)) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    os.replace(tmp, out_path)
    logger.info("Downloaded %s (%d bytes)", out_path, os.path.getsize(out_path))

# --- Model bundle (reloadable) ---
@dataclass
class ModelBundle:
    model_path: str = MODEL_PATH
    vocab_path: str = VOCAB_PATH
    model: Optional[Any] = None
    move_list: List[str] = field(default_factory=list)
    move_to_idx: Dict[str, int] = field(default_factory=dict)
    input_dim: int = 768  # default flattened 8*8*12

    def maybe_download(self):
        if MODEL_URL and not os.path.exists(self.model_path):
            try:
                download_file(MODEL_URL, self.model_path)
            except Exception:
                logger.exception("Failed to download model from MODEL_URL")
        if VOCAB_URL and not os.path.exists(self.vocab_path):
            try:
                download_file(VOCAB_URL, self.vocab_path)
            except Exception:
                logger.exception("Failed to download vocab from VOCAB_URL")

    def load_vocab(self):
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"Vocab file missing: {self.vocab_path}")
        logger.info("Loading vocab from %s", self.vocab_path)
        data = np.load(self.vocab_path, allow_pickle=True)
        logger.info("Vocab keys: %s", data.files)
        # Your uploaded vocab uses key 'moves' per inspection
        if "moves" in data.files:
            moves_arr = data["moves"]
            move_list = [str(x) for x in moves_arr]
        else:
            # fallback: take the first array present
            keys = list(data.files)
            if not keys:
                raise ValueError("Empty vocab.npz")
            moves_arr = data[keys[0]]
            move_list = [str(x) for x in moves_arr]
            logger.warning("Using first key '%s' from vocab.npz", keys[0])
        self.move_list = move_list
        self.move_to_idx = {m: i for i, m in enumerate(self.move_list)}
        logger.info("Loaded vocab: %d moves", len(self.move_list))

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file missing: {self.model_path}")
        logger.info("Loading Keras model from %s", self.model_path)
        from tensorflow.keras.models import load_model
        self.model = load_model(self.model_path)
        # infer input size if possible
        try:
            if hasattr(self.model, "inputs") and self.model.inputs:
                shp = tuple(self.model.inputs[0].shape.as_list()[1:])
                dim = 1
                for d in shp:
                    dim *= (d or 1)
                self.input_dim = dim
                logger.info("Inferred model input dims: %s -> flattened %d", shp, dim)
            elif hasattr(self.model, "input_shape"):
                shp = tuple(self.model.input_shape[1:])
                dim = 1
                for d in shp:
                    dim *= (d or 1)
                self.input_dim = dim
                logger.info("Inferred model input_shape: %s -> flattened %d", shp, dim)
            else:
                logger.info("Using default input_dim %d", self.input_dim)
        except Exception:
            logger.exception("Failed to infer model input dim; using default %d", self.input_dim)

    def reload(self):
        logger.info("Reloading model bundle from disk/URLs")
        self.maybe_download()
        self.load_vocab()
        self.load_model()
        logger.info("Reload complete")

# global bundle
MODEL_BUNDLE = ModelBundle()
try:
    MODEL_BUNDLE.reload()
except Exception:
    logger.exception("Initial model/vocab load failed - call /admin/reload later")

# --- Board encoding (must match training) ---
# Map piece to plane index (same mapping as used to train)
PIECE_TO_PLANE = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
}

def fen_to_planes(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    arr = np.zeros((8, 8, 12), dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        r = chess.square_rank(sq)   # 0..7
        c = chess.square_file(sq)   # 0..7
        plane = PIECE_TO_PLANE.get(piece.symbol())
        if plane is not None:
            arr[r, c, plane] = 1
    return arr

def fen_to_flat(fen: str) -> np.ndarray:
    arr = fen_to_planes(fen)
    flat = arr.reshape(-1).astype(np.float32)
    if flat.size != MODEL_BUNDLE.input_dim:
        # if model expects different size, pad/truncate conservatively
        logger.warning("Input flat size %d != model input_dim %d; adjusting", flat.size, MODEL_BUNDLE.input_dim)
        if flat.size < MODEL_BUNDLE.input_dim:
            pad = np.zeros(MODEL_BUNDLE.input_dim - flat.size, dtype=np.float32)
            flat = np.concatenate([flat, pad])
        else:
            flat = flat[:MODEL_BUNDLE.input_dim]
    return flat.reshape(1, -1)

# --- Model inference & move picker ---
def model_predict_probs(fen: str) -> np.ndarray:
    if MODEL_BUNDLE.model is None:
        raise RuntimeError("Model not loaded")
    x = fen_to_flat(fen)
    preds = MODEL_BUNDLE.model.predict(x, verbose=0)[0]
    return np.array(preds, dtype=np.float64)

def softmax_with_temp(x: np.ndarray, temp: float) -> np.ndarray:
    if temp <= 0:
        temp = 1e-6
    x = x / float(temp)
    x = x - np.max(x)
    e = np.exp(x)
    p = e / (e.sum() + 1e-12)
    return p

def choose_legal_move_from_probs(board: chess.Board, probs: np.ndarray,
                                 argmax: bool = True, top_k: int = 40, temp: float = 1.0) -> Optional[str]:
    legal_moves = [m.uci() for m in board.legal_moves]
    if not legal_moves:
        return None
    legal_idx_probs = []
    for u in legal_moves:
        idx = MODEL_BUNDLE.move_to_idx.get(u)
        if idx is not None and 0 <= idx < len(probs):
            legal_idx_probs.append((u, float(probs[idx])))
    if not legal_idx_probs:
        # no legal move in vocab -> random legal
        return random.choice(legal_moves)
    legal_idx_probs.sort(key=lambda x: -x[1])
    ucis = [p[0] for p in legal_idx_probs]
    vals = np.array([p[1] for p in legal_idx_probs], dtype=np.float64)
    if argmax:
        return ucis[0]
    k = min(top_k, len(vals))
    top_vals = vals[:k]
    if top_vals.sum() <= 0:
        probs_sample = np.ones_like(top_vals) / len(top_vals)
    else:
        probs_sample = softmax_with_temp(top_vals, temp)
    choice = np.random.choice(range(k), p=probs_sample)
    return ucis[choice]

# --- PGN saving ---
def save_game_pgn(game_id: str, white: str, black: str, moves: List[str], result: Optional[str], out_dir: str = LOG_DIR):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{game_id}.pgn")
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
            logger.debug("Skipping invalid move while saving pgn: %s", uci)
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(game))
    logger.info("Saved PGN to %s", path)
    return path

# --- Safe Lichess client wrapper with retries/backoff ---
def retry_request(func, *args, max_retries: int = 6, initial_wait: float = 1.0, **kwargs):
    wait = initial_wait
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code == 429:
                logger.warning("HTTP 429 rate limit on attempt %d; sleeping %s", attempt + 1, wait)
                time.sleep(wait)
                wait *= 2
                continue
            else:
                logger.exception("HTTP error in Lichess request")
                raise
        except Exception as e:
            logger.warning("Request error attempt %d: %s. Retrying in %s s", attempt + 1, e, wait)
            time.sleep(wait)
            wait *= 2
    raise RuntimeError("Max retries exceeded for request")

# --- Lichess Bot class ---
class LichessBot:
    def __init__(self, token: Optional[str], argmax: bool = DEFAULT_ARGMAX, top_k: int = DEFAULT_TOPK, temp: float = DEFAULT_TEMP):
        self.token = token
        self.client = None
        self.argmax = argmax
        self.top_k = top_k
        self.temp = temp
        self.running = False
        self._thread = None
        self.active_games: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._last_event_ts = time.time()
        self._account_cache = None

    def connect(self):
        if not self.token:
            raise RuntimeError("LICHESS_TOKEN not set")
        logger.info("Connecting to Lichess API")
        session = berserk.TokenSession(self.token)
        client = berserk.Client(session=session)
        try:
            me = retry_request(client.account.get)
            self._account_cache = me
            logger.info("Connected as %s (id=%s)", me.get("username"), me.get("id"))
        except Exception:
            logger.exception("Failed to fetch account info")
            raise
        self.client = client
        return me

    def get_my_id(self):
        if self._account_cache:
            return self._account_cache.get("id")
        try:
            me = retry_request(self.client.account.get)
            self._account_cache = me
            return me.get("id")
        except Exception:
            return None

    def accept_challenge(self, chal_id: str):
        try:
            retry_request(self.client.bots.accept_challenge, chal_id)
            logger.info("Accepted challenge %s", chal_id)
        except Exception:
            logger.exception("Accept challenge failed %s", chal_id)

    def decline_challenge(self, chal_id: str):
        try:
            retry_request(self.client.bots.decline_challenge, chal_id)
            logger.info("Declined challenge %s", chal_id)
        except Exception:
            logger.exception("Decline challenge failed %s", chal_id)

    def handle_incoming_events(self):
        if self.client is None:
            self.connect()
        logger.info("Starting incoming events stream")
        while self.running:
            try:
                # stream incoming events (long-lived)
                for event in self.client.bots.stream_incoming_events():
                    self._last_event_ts = time.time()
                    try:
                        self._process_event(event)
                    except Exception:
                        logger.exception("Error while processing incoming event")
                # if stream stops naturally, loop and reconnect
                logger.warning("Incoming events stream closed; reconnecting...")
            except requests.exceptions.HTTPError as e:
                code = getattr(e.response, "status_code", None)
                logger.exception("Stream incoming events HTTP error: %s", code)
                time.sleep(5)
            except Exception:
                logger.exception("Exception in incoming event stream; reconnecting in 5s")
                time.sleep(5)

    def _process_event(self, event: dict):
        etype = event.get("type")
        if etype == "challenge":
            chal = event.get("challenge", {})
            variant = chal.get("variant", {}).get("key")
            challenger = chal.get("challenger", {}).get("name", "<unknown>")
            cid = chal.get("id")
            logger.info("Incoming challenge from %s variant=%s id=%s", challenger, variant, cid)
            # accept standard and fromPosition, decline others
            if variant in ("standard", "fromPosition"):
                # small sleep to avoid rapid-fire accepts
                time.sleep(0.1)
                try:
                    self.accept_challenge(cid)
                except Exception:
                    logger.exception("Accept challenge exception, trying decline")
                    try:
                        self.decline_challenge(cid)
                    except Exception:
                        logger.exception("Decline also failed")
            else:
                try:
                    self.decline_challenge(cid)
                except Exception:
                    logger.exception("Decline unsupported variant failed")
        elif etype == "gameStart":
            game_id = event["game"]["id"]
            logger.info("Game started: %s", game_id)
            t = threading.Thread(target=self._game_handler_thread, args=(game_id,))
            t.daemon = True
            t.start()
            with self._lock:
                self.active_games[game_id] = t
        else:
            logger.debug("Unhandled event type: %s", etype)

    def _game_handler_thread(self, game_id: str):
        logger.info("[%s] Launching game handler", game_id)
        try:
            for state in self.client.bots.stream_game_state(game_id):
                try:
                    # update last_event timestamp
                    self._last_event_ts = time.time()
                    moves_token = state.get("moves", "").strip()
                    moves = moves_token.split() if moves_token else []
                    board = chess.Board()
                    for mv in moves:
                        try:
                            board.push_uci(mv)
                        except Exception:
                            logger.debug("[%s] ignoring illegal recorded move: %s", game_id, mv)
                    # update names (may be None)
                    white_name = state.get("white", {}).get("name") or state.get("white", {}).get("id")
                    black_name = state.get("black", {}).get("name") or state.get("black", {}).get("id")
                    status = state.get("status")
                    if status in ("mate", "resign", "timeout", "draw", "outoftime", "stalemate"):
                        winner = state.get("winner")
                        result_str = "1-0" if winner == "white" else ("0-1" if winner == "black" else "1/2-1/2")
                        save_game_pgn(game_id, white_name or "white", black_name or "black", moves, result_str)
                        logger.info("[%s] Game finished status=%s result=%s", game_id, status, result_str)
                        break
                    # check turn: True if white to move
                    my_id = self.get_my_id()
                    if my_id is None:
                        logger.warning("[%s] Could not determine bot id; skipping", game_id)
                        time.sleep(1.0)
                        continue
                    # Determine if it's our turn
                    if board.turn:
                        # white to move
                        is_my_turn = (state.get("white", {}).get("id") == my_id)
                    else:
                        is_my_turn = (state.get("black", {}).get("id") == my_id)
                    if is_my_turn:
                        logger.debug("[%s] It's our turn. Predicting...", game_id)
                        try:
                            probs = model_predict_probs(board.fen())
                        except Exception:
                            logger.exception("[%s] Model inference failed, falling back random", game_id)
                            fallback = random.choice([m.uci() for m in board.legal_moves])
                            try:
                                retry_request(self.client.bots.make_move, game_id, fallback)
                                logger.info("[%s] fallback random move %s", game_id, fallback)
                            except Exception:
                                logger.exception("[%s] fallback make_move failed", game_id)
                            time.sleep(MOVE_DELAY_SECONDS)
                            continue
                        chosen = choose_legal_move_from_probs(board, probs, argmax=self.argmax, top_k=self.top_k, temp=self.temp)
                        if chosen is None:
                            logger.warning("[%s] No move chosen; skipping", game_id)
                        else:
                            # ensure legality
                            if chosen not in [m.uci() for m in board.legal_moves]:
                                logger.warning("[%s] chosen move %s not legal; selecting random legal", game_id, chosen)
                                chosen = random.choice([m.uci() for m in board.legal_moves])
                            try:
                                retry_request(self.client.bots.make_move, game_id, chosen)
                                logger.info("[%s] Played %s", game_id, chosen)
                            except Exception:
                                logger.exception("[%s] make_move failed for %s", game_id, chosen)
                        time.sleep(MOVE_DELAY_SECONDS)
                except Exception:
                    logger.exception("[%s] inner game loop error", game_id)
        except Exception:
            logger.exception("[%s] stream_game_state ended with exception", game_id)
        finally:
            with self._lock:
                try:
                    if game_id in self.active_games:
                        del self.active_games[game_id]
                except Exception:
                    pass
            logger.info("[%s] game handler terminated", game_id)

    def start(self):
        if self.running:
            logger.info("Bot already running")
            return
        self.running = True
        def _run():
            while self.running:
                try:
                    self.connect()
                    self.handle_incoming_events()
                except Exception:
                    logger.exception("Exception in main event loop; will retry in 5s")
                    time.sleep(5)
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        logger.info("Bot main thread started")

    def stop(self):
        self.running = False
        logger.info("Stopping bot")
        try:
            if self._thread:
                self._thread.join(timeout=2.0)
        except Exception:
            pass

# --- Heartbeat monitor to restart stream if no events for a while ---
def heartbeat_monitor(bot: LichessBot, timeout: int = HEARTBEAT_TIMEOUT):
    logger.info("Starting heartbeat monitor (timeout %d s)", timeout)
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
                    logger.exception("Failed to restart bot main thread")
            time.sleep(max(5, timeout // 6))
        except Exception:
            logger.exception("heartbeat monitor exception; sleeping briefly")
            time.sleep(5)

# --- Flask app & admin endpoints ---
flask_app = Flask(__name__)

@flask_app.route("/", methods=["GET"])
def index():
    try:
        return jsonify({
            "status": "ok",
            "model_loaded": MODEL_BUNDLE.model is not None,
            "vocab_size": len(MODEL_BUNDLE.move_list),
            "active_games": list(BOT.active_games.keys()) if BOT else []
        })
    except Exception:
        return jsonify({"status": "error"}), 500

@flask_app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json(force=True)
        fen = data.get("fen")
        if not fen:
            return jsonify({"error": "missing fen"}), 400
        argmax = data.get("argmax", BOT.argmax if BOT else DEFAULT_ARGMAX)
        top_k = int(data.get("top_k", BOT.top_k if BOT else DEFAULT_TOPK))
        temp = float(data.get("temp", BOT.temp if BOT else DEFAULT_TEMP))
        # validate fen
        try:
            board = chess.Board(fen)
        except Exception as e:
            return jsonify({"error": f"invalid fen: {e}"}), 400
        probs = model_predict_probs(fen)
        topn = min(50, len(probs))
        idxs = np.argsort(probs)[::-1][:topn]
        top_moves = [MODEL_BUNDLE.move_list[i] if i < len(MODEL_BUNDLE.move_list) else None for i in idxs]
        top_scores = [float(probs[i]) for i in idxs]
        chosen = choose_legal_move_from_probs(board, probs, argmax=argmax, top_k=top_k, temp=temp)
        return jsonify({
            "chosen_move": chosen,
            "argmax": bool(argmax),
            "top_k": top_k,
            "temp": temp,
            "top_moves": top_moves,
            "top_scores": top_scores
        })
    except Exception:
        logger.exception("Error in /predict")
        return jsonify({"error": "server error"}), 500

def _check_secret(payload: Dict[str, Any]) -> bool:
    if not HEALTH_SECRET:
        return True
    if not payload:
        return False
    return payload.get("secret") == HEALTH_SECRET

@flask_app.route("/admin/reload", methods=["POST"])
def admin_reload():
    try:
        payload = request.get_json(force=True)
        if not _check_secret(payload):
            return jsonify({"error": "bad secret"}), 403
        MODEL_BUNDLE.reload()
        return jsonify({"status": "reloaded", "vocab_size": len(MODEL_BUNDLE.move_list)})
    except Exception:
        logger.exception("Reload failed")
        return jsonify({"error": "reload failed"}), 500

@flask_app.route("/admin/stats", methods=["GET"])
def admin_stats():
    try:
        return jsonify({
            "model_loaded": MODEL_BUNDLE.model is not None,
            "vocab_size": len(MODEL_BUNDLE.move_list),
            "active_games_count": len(BOT.active_games) if BOT else 0,
            "argmax": BOT.argmax if BOT else DEFAULT_ARGMAX,
            "top_k": BOT.top_k if BOT else DEFAULT_TOPK,
            "temp": BOT.temp if BOT else DEFAULT_TEMP
        })
    except Exception:
        logger.exception("admin/stats failed")
        return jsonify({"error": "server error"}), 500

@flask_app.route("/admin/set_params", methods=["POST"])
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
        logger.exception("admin/set_params failed")
        return jsonify({"error": "server error"}), 500

# --- Expose module-level app for gunicorn ---
app = flask_app

# --- Instantiate and start bot and heartbeat ---
BOT: Optional[LichessBot] = None
try:
    BOT = LichessBot(token=LICHESS_TOKEN)
    # start bot
    BOT.start()
    # start heartbeat monitor thread
    hb = threading.Thread(target=heartbeat_monitor, args=(BOT,), daemon=True)
    hb.start()
except Exception:
    logger.exception("Failed to create/start BOT - use /admin/reload to recover")

# --- Local run support ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--no-start", action="store_true", help="Don't auto-start the bot thread")
    args = parser.parse_args()
    if args.no_start:
        logger.info("Starting Flask only (bot not started)")
    else:
        if BOT is None:
            BOT = LichessBot(token=LICHESS_TOKEN)
            BOT.start()
    logger.info("Starting Flask server")
    app.run(host="0.0.0.0", port=args.port, threaded=True)
