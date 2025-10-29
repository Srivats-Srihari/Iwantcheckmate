#!/usr/bin/env python3
# app.py
# -----------------------------------------------------------------------------
# Lichess TF Bot — Deployment-ready script
#
# - Expects model file: model.h5
# - Expects vocab file: vocab.npz (contains array 'moves' or similar)
# - Optional env vars:
#     LICHESS_TOKEN   - your bot token (required for playing)
#     MODEL_URL       - HTTP(S) URL to download model.h5 if not committed
#     VOCAB_URL       - HTTP(S) URL to download vocab.npz if not committed
#     MODEL_PATH      - path to local model file (defaults model.h5)
#     VOCAB_PATH      - path to local vocab file (defaults vocab.npz)
#     HEALTH_SECRET   - optional secret required for admin endpoints
#     DEFAULT_ARGMAX  - "true"/"false" default argmax behaviour
#     DEFAULT_TOPK    - integer top-k default for sampling
#     DEFAULT_TEMP    - float temperature default
#     LOGDIR          - where to save PGNs
#
# - Exposes Flask app as module-level `app` for gunicorn app:app
# - Robust streaming of lichess events and per-game handler
# -----------------------------------------------------------------------------

import os
import sys
import time
import json
import math
import random
import shutil
import logging
import threading
import argparse
import traceback
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

# third-party libs
import numpy as np
import chess
import chess.pgn
import berserk
from flask import Flask, jsonify, request, abort
from tensorflow.keras.models import load_model
import requests
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration (environment-driven)
# -----------------------------------------------------------------------------
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
CHALLENGE_ACCEPT_DELAY = float(os.environ.get("CHALLENGE_ACCEPT_DELAY", "0.2"))

# Create log directory
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("iwcm_bot")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
logger.addHandler(ch)

# -----------------------------------------------------------------------------
# Convenience: download file (with progress) if remote URL is provided
# -----------------------------------------------------------------------------
def download_file(url: str, out_path: str, chunk_size: int = 8192, timeout: int = 120):
    """
    Download a file from url to out_path. Skip if file exists.
    """
    if not url:
        raise ValueError("No URL provided to download_file")
    if os.path.exists(out_path):
        logger.info("download_file: %s exists; skipping", out_path)
        return
    logger.info("Downloading %s -> %s", url, out_path)
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        if total == 0:
            # unknown size
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        else:
            with tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(out_path)) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    os.replace(tmp, out_path)
    logger.info("Downloaded %s (size=%d)", out_path, os.path.getsize(out_path))

# -----------------------------------------------------------------------------
# Model / Vocab loader bundle
# -----------------------------------------------------------------------------
@dataclass
class ModelBundle:
    model_path: str = MODEL_PATH
    vocab_path: str = VOCAB_PATH
    model: Optional[Any] = None
    move_list: List[str] = field(default_factory=list)
    move_to_idx: Dict[str, int] = field(default_factory=dict)
    input_shape: Tuple[int, ...] = (768,)  # default flattened 8*8*12

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
            raise FileNotFoundError(f"vocab file not found: {self.vocab_path}")
        logger.info("Loading vocab from %s", self.vocab_path)
        data = np.load(self.vocab_path, allow_pickle=True)
        # Expect an array `moves` or similar; try keys
        if "moves" in data:
            moves = list(data["moves"])
            logger.info("Loaded 'moves' key with %d entries", len(moves))
        else:
            # use first array available
            files = list(data.files)
            if not files:
                raise ValueError("vocab.npz contains no arrays")
            moves = list(data[files[0]])
            logger.info("Loaded first key '%s' with %d entries", files[0], len(moves))
        self.move_list = moves
        self.move_to_idx = {m: i for i, m in enumerate(moves)}
        logger.info("Vocab loaded, size=%d", len(self.move_list))

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"model file not found: {self.model_path}")
        logger.info("Loading model from %s", self.model_path)
        self.model = load_model(self.model_path)
        # attempt to determine input size
        try:
            if hasattr(self.model, "inputs") and self.model.inputs:
                shape = tuple(self.model.inputs[0].shape.as_list()[1:])
                # flatten
                dim = 1
                for d in shape:
                    dim *= (d or 1)
                self.input_shape = (dim,)
                logger.info("Inferred model input shape: %s -> flattened=%d", shape, dim)
            elif hasattr(self.model, "input_shape"):
                shape = tuple(self.model.input_shape[1:])
                dim = 1
                for d in shape:
                    dim *= (d or 1)
                self.input_shape = (dim,)
                logger.info("Inferred model input shape: %s -> flattened=%d", shape, dim)
            else:
                logger.info("Could not infer input shape; defaulting to %s", self.input_shape)
        except Exception:
            logger.exception("Error inferring model input shape; using default")

    def reload(self):
        logger.info("Reloading model bundle...")
        self.maybe_download()
        self.load_vocab()
        self.load_model()
        logger.info("Reload complete.")

# instantiate bundle
MODEL_BUNDLE = ModelBundle()

# Try initial load but don't crash entire process; log instead.
try:
    MODEL_BUNDLE.reload()
except Exception as e:
    logger.exception("Initial model/vocab load failed: %s", e)

# -----------------------------------------------------------------------------
# Board encoding utilities (must match training)
# -----------------------------------------------------------------------------
PIECE_TO_PLANE = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
}

def fen_to_planes(fen: str) -> np.ndarray:
    """
    Convert FEN to 8x8x12 binary planes (dtype uint8).
    Uses chess.Board for parsing.
    Orientation used: rank 0 is chess.square_rank 0 (board bottom)
    This matches training if you trained with the same function.
    """
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
    return flat.reshape(1, -1)  # shape (1, 768)

# -----------------------------------------------------------------------------
# Model inference helpers
# -----------------------------------------------------------------------------
def predict_prob_vector(fen: str) -> np.ndarray:
    if MODEL_BUNDLE.model is None:
        raise RuntimeError("Model is not loaded")
    x = fen_to_flat(fen)
    preds = MODEL_BUNDLE.model.predict(x, verbose=0)[0]
    return np.array(preds, dtype=np.float64)

def softmax_temp(arr: np.ndarray, temp: float = 1.0) -> np.ndarray:
    if temp <= 0:
        temp = 1e-6
    a = arr.astype(np.float64) / temp
    a = a - np.max(a)
    e = np.exp(a)
    return e / (e.sum() + 1e-12)

def choose_move_from_probs(board: chess.Board, probs: np.ndarray,
                           argmax: bool = True, top_k: int = 40, temp: float = 1.0) -> Optional[str]:
    """
    Choose a legal move for `board` guided by full-vocab `probs`.
    If no legal moves are present in the vocab, pick a random legal move.
    """
    legal = [m.uci() for m in board.legal_moves]
    if not legal:
        return None
    legal_pairs = []
    for u in legal:
        idx = MODEL_BUNDLE.move_to_idx.get(u)
        if idx is not None and 0 <= idx < len(probs):
            legal_pairs.append((u, float(probs[idx])))
    if not legal_pairs:
        # fallback: random legal move
        return random.choice(legal)
    legal_pairs.sort(key=lambda x: -x[1])
    ucis = [p[0] for p in legal_pairs]
    vals = np.array([p[1] for p in legal_pairs], dtype=np.float64)
    if argmax:
        return ucis[0]
    k = min(top_k, len(vals))
    top_vals = vals[:k]
    if top_vals.sum() <= 0:
        probs_sample = np.ones_like(top_vals) / len(top_vals)
    else:
        probs_sample = softmax_temp(top_vals, temp)
    choice = np.random.choice(range(k), p=probs_sample)
    return ucis[choice]

# -----------------------------------------------------------------------------
# PGN logging
# -----------------------------------------------------------------------------
def save_pgn(game_id: str, white_name: str, black_name: str, moves: List[str], result: Optional[str], outdir: str = LOG_DIR) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{game_id}.pgn")
    game = chess.pgn.Game()
    game.headers["Event"] = "Lichess Bot Game"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Result"] = result if result else "*"
    node = game
    board = chess.Board()
    for uci in moves:
        try:
            mv = chess.Move.from_uci(uci)
            node = node.add_variation(mv)
            board.push(mv)
        except Exception:
            # skip invalid during saving
            pass
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(game))
    logger.info("Saved PGN to %s", path)
    return path

# -----------------------------------------------------------------------------
# Lichess Bot Class
# -----------------------------------------------------------------------------
class LichessBot:
    def __init__(self, token: Optional[str] = None, argmax: bool = DEFAULT_ARGMAX, top_k: int = DEFAULT_TOPK, temp: float = DEFAULT_TEMP):
        self.token = token or LICHESS_TOKEN
        self.client = None
        self.argmax = argmax
        self.top_k = top_k
        self.temp = temp
        self.running = False
        self._thread = None
        self.active_games: Dict[str, threading.Thread] = {}
        self._account_cache: Optional[dict] = None
        self._lock = threading.Lock()
        self._last_event_time = time.time()

    def connect(self):
        if not self.token:
            raise RuntimeError("No LICHESS token provided")
        logger.info("Connecting to Lichess API")
        session = berserk.TokenSession(self.token)
        client = berserk.Client(session=session)
        # test account
        me = client.account.get()
        logger.info("Connected to Lichess as %s (id=%s)", me.get("username"), me.get("id"))
        self.client = client
        self._account_cache = me
        return me

    def get_my_id(self):
        if self._account_cache:
            return self._account_cache.get("id")
        try:
            me = self.client.account.get()
            self._account_cache = me
            return me.get("id")
        except Exception:
            return None

    def accept_challenge(self, chal_id: str):
        try:
            self.client.bots.accept_challenge(chal_id)
            logger.info("Accepted challenge %s", chal_id)
        except Exception:
            logger.exception("Failed to accept challenge %s", chal_id)

    def decline_challenge(self, chal_id: str):
        try:
            self.client.bots.decline_challenge(chal_id)
            logger.info("Declined challenge %s", chal_id)
        except Exception:
            logger.exception("Failed to decline challenge %s", chal_id)

    def handle_event(self, event: dict):
        t = event.get("type")
        if t == "challenge":
            chal = event["challenge"]
            variant = chal.get("variant", {}).get("key")
            challenger = chal.get("challenger", {}).get("name")
            chal_id = chal.get("id")
            logger.info("Challenge from %s variant=%s id=%s", challenger, variant, chal_id)
            # Accept standard and fromPosition variant; else decline
            if variant in ("standard", "fromPosition"):
                try:
                    time.sleep(CHALLENGE_ACCEPT_DELAY)
                    self.accept_challenge(chal_id)
                except Exception:
                    # fallback: decline
                    try:
                        self.decline_challenge(chal_id)
                    except Exception:
                        logger.exception("Failed both accept and decline for %s", chal_id)
            else:
                try:
                    self.decline_challenge(chal_id)
                except Exception:
                    logger.exception("Failed to decline unsupported variant chal %s", chal_id)

        elif t == "gameStart":
            game_id = event["game"]["id"]
            logger.info("Game started: %s", game_id)
            th = threading.Thread(target=self.game_loop, args=(game_id,))
            th.daemon = True
            th.start()
            with self._lock:
                self.active_games[game_id] = th
        else:
            logger.debug("Unhandled incoming event: %s", t)

    def stream_events_forever(self):
        if self.client is None:
            self.connect()
        logger.info("Starting incoming events stream")
        try:
            for event in self.client.bots.stream_incoming_events():
                self._last_event_time = time.time()
                try:
                    self.handle_event(event)
                except Exception:
                    logger.exception("Error handling incoming event")
        except Exception:
            logger.exception("Stream incoming events terminated unexpectedly")
            raise

    def run(self):
        # Keep reconnecting if stream breaks
        self.running = True
        while self.running:
            try:
                if self.client is None:
                    self.connect()
                self.stream_events_forever()
            except Exception:
                logger.exception("Bot main loop exception; reconnecting in 5s")
                time.sleep(5)

    def start(self):
        if self._thread and self._thread.is_alive():
            logger.info("Bot already running")
            return
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        logger.info("Bot thread started")

    def stop(self):
        self.running = False
        logger.info("Stopping bot")
        if self._thread:
            self._thread.join(timeout=2.0)

    def game_loop(self, game_id: str):
        """
        Stream game state for a specific game and play moves when it's our turn.
        Uses client.bots.stream_game_state(game_id)
        """
        logger.info("[%s] Starting game loop", game_id)
        moves_list: List[str] = []
        white_name = black_name = None
        try:
            for state in self.client.bots.stream_game_state(game_id):
                # State may contain 'moves' string
                moves_token = state.get("moves", "").strip()
                moves_list = moves_token.split() if moves_token else []
                # Build board
                board = chess.Board()
                for mv in moves_list:
                    try:
                        board.push_uci(mv)
                    except Exception:
                        # ignore illegal pushes when reconstructing board for safety
                        logger.debug("[%s] invalid move in moves_list: %s", game_id, mv)

                # Update player names
                white_name = state.get("white", {}).get("name") or state.get("white", {}).get("id")
                black_name = state.get("black", {}).get("name") or state.get("black", {}).get("id")

                # Check for finished game
                status = state.get("status")
                if status in ("mate", "resign", "timeout", "draw", "outoftime", "stalemate"):
                    winner = state.get("winner")
                    result_str = "1-0" if winner == "white" else ("0-1" if winner == "black" else "1/2-1/2")
                    save_pgn(game_id, white_name or "white", black_name or "black", moves_list, result_str)
                    logger.info("[%s] Game ended status=%s result=%s", game_id, status, result_str)
                    break

                # Determine if it's our turn
                # Obtain my id
                my_id = self.get_my_id()
                if my_id is None:
                    # try updating account cache
                    try:
                        me = self.client.account.get()
                        my_id = me.get("id")
                        self._account_cache = me
                    except Exception:
                        logger.warning("[%s] Could not get my account id; skipping move detection", game_id)
                        time.sleep(1.0)
                        continue

                # Determine whose turn by board.turn boolean
                is_my_turn = (board.turn and state.get("white", {}).get("id") == my_id) or (not board.turn and state.get("black", {}).get("id") == my_id)

                if is_my_turn:
                    # It's our turn: predict and play
                    try:
                        probs = predict_prob_vector(board.fen())
                    except Exception as e:
                        logger.exception("[%s] Model prediction failed: %s", game_id, e)
                        # fallback: random legal
                        fallback = random.choice([m.uci() for m in board.legal_moves])
                        try:
                            self.client.bots.make_move(game_id, fallback)
                            logger.info("[%s] Fallback random move played: %s", game_id, fallback)
                        except Exception:
                            logger.exception("[%s] Fallback make_move failed", game_id)
                        time.sleep(MOVE_DELAY_SECONDS)
                        continue

                    chosen = choose_move_from_probs(board, probs, argmax=self.argmax, top_k=self.top_k, temp=self.temp)
                    if chosen is None:
                        logger.warning("[%s] No chosen move; skipping", game_id)
                    else:
                        # sanity: ensure legality
                        if chosen not in [m.uci() for m in board.legal_moves]:
                            logger.warning("[%s] Chosen move %s not legal; picking random legal", game_id, chosen)
                            chosen = random.choice([m.uci() for m in board.legal_moves])
                        try:
                            self.client.bots.make_move(game_id, chosen)
                            logger.info("[%s] Played move %s", game_id, chosen)
                        except Exception:
                            logger.exception("[%s] make_move failed for %s", game_id, chosen)
                    # slight delay to avoid potential rate limit issues
                    time.sleep(MOVE_DELAY_SECONDS)
                # continue streaming until end
        except Exception:
            logger.exception("[%s] Exception in game loop", game_id)
        finally:
            with self._lock:
                try:
                    if game_id in self.active_games:
                        del self.active_games[game_id]
                except Exception:
                    pass
            logger.info("[%s] game loop terminated", game_id)

# -----------------------------------------------------------------------------
# Flask app and endpoints
# -----------------------------------------------------------------------------
flask_app = Flask(__name__)

@flask_app.route("/", methods=["GET"])
def health():
    try:
        model_ok = MODEL_BUNDLE.model is not None
        vocab_ok = bool(MODEL_BUNDLE.move_list)
        active_count = len(BOT.active_games) if BOT is not None else 0
        return jsonify({
            "status": "ok",
            "model_loaded": model_ok,
            "vocab_size": len(MODEL_BUNDLE.move_list),
            "active_games": active_count,
            "argmax_default": DEFAULT_ARGMAX,
            "top_k_default": DEFAULT_TOPK,
            "temp_default": DEFAULT_TEMP
        })
    except Exception:
        logger.exception("health endpoint error")
        return jsonify({"status": "error"}), 500

@flask_app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    POST JSON:
      {
        "fen": "<FEN>",
        "argmax": true/false (optional),
        "top_k": int (optional),
        "temp": float (optional)
      }
    Response:
      {
        "chosen_move": "e2e4",
        "top_moves": [...],
        "top_scores": [...]
      }
    """
    try:
        payload = request.get_json(force=True)
        fen = payload.get("fen")
        if not fen:
            return jsonify({"error": "missing fen"}), 400
        argmax = payload.get("argmax", BOT.argmax if BOT else DEFAULT_ARGMAX)
        top_k = int(payload.get("top_k", BOT.top_k if BOT else DEFAULT_TOPK))
        temp = float(payload.get("temp", BOT.temp if BOT else DEFAULT_TEMP))
        # Validate fen
        try:
            board = chess.Board(fen)
        except Exception as e:
            return jsonify({"error": f"invalid fen: {e}"}), 400
        probs = predict_prob_vector(fen)
        # top-n over full vocab
        top_n = min(50, len(probs))
        idxs = np.argsort(probs)[::-1][:top_n]
        top_moves = [MODEL_BUNDLE.move_list[i] if i < len(MODEL_BUNDLE.move_list) else None for i in idxs]
        top_scores = [float(probs[i]) for i in idxs]
        chosen = choose_move_from_probs(board, probs, argmax=argmax, top_k=top_k, temp=temp)
        return jsonify({
            "chosen_move": chosen,
            "argmax": bool(argmax),
            "top_k": top_k,
            "temp": temp,
            "top_moves": top_moves,
            "top_scores": top_scores
        })
    except Exception:
        logger.exception("predict endpoint error")
        return jsonify({"error": "server error"}), 500

def _check_health_secret(payload: dict) -> bool:
    if not HEALTH_SECRET:
        return True
    if not payload:
        return False
    return payload.get("secret") == HEALTH_SECRET

@flask_app.route("/admin/reload", methods=["POST"])
def admin_reload():
    try:
        payload = request.get_json(force=True)
        if not _check_health_secret(payload):
            return jsonify({"error": "bad secret"}), 403
        MODEL_BUNDLE.reload()
        return jsonify({"status": "reloaded", "vocab_size": len(MODEL_BUNDLE.move_list)})
    except Exception:
        logger.exception("admin reload failed")
        return jsonify({"error": "reload failed"}), 500

@flask_app.route("/admin/stats", methods=["GET"])
def admin_stats():
    try:
        stats = {
            "model_loaded": MODEL_BUNDLE.model is not None,
            "vocab_size": len(MODEL_BUNDLE.move_list),
            "active_games": list(BOT.active_games.keys()) if BOT else [],
            "argmax": BOT.argmax if BOT else DEFAULT_ARGMAX,
            "top_k": BOT.top_k if BOT else DEFAULT_TOPK,
            "temp": BOT.temp if BOT else DEFAULT_TEMP
        }
        return jsonify(stats)
    except Exception:
        logger.exception("admin stats failed")
        return jsonify({"error": "server error"}), 500

@flask_app.route("/admin/set_params", methods=["POST"])
def admin_set_params():
    try:
        payload = request.get_json(force=True)
        if not _check_health_secret(payload):
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

# -----------------------------------------------------------------------------
# Module-level BOT instance and app exposure for gunicorn
# -----------------------------------------------------------------------------
BOT = None
try:
    BOT = LichessBot(token=LICHESS_TOKEN, argmax=DEFAULT_ARGMAX, top_k=DEFAULT_TOPK, temp=DEFAULT_TEMP)
    def _starter():
        # small delay to allow import to complete and Render to finish startup
        time.sleep(1.0)
        try:
            logger.info("Starting BOT from module import")
            BOT.start()
        except Exception:
            logger.exception("Failed to start BOT on import")
    threading.Thread(target=_starter, daemon=True).start()
except Exception:
    logger.exception("Failed to instantiate BOT at module import")

# Expose Flask app as `app`
app = flask_app

# -----------------------------------------------------------------------------
# CLI & local-run support
# -----------------------------------------------------------------------------
def run_local_server(port: int = 8000, host: str = "0.0.0.0"):
    logger.info("Running local Flask server on %s:%d", host, port)
    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lichess TF Bot - app.py")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to model.h5")
    parser.add_argument("--vocab", default=VOCAB_PATH, help="Path to vocab.npz")
    parser.add_argument("--token", default=os.environ.get("LICHESS_TOKEN"), help="Lichess bot token")
    parser.add_argument("--no-start", action="store_true", help="Don't auto-start bot thread (debug)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Local port")
    parser.add_argument("--argmax", action="store_true", help="Start bot with argmax mode")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="Top-k sampling")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature")
    args = parser.parse_args()

    # override bundle paths if provided
    MODEL_BUNDLE.model_path = args.model
    MODEL_BUNDLE.vocab_path = args.vocab
    try:
        MODEL_BUNDLE.reload()
    except Exception:
        logger.exception("Initial reload at CLI failed")

    # replace token if provided
    if args.token:
        if BOT:
            BOT.token = args.token
        else:
            BOT = LichessBot(token=args.token)
    if BOT:
        BOT.argmax = args.argmax or BOT.argmax
        BOT.top_k = args.topk
        BOT.temp = args.temp
        if not args.no_start:
            BOT.start()
    else:
        if not args.no_start:
            BOT = LichessBot(token=args.token)
            BOT.start()

    run_local_server(port=args.port)

# End of file
