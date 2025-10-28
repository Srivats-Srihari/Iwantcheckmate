# app.py
# -----------------------------------------------------------------------------
# Lichess TensorFlow Bot - Render-ready
#
# - Expects model file:  model.h5
# - Expects vocab file:  vocab.npz  (contains array 'moves' mapping idx -> uci)
# - Exposes Flask app object at module-level for Gunicorn: `gunicorn app:app`
# - Connects to Lichess (use LICHESS_TOKEN env var), accepts challenges,
#   supports both standard and position-based challenges (fromPosition),
#   and plays moves using the TF model.
#
# Notes:
# - Model trained on flattened 8x8x12 encoding (input shape 768).
# - If you want to use external MODEL_URL/VOCAB_URL to download artifacts
#   at boot, set those env vars in Render; the bot will download them once.
#
# Author: generated for Srivats (customized for your model/vocab)
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

# Third-party dependencies
# Ensure these are present in requirements.txt for Render: berserk, tensorflow, python-chess, Flask, gunicorn, numpy, requests, tqdm
import numpy as np
import chess
import chess.pgn
import berserk
from flask import Flask, jsonify, request, abort
from tensorflow.keras.models import load_model

# Optional: for robust download with progress
import requests
from tqdm import tqdm

# -----------------------------------------------------------------------------
# CONFIGURATION (change via env vars or edit here)
# -----------------------------------------------------------------------------
MODEL_FILENAME = os.environ.get("MODEL_PATH", "model.h5")
VOCAB_FILENAME = os.environ.get("VOCAB_PATH", "vocab.npz")
MODEL_URL = os.environ.get("MODEL_URL", None)  # optional remote URL to fetch model
VOCAB_URL = os.environ.get("VOCAB_URL", None)  # optional remote URL to fetch vocab
LICHESS_TOKEN_ENV = os.environ.get("LICHESS_TOKEN", None)  # must be set to bot token or provided at runtime
LOG_DIR = os.environ.get("LOGDIR", "games")
HEALTH_SECRET = os.environ.get("HEALTH_SECRET", None)  # optional: require secret for /admin endpoints

# Move selection defaults
DEFAULT_ARGMAX = os.environ.get("DEFAULT_ARGMAX", "true").lower() in ("1", "true", "yes")
DEFAULT_TOPK = int(os.environ.get("DEFAULT_TOPK", "40"))
DEFAULT_TEMP = float(os.environ.get("DEFAULT_TEMP", "1.0"))

# Lichess retry/delay config
MOVE_DELAY_SECONDS = float(os.environ.get("MOVE_DELAY_SECONDS", "0.6"))
CHALLENGE_ACCEPT_DELAY = float(os.environ.get("CHALLENGE_ACCEPT_DELAY", "0.1"))

# Make sure the games dir exists
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logger = logging.getLogger("iwcm-bot")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
logger.addHandler(handler)


# -----------------------------------------------------------------------------
# Utility: download file if URL provided
# -----------------------------------------------------------------------------
def download_file(url: str, out_path: str, chunk_size: int = 8192, timeout: int = 120) -> None:
    """
    Download a file from `url` to `out_path`. If out_path exists, skip download.
    Uses streaming requests with a tqdm progress bar.
    """
    if not url:
        raise ValueError("download_file: url is empty")
    if os.path.exists(out_path):
        logger.info("download_file: %s already exists -> skipping download", out_path)
        return

    logger.info("Downloading %s -> %s", url, out_path)
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        if total == 0:
            # Unknown length
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        else:
            with tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(out_path)) as pbar:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    os.rename(tmp_path, out_path)
    logger.info("Downloaded %s (%d bytes).", out_path, os.path.getsize(out_path))


# -----------------------------------------------------------------------------
# Model and vocab loader (safe reloadable)
# -----------------------------------------------------------------------------
@dataclass
class ModelBundle:
    model_path: str
    vocab_path: str
    model: Optional[Any] = None
    move_list: List[str] = field(default_factory=list)  # idx -> uci
    move_to_idx: Dict[str, int] = field(default_factory=dict)
    input_shape: Tuple[int, ...] = (768,)

    def ensure_files(self):
        """
        If MODEL_URL / VOCAB_URL env vars are set, attempt to download
        the files to the configured paths.
        """
        if MODEL_URL and not os.path.exists(self.model_path):
            try:
                download_file(MODEL_URL, self.model_path)
            except Exception as e:
                logger.exception("Failed to download model: %s", e)
        if VOCAB_URL and not os.path.exists(self.vocab_path):
            try:
                download_file(VOCAB_URL, self.vocab_path)
            except Exception as e:
                logger.exception("Failed to download vocab: %s", e)

    def load_vocab(self):
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"Vocab file not found: {self.vocab_path}")
        logger.info("Loading vocab from %s", self.vocab_path)
        data = np.load(self.vocab_path, allow_pickle=True)
        # We expect an array `moves` with UCI strings (idx->move)
        if "moves" in data:
            moves = list(data["moves"])
            logger.info("Found 'moves' in vocab: %d moves", len(moves))
        else:
            # fallback: take first key
            first_key = list(data.files)[0]
            moves = list(data[first_key])
            logger.info("No 'moves' key; using first key '%s' (%d entries)", first_key, len(moves))
        self.move_list = moves
        self.move_to_idx = {m: i for i, m in enumerate(self.move_list)}
        logger.info("Vocab loaded. Moves: %d", len(self.move_list))

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        logger.info("Loading TF model from %s", self.model_path)
        # Load via Keras; this will bring in TensorFlow
        self.model = load_model(self.model_path)
        # infer expected input shape from model if possible:
        try:
            # keras models often have input_shape attr on first layer or model._functions
            layer_input_shape = None
            if hasattr(self.model, "inputs") and self.model.inputs:
                layer_input_shape = tuple(self.model.inputs[0].shape.as_list()[1:])
            elif hasattr(self.model, "input_shape"):
                layer_input_shape = tuple(self.model.input_shape[1:])
            if layer_input_shape:
                # flatten multi-dim to single dimension if necessary
                dim = 1
                for d in layer_input_shape:
                    dim *= d
                self.input_shape = (dim,)
                logger.info("Inferred model input size: %s -> flattened %d", layer_input_shape, dim)
            else:
                logger.info("Could not infer model input shape; using default %s", self.input_shape)
        except Exception:
            logger.exception("Error inferring model input shape; using default %s", self.input_shape)

    def reload(self):
        """
        Download (if configured) and reload vocab + model.
        """
        logger.info("Reloading model bundle (model=%s, vocab=%s)", self.model_path, self.vocab_path)
        self.ensure_files()
        self.load_vocab()
        self.load_model()
        logger.info("Reload complete.")


# create global bundle
MODEL_BUNDLE = ModelBundle(model_path=MODEL_FILENAME, vocab_path=VOCAB_FILENAME)
# attempt initial load, but don't crash the import if files are missing; we log and continue
try:
    MODEL_BUNDLE.reload()
except Exception:
    logger.exception("Initial model/vocab load failed; you can call /admin/reload to retry")


# -----------------------------------------------------------------------------
# Board encoding utilities (must match training)
# - Training used 8x8x12 piece planes flattened.
# - We must replicate exactly: planes order and rank/file orientation.
# -----------------------------------------------------------------------------
# plane mapping: piece symbol -> plane index
PIECE_TO_PLANE = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
}

def fen_to_plane_array(fen: str) -> np.ndarray:
    """
    Convert FEN to 8x8x12 binary planes as used in training, shaped (8,8,12).
    Note: This must match how you encoded boards when training your model.
    We interpret individual board squares by `chess` package's square indexing:
      - chess.square_file(sq): file 0..7 (a..h)
      - chess.square_rank(sq): rank 0..7 (1..8)  -> bottom (rank 0) is first row here
    The training used rank/file orientation producing an array where arr[rank,file,plane] = 1
    """
    board = chess.Board(fen)
    arr = np.zeros((8, 8, 12), dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        # chess.square_rank: 0..7; chess.square_file: 0..7
        r = chess.square_rank(sq)
        c = chess.square_file(sq)
        plane = PIECE_TO_PLANE.get(piece.symbol())
        if plane is None:
            continue
        arr[r, c, plane] = 1
    return arr

def fen_to_flat_input(fen: str) -> np.ndarray:
    """
    Convert FEN to flattened input vector of shape (1, 768) of dtype float32.
    """
    arr = fen_to_plane_array(fen)
    flat = arr.reshape(-1).astype(np.float32)
    return flat.reshape(1, -1)  # shape (1, 768)


# -----------------------------------------------------------------------------
# Model inference / move selection helpers
# -----------------------------------------------------------------------------
def model_predict_probs(fen: str) -> np.ndarray:
    """
    Given a FEN, return model probability vector (len == vocab_size).
    If model is not loaded, raise.
    """
    if MODEL_BUNDLE.model is None:
        raise RuntimeError("Model not loaded")
    x = fen_to_flat_input(fen)
    preds = MODEL_BUNDLE.model.predict(x, verbose=0)[0]  # vector
    # ensure numeric numpy array
    return np.array(preds, dtype=np.float64)


def softmax_with_temperature(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    if temp <= 0:
        # treat 0 as argmax behavior in external code, but here we avoid division by zero
        temp = 1e-6
    x = x / float(temp)
    x = x - np.max(x)
    e = np.exp(x)
    p = e / (e.sum() + 1e-12)
    return p


def choose_legal_move_from_probs(board: chess.Board, probs: np.ndarray,
                                 argmax: bool = True, top_k: int = 40, temp: float = 1.0) -> Optional[str]:
    """
    Given a board and a full-vocab probability vector, pick a legal UCI move.
    - argmax True -> pick highest-prob legal
    - argmax False -> sample among top_k legal moves using temperature
    If no legal move is present in the vocab, return random legal move (string uci).
    """
    legal_moves = [m.uci() for m in board.legal_moves]
    if not legal_moves:
        return None

    # Map legal moves to vocabulary indices (if present)
    legal_with_probs = []
    for u in legal_moves:
        idx = MODEL_BUNDLE.move_to_idx.get(u)
        if idx is not None and idx < len(probs):
            legal_with_probs.append((u, float(probs[idx])))
        else:
            # Not in vocab -> will be considered separately
            pass

    if not legal_with_probs:
        # No legal moves in vocab: fallback to random legal
        return random.choice(legal_moves)

    # sort descending by prob
    legal_with_probs.sort(key=lambda x: -x[1])
    ucis = [p[0] for p in legal_with_probs]
    pv = np.array([p[1] for p in legal_with_probs], dtype=np.float64)

    if argmax:
        return ucis[0]

    # sampling path
    k = min(top_k, len(pv))
    top_p = pv[:k]
    if top_p.sum() <= 0:
        # uniform among top_k legal moves
        probs_sample = np.ones_like(top_p) / len(top_p)
    else:
        probs_sample = softmax_with_temperature(top_p, temp)
    choice = np.random.choice(range(k), p=probs_sample)
    return ucis[choice]


# -----------------------------------------------------------------------------
# PGN saving utility
# -----------------------------------------------------------------------------
def save_game_pgn_file(game_id: str, white_name: str, black_name: str, moves: List[str], result: str, out_dir: str = LOG_DIR) -> str:
    """
    Save a simple PGN file to out_dir/<game_id>.pgn with headers and move list.
    Returns path.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{game_id}.pgn")
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
            # skip illegal moves in logging
            pass

    with open(path, "w", encoding="utf-8") as f:
        f.write(str(game))
    logger.info("Saved PGN to %s", path)
    return path


# -----------------------------------------------------------------------------
# Lichess bot event handling and main loop
# -----------------------------------------------------------------------------
class LichessBot:
    def __init__(self, token: Optional[str] = None,
                 argmax: bool = DEFAULT_ARGMAX, top_k: int = DEFAULT_TOPK, temp: float = DEFAULT_TEMP):
        self.token = token or LICHESS_TOKEN_ENV
        self.client = None
        self.running = False
        self._thread = None
        self.argmax = argmax
        self.top_k = top_k
        self.temp = temp
        self.active_games: Dict[str, threading.Thread] = {}  # game_id -> thread

    def connect(self):
        if not self.token:
            raise RuntimeError("Lichess token not provided")
        session = berserk.TokenSession(self.token)
        self.client = berserk.Client(session=session)
        me = self.client.account.get()
        logger.info("Connected to lichess as %s (id=%s)", me.get("username"), me.get("id"))
        return me

    def accept_challenge(self, challenge_id: str):
        try:
            self.client.bots.accept_challenge(challenge_id)
            logger.info("Accepted challenge %s", challenge_id)
        except Exception as e:
            logger.exception("Failed to accept challenge %s: %s", challenge_id, e)

    def decline_challenge(self, challenge_id: str):
        try:
            self.client.bots.decline_challenge(challenge_id)
            logger.info("Declined challenge %s", challenge_id)
        except Exception as e:
            logger.exception("Failed to decline challenge %s: %s", challenge_id, e)

    def handle_incoming_events(self):
        """
        Main loop streaming incoming events. This method runs in the bot thread.
        """
        if self.client is None:
            self.connect()

        logger.info("Starting stream of incoming events...")
        try:
            for event in self.client.bots.stream_incoming_events():
                try:
                    self._process_event(event)
                except Exception as e:
                    logger.exception("Error processing incoming event: %s", e)
        except Exception as e:
            logger.exception("Exception while streaming incoming events: %s", e)
            # try reconnect after short sleep
            time.sleep(5)
            raise

    def _process_event(self, event: dict):
        etype = event.get("type")
        if etype == "challenge":
            chal = event["challenge"]
            challenger = chal.get("challenger", {}).get("name", "<unknown>")
            variant = chal.get("variant", {}).get("key")
            speed = chal.get("speed")
            logger.info("Incoming challenge from %s: variant=%s speed=%s id=%s", challenger, variant, speed, chal.get("id"))
            # Accept standard and 'fromPosition' (position) challenges, decline others
            if variant == "standard" or variant == "fromPosition":
                try:
                    # small delay to avoid rate-limit flurries
                    time.sleep(CHALLENGE_ACCEPT_DELAY)
                    self.client.bots.accept_challenge(chal["id"])
                    logger.info("Accepted challenge %s (from %s)", chal["id"], challenger)
                except Exception:
                    # If accept fails for any reason, decline
                    try:
                        self.client.bots.decline_challenge(chal["id"])
                        logger.info("Declined challenge %s after acceptance error", chal["id"])
                    except Exception:
                        logger.exception("Both accept and decline failed for challenge %s", chal["id"])
            else:
                # decline unsupported variant
                try:
                    self.client.bots.decline_challenge(chal["id"])
                    logger.info("Declined unsupported challenge variant=%s id=%s", variant, chal["id"])
                except Exception:
                    logger.exception("Failed to decline challenge %s", chal["id"])

        elif etype == "gameStart":
            game_id = event["game"]["id"]
            logger.info("Game started: %s", game_id)
            # spawn per-game handler in separate thread
            t = threading.Thread(target=self.run_game_loop, args=(game_id,))
            t.daemon = True
            t.start()
            self.active_games[game_id] = t
        else:
            logger.debug("Unhandled incoming event type: %s", etype)

    def run_game_loop(self, game_id: str):
        """
        For a started game, stream the game state and play when it is our turn.
        """
        logger.info("[%s] Starting game loop", game_id)
        moves_so_far: List[str] = []
        try:
            for state in self.client.bots.stream_game_state(game_id):
                try:
                    # state may be 'gameFull' (initial) and then 'gameState' updates
                    moves_token = state.get("moves", "").strip()
                    moves_list = moves_token.split() if moves_token else []
                    # rebuild the board
                    board = chess.Board()
                    for mv in moves_list:
                        try:
                            board.push_uci(mv)
                        except Exception:
                            logger.debug("[%s] ignore push_uci failed for mv=%s", game_id, mv)
                    # who is white/black
                    white_id = state.get("white", {}).get("id")
                    black_id = state.get("black", {}).get("id")
                    white_name = state.get("white", {}).get("name", white_id)
                    black_name = state.get("black", {}).get("name", black_id)

                    # check if game has ended
                    status = state.get("status")
                    if status in ("mate", "resign", "timeout", "draw", "outoftime", "stalemate"):
                        # save PGN and exit
                        winner = state.get("winner")
                        result_str = "1-0" if winner == "white" else ("0-1" if winner == "black" else "1/2-1/2")
                        save_game_pgn_file(game_id, white_name, black_name, moves_list, result_str, out_dir=LOG_DIR)
                        logger.info("[%s] Game ended status=%s result=%s", game_id, status, result_str)
                        break

                    # is it our turn?
                    is_my_turn = False
                    me = self.client.account.get()
                    my_id = me.get("id")
                    if board.turn:
                        is_my_turn = (white_id == my_id)
                    else:
                        is_my_turn = (black_id == my_id)

                    if is_my_turn:
                        # choose move
                        try:
                            probs = model_predict_probs(board.fen())
                        except Exception as e:
                            logger.exception("[%s] model predict failed: %s", game_id, e)
                            # fallback to random
                            chosen = random.choice([m.uci() for m in board.legal_moves])
                            try:
                                self.client.bots.make_move(game_id, chosen)
                                logger.info("[%s] Fallback played %s", game_id, chosen)
                            except Exception:
                                logger.exception("[%s] fallback make_move failed", game_id)
                            time.sleep(MOVE_DELAY_SECONDS)
                            continue

                        chosen_uci = choose_legal_move_from_probs(board, probs,
                                                                  argmax=self.argmax,
                                                                  top_k=self.top_k,
                                                                  temp=self.temp)
                        if chosen_uci is None:
                            logger.warning("[%s] No move chosen by model -> skipping", game_id)
                        else:
                            # sanity check legality again
                            if chosen_uci not in [m.uci() for m in board.legal_moves]:
                                logger.warning("[%s] Chosen move %s not legal; picking random", game_id, chosen_uci)
                                chosen_uci = random.choice([m.uci() for m in board.legal_moves])

                            try:
                                self.client.bots.make_move(game_id, chosen_uci)
                                logger.info("[%s] Played move %s", game_id, chosen_uci)
                            except Exception as e:
                                logger.exception("[%s] make_move failed for %s : %s", game_id, chosen_uci, e)
                        # avoid flooding
                        time.sleep(MOVE_DELAY_SECONDS)
                except Exception:
                    logger.exception("[%s] inner game loop error", game_id)
        except Exception:
            logger.exception("[%s] stream_game_state ended with exception", game_id)
        finally:
            # ensure we remove from active list
            try:
                if game_id in self.active_games:
                    del self.active_games[game_id]
            except Exception:
                pass
            logger.info("[%s] game loop finished", game_id)

    def start(self):
        if self.running:
            logger.info("Bot already running")
            return
        self.running = True
        self._thread = threading.Thread(target=self._run_thread)
        self._thread.daemon = True
        self._thread.start()
        logger.info("Bot thread started")

    def _run_thread(self):
        # loop and attempt to reconnect if streaming breaks
        while self.running:
            try:
                self.connect()
                self.handle_incoming_events()
            except Exception:
                logger.exception("Bot encountered exception in main loop, will retry in 5s")
                time.sleep(5)

    def stop(self):
        self.running = False
        logger.info("Stopping bot")
        try:
            if self._thread is not None:
                self._thread.join(timeout=2.0)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Flask app and endpoints (expose health, status, reload)
# -----------------------------------------------------------------------------
flask_app = Flask(__name__)

@flask_app.route("/", methods=["GET"])
def health():
    """
    Basic health endpoint. Returns status and some runtime info.
    """
    try:
        model_ok = MODEL_BUNDLE.model is not None
        vocab_ok = bool(MODEL_BUNDLE.move_list)
        active_count = len(BOT.active_games) if BOT is not None else 0
        return jsonify({
            "status": "ok",
            "model_loaded": model_ok,
            "vocab_size": len(MODEL_BUNDLE.move_list),
            "active_games": active_count,
            "argmax": BOT.argmax if BOT is not None else DEFAULT_ARGMAX,
            "top_k": BOT.top_k if BOT is not None else DEFAULT_TOPK,
            "temp": BOT.temp if BOT is not None else DEFAULT_TEMP
        })
    except Exception:
        logger.exception("Error in health endpoint")
        return jsonify({"status": "error"}), 500


@flask_app.route("/predict", methods=["POST"])
def predict_move_endpoint():
    """
    POST JSON: { "fen": "<FEN>", "argmax": true/false (optional), "top_k": int, "temp": float }
    Returns predicted UCI move and scores for top-N.
    """
    try:
        payload = request.get_json(force=True)
        fen = payload.get("fen")
        if not fen:
            return jsonify({"error": "Missing 'fen'"}), 400
        argmax = payload.get("argmax", BOT.argmax if BOT else DEFAULT_ARGMAX)
        top_k = int(payload.get("top_k", BOT.top_k if BOT else DEFAULT_TOPK))
        temp = float(payload.get("temp", BOT.temp if BOT else DEFAULT_TEMP))

        # quick board validation
        try:
            board = chess.Board(fen)
        except Exception as e:
            return jsonify({"error": f"Invalid FEN: {str(e)}"}), 400

        probs = model_predict_probs(fen)
        # compute top moves (over full vocab)
        top_indices = np.argsort(probs)[::-1][:min(50, len(probs))]
        top_moves = []
        scores = []
        for idx in top_indices:
            move = MODEL_BUNDLE.move_list[idx] if idx < len(MODEL_BUNDLE.move_list) else None
            top_moves.append(move)
            scores.append(float(probs[idx]))

        # choose legal recommended move
        chosen = choose_legal_move_from_probs(board, probs, argmax=argmax, top_k=top_k, temp=temp)
        return jsonify({
            "chosen_move": chosen,
            "argmax": argmax,
            "top_k": top_k,
            "temp": temp,
            "top_moves": top_moves,
            "top_scores": scores
        })
    except Exception:
        logger.exception("Error in /predict")
        return jsonify({"error": "server error"}), 500


@flask_app.route("/admin/reload", methods=["POST"])
def admin_reload():
    """
    Admin endpoint to reload the model and vocab. If HEALTH_SECRET is set, require it.
    POST JSON {"secret": "..."} if HEALTH_SECRET is used.
    """
    try:
        if HEALTH_SECRET:
            payload = request.get_json(force=True)
            if not payload or payload.get("secret") != HEALTH_SECRET:
                return jsonify({"error": "bad secret"}), 403

        MODEL_BUNDLE.reload()
        return jsonify({"status": "reloaded", "vocab_size": len(MODEL_BUNDLE.move_list)})
    except Exception:
        logger.exception("Reload failed")
        return jsonify({"error": "reload failed"}), 500


@flask_app.route("/admin/stats", methods=["GET"])
def admin_stats():
    """
    Return some debug stats for the running bot
    """
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
        logger.exception("admin_stats failed")
        return jsonify({"error": "server error"}), 500


@flask_app.route("/admin/set_params", methods=["POST"])
def admin_set_params():
    """
    Change sampling parameters at runtime. Body JSON examples:
    { "argmax": false, "top_k": 20, "temp": 0.8 }
    """
    try:
        payload = request.get_json(force=True)
        if HEALTH_SECRET:
            if not payload or payload.get("secret") != HEALTH_SECRET:
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
        logger.exception("admin_set_params failed")
        return jsonify({"error": "server error"}), 500


# -----------------------------------------------------------------------------
# Module-level bot instance and Flask app exposure for gunicorn
# -----------------------------------------------------------------------------
# Instantiate the bot (token may be set via env var on Render)
BOT = None
try:
    BOT = LichessBot(token=os.environ.get("LICHESS_TOKEN"), argmax=DEFAULT_ARGMAX, top_k=DEFAULT_TOPK, temp=DEFAULT_TEMP)
    # Start the bot in background once the module is imported (Gunicorn will import)
    # We start after a short delay via a thread so app import completes fast.
    def _delayed_start():
        time.sleep(1.0)
        try:
            BOT.start()
        except Exception:
            logger.exception("Failed to start BOT")
    threading.Thread(target=_delayed_start, daemon=True).start()
except Exception:
    logger.exception("Failed to create BOT instance at module import; you can create manually via /admin endpoints")


# Expose Flask app variable for gunicorn: `gunicorn app:app`
app = flask_app


# -----------------------------------------------------------------------------
# If run directly (python app.py) provide CLI to run in foreground for local testing
# -----------------------------------------------------------------------------
def run_local_server(host="0.0.0.0", port:int = None):
    host = host or "0.0.0.0"
    port = port or int(os.environ.get("PORT", 8000))
    logger.info("Starting local Flask server on %s:%d", host, port)
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lichess TF Bot (app.py)")
    parser.add_argument("--model", default=MODEL_FILENAME, help="Path to .h5 model file")
    parser.add_argument("--vocab", default=VOCAB_FILENAME, help="Path to vocab.npz")
    parser.add_argument("--token", default=None, help="Lichess bot token (or set LICHESS_TOKEN env var)")
    parser.add_argument("--no-start", action="store_true", help="Do not auto-start the bot thread (for debugging)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = parser.parse_args()

    # If user requests different model/vocab paths, reload bundle
    if args.model != MODEL_BUNDLE.model_path or args.vocab != MODEL_BUNDLE.vocab_path:
        MODEL_BUNDLE.model_path = args.model
        MODEL_BUNDLE.vocab_path = args.vocab
        try:
            MODEL_BUNDLE.reload()
        except Exception:
            logger.exception("reload failed at start")

    # Replace the running BOT instance's token if provided
    if args.token:
        if BOT:
            BOT.token = args.token
        else:
            BOT = LichessBot(token=args.token)

    if not args.no_start:
        if BOT is None:
            BOT = LichessBot(token=args.token)
        BOT.start()

    run_local_server(port=args.port)

# End of file
