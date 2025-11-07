#!/usr/bin/env python3
"""
app.py - Lichess imitation bot + simple web UI for Render

Requirements:
    pip install flask berserk python-chess tensorflow numpy

Usage:
    - Put your model (model.h5, chess_model.keras, or similar) and vocab.npz next to this file.
    - Set env var LICHESS_TOKEN to your bot token (Render: add as secret).
    - Run: python app.py  (or let Render run it)
"""

import os
import time
import json
import math
import random
import threading
import traceback
from collections import defaultdict, deque
from typing import List, Dict, Tuple

import numpy as np
from flask import Flask, jsonify, request, render_template_string
import berserk
import chess

# Try to import tensorflow/keras last (costly)
try:
    import tensorflow as tf
    from tensorflow import keras
except Exception as ex:
    keras = None
    tf = None
    print("WARNING: TensorFlow/keras not available:", ex)

# ---------------------------
# CONFIG
# ---------------------------
PORT = int(os.environ.get("PORT", 10000))
LICHESS_TOKEN = os.environ.get("LICHESS_TOKEN", None)  # must be set for lichess features
MODEL_FILENAMES = ["model.h5", "model.keras", "chess_model.keras", "model.hdf5"]
VOCAB_FILENAME = os.environ.get("VOCAB_PATH", "vocab.npz")
MAX_HISTORY_LEN = int(os.environ.get("MAX_HISTORY_LEN", 48))
TEMPERATURE = float(os.environ.get("TEMP", 1.0))   # sampling temperature
TOP_K = int(os.environ.get("TOP_K", 5))            # optional top-k filtering
EVENT_BACKOFF_BASE = 2.0                            # backoff multiplier on 429s
HEARTBEAT_TIMEOUT = 180                             # seconds without events => reconnect

# ---------------------------
# GLOBALS: model + vocab + runtime state
# ---------------------------
MODEL = None
MOVE_VOCAB = None              # numpy array of moves
MOVE_TO_IDX = {}               # uci string -> idx (int)
IDX_TO_MOVE = {}               # idx -> uci
VOCAB_SIZE = 0

# Tracking active games and recent thoughts for web UI
ACTIVE_GAMES = {}              # game_id -> dict with keys: white, black, moves(list), last_thought, result
ACTIVE_GAMES_LOCK = threading.Lock()

# Lichess client
BERSERK_CLIENT = None

# Flask app
app = Flask(__name__)

# ---------------------------
# UTIL: load model + vocab
# ---------------------------
def load_vocab(vocab_path: str):
    global MOVE_VOCAB, MOVE_TO_IDX, IDX_TO_MOVE, VOCAB_SIZE
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"vocab not found at {vocab_path}")
    data = np.load(vocab_path, allow_pickle=True)
    # Common formats: data['moves'] (array of strings), or keys may differ
    if 'moves' in data:
        moves = list(data['moves'])
    else:
        # fallback: take first array-like value
        keys = list(data.keys())
        moves = list(data[keys[0]])
    # Ensure python strings
    moves = [str(m) for m in moves]
    MOVE_VOCAB = np.array(moves, dtype=object)
    MOVE_TO_IDX = {m: i for i, m in enumerate(MOVE_VOCAB)}
    IDX_TO_MOVE = {i: m for i, m in enumerate(MOVE_VOCAB)}
    VOCAB_SIZE = len(MOVE_VOCAB)
    print(f"Loaded vocab: {VOCAB_SIZE} moves from {vocab_path}")

def load_model_try(paths: List[str]):
    """Try to load a keras model from several candidate filenames."""
    global MODEL
    for p in paths:
        if os.path.exists(p):
            print(f"Loading Keras model from {p} ...")
            MODEL = keras.models.load_model(p)
            print("Model loaded.")
            return p
    # not found
    raise FileNotFoundError(f"No model found in paths: {paths}")

# ---------------------------
# ENCODING / SAMPLING
# ---------------------------
def encode_history(history: List[str]) -> np.ndarray:
    """Turn move-history (list of UCI strings) into padded index vector shape (1, MAX_HISTORY_LEN)"""
    idxs = [MOVE_TO_IDX.get(m, -1) for m in history[-MAX_HISTORY_LEN:]]
    # replace missing moves with 0 (padding); we used 0 as 'unknown/pad' in training (if that was the case)
    # if your training used 1-based indexing change appropriately.
    idxs = [ (i if i >= 0 else -1) for i in idxs ]
    # we'll map unknown moves to -1 -> convert to 0 index for model (assumes model handled 0)
    idxs = [ (i if i >= 0 else 0) for i in idxs ]
    if len(idxs) < MAX_HISTORY_LEN:
        idxs = [0] * (MAX_HISTORY_LEN - len(idxs)) + idxs
    arr = np.array(idxs, dtype=np.int32).reshape(1, -1)
    return arr

def sample_from_probs(probs: np.ndarray, legal_moves: List[chess.Move], legal_scores: np.ndarray):
    """Given predicted probs over VOCAB, restrict to legal moves and sample according to temperature & top-k"""
    # legal_scores is the slice of probs for each legal move index
    # apply temperature
    if TEMPERATURE <= 0.0:
        # deterministic argmax
        choice_idx = int(np.argmax(legal_scores))
    else:
        # temp sampling
        logits = np.log(legal_scores + 1e-12) / TEMPERATURE
        exps = np.exp(logits - np.max(logits))
        p = exps / np.sum(exps)
        # optionally top-k: zero out all but top_k
        if TOP_K and TOP_K < len(p):
            topk_idx = np.argsort(p)[-TOP_K:]
            mask = np.zeros_like(p)
            mask[topk_idx] = 1.0
            p = p * mask
            if p.sum() <= 0:
                p = np.ones_like(p) / p.size
            else:
                p = p / p.sum()
        choice_idx = int(np.random.choice(len(legal_moves), p=p))
    return legal_moves[choice_idx]

# ---------------------------
# PREDICT MOVE (main logic)
# ---------------------------
def predict_move_from_model(board: chess.Board, history: List[str]) -> Tuple[str, Dict]:
    """
    Return (uci_move_string, thought_data)
    thought_data: dict with keys 'legal_moves' (list of uci), 'probs' (corresponding floats), 'raw' (optional raw preds)
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, {"error": "no legal moves"}

    # encode history
    inp = encode_history(history)   # shape (1, MAX_HISTORY_LEN)
    try:
        preds = MODEL.predict(inp, verbose=0)[0]   # shape (vocab_size,)
    except Exception as e:
        # some models return logits: apply softmax then
        try:
            raw = MODEL(inp)
            preds = tf.nn.softmax(raw, axis=-1).numpy()[0]
        except Exception as ex:
            print("Model prediction failed:", e, ex)
            preds = None

    # Build legal move scores
    legal_scores = []
    legal_uci = []
    for m in legal_moves:
        u = m.uci()
        legal_uci.append(u)
        idx = MOVE_TO_IDX.get(u, None)
        if idx is None:
            # If the move is not in vocab, score as very small
            score = 1e-12
        else:
            # if model output length equals VOCAB_SIZE, but indices are 0..VOCAB_SIZE-1
            if preds is None:
                score = 1e-12
            else:
                if idx >= len(preds):
                    score = 1e-12
                else:
                    score = float(preds[idx])
        legal_scores.append(score)

    legal_scores = np.array(legal_scores, dtype=float)
    if legal_scores.sum() <= 0:
        # fallback: uniform random
        legal_scores = np.ones_like(legal_scores) / len(legal_scores)
    else:
        legal_scores = legal_scores / legal_scores.sum()

    # sample one
    chosen_move = sample_from_probs(preds, legal_moves, legal_scores)
    thought = {"legal_moves": legal_uci, "probs": legal_scores.tolist(), "raw_len": (len(preds) if preds is not None else None)}
    return chosen_move.uci(), thought

# ---------------------------
# Lichess event handling
# ---------------------------
def handle_game_loop(game_id: str, my_color: chess.Color):
    """
    Stream a game's state and reply when it's our turn.
    Runs in a background thread per active game.
    """
    try:
        stream = BERSERK_CLIENT.bots.stream_game_state(game_id)
    except Exception as e:
        print(f"[{game_id}] Failed to open stream: {e}")
        return

    board = chess.Board()
    move_history = []
    print(f"[{game_id}] game handler started for color {my_color}")
    for event in stream:
        try:
            # events: gameFull (first) then gameState with moves
            state = event.get("state", event)
            moves_text = state.get("moves", "")
            moves = moves_text.split() if moves_text else []
            # rebuild board from moves
            board = chess.Board()
            for u in moves:
                try:
                    board.push_uci(u)
                except Exception:
                    # sometimes lichess uses long algebraic? we assume uci
                    pass
            # update history stored (list of uci)
            move_history = moves.copy()

            # update ACTIVE_GAMES
            with ACTIVE_GAMES_LOCK:
                if game_id not in ACTIVE_GAMES:
                    ACTIVE_GAMES[game_id] = {"moves": move_history.copy(), "last_thought": None, "white": None, "black": None, "result": None}
                else:
                    ACTIVE_GAMES[game_id]["moves"] = move_history.copy()

            # check if it's our turn
            if board.turn == my_color and not board.is_game_over():
                print(f"[{game_id}] It's our turn (moves len={len(move_history)})")
                uci_move, thought = predict_move_from_model(board, move_history)
                if uci_move is None:
                    print(f"[{game_id}] No move predicted, resign or offer draw fallback")
                    try:
                        BERSERK_CLIENT.bots.resign_game(game_id)
                    except Exception:
                        pass
                    continue
                # send move
                try:
                    BERSERK_CLIENT.bots.make_move(game_id, uci_move)
                    print(f"[{game_id}] Played {uci_move}")
                    with ACTIVE_GAMES_LOCK:
                        ACTIVE_GAMES[game_id]["moves"].append(uci_move)
                        ACTIVE_GAMES[game_id]["last_thought"] = thought
                except Exception as e:
                    print(f"[{game_id}] Failed to make move {uci_move}: {e}")
            else:
                # not our turn, skip
                pass

            # check game over
            if "status" in state and state.get("status") in ("mate", "resign", "draw", "stalemate", "timeout"):
                res = state.get("status")
                with ACTIVE_GAMES_LOCK:
                    ACTIVE_GAMES[game_id]["result"] = res
                print(f"[{game_id}] Game ended: {res}")
                break

        except Exception as e:
            print(f"[{game_id}] Error in game loop: {e}\n{traceback.format_exc()}")
            continue

    print(f"[{game_id}] exiting handler")

def incoming_event_loop():
    """
    Main loop listening to incoming bot events (challenges & game starts).
    Reconnects with exponential backoff on 429 or other transient errors.
    """
    global BERSERK_CLIENT
    if not LICHESS_TOKEN:
        print("LICHESS_TOKEN not set; skipping lichess connection.")
        return

    session = berserk.TokenSession(LICHESS_TOKEN)
    BERSERK_CLIENT = berserk.Client(session=session, timeout=60)

    backoff = 1.0
    last_event_time = time.time()
    while True:
        try:
            print("Connecting to Lichess event stream...")
            for event in BERSERK_CLIENT.bots.stream_incoming_events():
                last_event_time = time.time()
                # handle challenge
                if event["type"] == "challenge":
                    chal = event["challenge"]
                    cid = chal["id"]
                    variant_key = chal["variant"]["key"]
                    print("Received challenge:", cid, "variant:", variant_key)
                    # Accept standard & 'fromPosition' (position)
                    if variant_key in ("standard", "fromPosition"):
                        try:
                            BERSERK_CLIENT.bots.accept_challenge(cid)
                            print("Accepted challenge", cid)
                        except Exception as e:
                            print("Failed to accept challenge:", e)
                    else:
                        try:
                            BERSERK_CLIENT.bots.decline_challenge(cid)
                            print("Declined non-standard challenge", cid)
                        except Exception as e:
                            print("Failed to decline challenge:", e)

                # handle gameStart
                elif event["type"] == "gameStart":
                    gid = event["game"]["id"]
                    my_color_str = event["game"]["color"]
                    my_color = chess.WHITE if my_color_str == "white" else chess.BLACK
                    print("Game started:", gid, "color:", my_color_str)
                    # spawn thread per game
                    t = threading.Thread(target=handle_game_loop, args=(gid, my_color), daemon=True)
                    t.start()

                else:
                    # other event types: challengeCanceled, etc.
                    pass

            # If stream finished normally, reconnect
            print("Event stream closed; reconnecting...")
            time.sleep(1)
            backoff = 1.0

        except berserk.exceptions.ResponseError as re:
            # often 401 or 429
            code = getattr(re, "status_code", None)
            print("Berserk ResponseError:", re)
            if "Too Many Requests" in str(re) or (code == 429):
                wait = backoff
                print(f"429 detected. Backing off for {wait:.1f}s")
                time.sleep(wait)
                backoff = min(300, backoff * EVENT_BACKOFF_BASE)
            elif "Unauthorized" in str(re) or code == 401:
                print("401 Unauthorized for Lichess token. Check LICHESS_TOKEN.")
                break
            else:
                # generic transient
                print("Transient error, sleeping briefly.")
                time.sleep(5)
        except Exception as e:
            print("Exception in incoming_event_loop:", e, traceback.format_exc())
            time.sleep(5)

# ---------------------------
# FLASK UI endpoints
# ---------------------------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Imitation Bot Dashboard</title>
  <style>
    body { font-family: Inter, system-ui, sans-serif; background:#0b1020; color:#e6eef8; padding:20px; }
    .game { background:#0f1724; border-radius:8px; padding:12px; margin:8px 0; box-shadow:0 6px 18px rgba(0,0,0,0.6); }
    pre { white-space:pre-wrap; word-wrap:break-word; }
    .small { color:#9fb0d5; font-size:0.9em }
    .muted { color:#7b8aa1 }
  </style>
</head>
<body>
  <h1>♟ Imitation Bot — Dashboard</h1>
  <p class="small">Active games and latest model thoughts. Bot connects to Lichess when LICHESS_TOKEN provided.</p>
  <div id="games"></div>
  <script>
    async function refresh(){
      const r = await fetch("/status");
      const j = await r.json();
      const g = document.getElementById("games");
      g.innerHTML = "";
      if(Object.keys(j).length === 0){
        g.innerHTML = "<p class='muted'>No active games tracked.</p>";
      }
      for(const [id, info] of Object.entries(j)){
        const moves = info.moves.join(" ");
        const lm = info.last_thought ? (info.last_thought.legal_moves.slice(0,10).join(", ")) : "";
        const probs = info.last_thought ? info.last_thought.probs.slice(0,10).map(x => x.toFixed(3)).join(", ") : "";
        const res = info.result || "";
        const html = `<div class="game"><strong>${id}</strong> ${res ? ("<span class='small'>Result: "+res+"</span>": "")}
        <div class="small">White: ${info.white || "-"} | Black: ${info.black || "-"}</div>
        <div><pre>${moves}</pre></div>
        <div class="small">Top legal moves: ${lm}</div>
        <div class="small">Top probs: ${probs}</div>
        </div>`;
        g.insertAdjacentHTML("beforeend", html);
      }
    }
    setInterval(refresh, 2000);
    refresh();
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/status")
def status():
    with ACTIVE_GAMES_LOCK:
        # shallow copy for safety
        return jsonify(ACTIVE_GAMES)

@app.route("/test_move", methods=["POST"])
def test_move():
    """
    POST JSON: {"fen": "...", "history": ["e2e4", ...]} returns predicted move + thought
    """
    data = request.get_json(force=True)
    fen = data.get("fen")
    history = data.get("history", [])
    try:
        board = chess.Board(fen) if fen else chess.Board()
    except Exception as e:
        return jsonify({"error": "invalid fen", "details": str(e)}), 400
    try:
        uci, thought = predict_move_from_model(board, history)
        return jsonify({"move": uci, "thought": thought})
    except Exception as e:
        return jsonify({"error": "prediction failed", "details": str(e)}), 500

# ---------------------------
# STARTUP: load model/vocab & start lichess thread
# ---------------------------
def startup():
    global MODEL, MOVE_TO_IDX, MOVE_VOCAB
    # load vocab
    if not os.path.exists(VOCAB_FILENAME):
        raise RuntimeError(f"vocab not found at {VOCAB_FILENAME}: upload your vocab.npz into project root or set VOCAB_PATH env var.")
    load_vocab(VOCAB_FILENAME)

    # load model (try many names)
    model_path = None
    for fn in MODEL_FILENAMES:
        if os.path.exists(fn):
            model_path = fn
            break
    # Also consider environment override
    if model_path is None:
        env_model = os.environ.get("MODEL_PATH", None)
        if env_model and os.path.exists(env_model):
            model_path = env_model

    if model_path is None:
        raise RuntimeError(f"No model file found. Place one of {MODEL_FILENAMES} in project root or set MODEL_PATH env var.")

    # load with keras
    print("Loading model from:", model_path)
    # allow custom object scope if no tf/keras available
    if keras is None:
        raise RuntimeError("Keras/TensorFlow is not importable in this environment.")
    model = keras.models.load_model(model_path)
    # ensure global pointer
    globals()['MODEL'] = model
    print("Model loaded OK.")

    # start lichess listener in background thread (if token provided)
    if LICHESS_TOKEN:
        t = threading.Thread(target=incoming_event_loop, daemon=True)
        t.start()
    else:
        print("LICHESS_TOKEN not set, skipping lichess connection. Use /test_move for local testing.")

# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    # startup (load model & vocab)
    try:
        startup()
    except Exception as e:
        print("Startup failed:", e)
        traceback.print_exc()
        # Still start web server so you can inspect status and test_move
    # Run flask with threaded workers
    app.run(host="0.0.0.0", port=PORT, threaded=True)
