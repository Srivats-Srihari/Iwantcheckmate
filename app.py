# app.py
"""
Imitation bot app - robust, hybrid model + position DB selection
- Loads model.h5, vocab.npz, pos_db.npz
- Connects to Lichess with berserk
- Accepts standard/fromPosition challenges
- Hybrid move selection: pos_db (exact FEN) + model logits + weighted imitation fallback
- Admin endpoints for diagnostics and reload
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
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import chess
import berserk
from flask import Flask, jsonify, request, render_template_string, send_file

# Optional TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:
    tf = None
    keras_load_model = None

# ----------------------------
# Config via ENV (defaults)
# ----------------------------
LICHESS_TOKEN = os.environ.get("Lichess_token")
if not LICHESS_TOKEN:
    print("WARNING: Set Lichess_token env var to connect to Lichess")

MODEL_PATH = os.environ.get("MODEL_PATH", "model.h5")
VOCAB_PATH = os.environ.get("VOCAB_PATH", "vocab.npz")
POS_DB_PATH = os.environ.get("POS_DB_PATH", "pos_db.npz")
PORT = int(os.environ.get("PORT", "10000"))

ARGMAX = os.environ.get("ARGMAX", "false").lower() in ("1", "true", "yes")
TOP_K = int(os.environ.get("TOP_K", "8"))
TEMP = float(os.environ.get("TEMP", "0.05"))
ALPHA = float(os.environ.get("ALPHA", "0.6"))  # model weight in hybrid mix
SEQ_LEN = int(os.environ.get("SEQ_LEN", "64"))

# Debug
DEBUG_FORCE_RANDOM = os.environ.get("DEBUG_FORCE_RANDOM", "false").lower() in ("1", "true", "yes")
DEBUG_LOG_PROBS = os.environ.get("DEBUG_LOG_PROBS", "true").lower() in ("1", "true", "yes")

# ----------------------------
# Global state
# ----------------------------
MODEL = None
MODEL_INPUT_SHAPE = None
IDX2MOVE: List[str] = []
MOVE2IDX: Dict[str, int] = {}
VOCAB_SIZE = 0
MOVE_FREQ = Counter()

POS_DB: Dict[str, List[Tuple[str,int]]] = {}  # fen -> [(move, count), ...]
POS_DB_LOADED = False

GAMES: Dict[str, Dict[str, Any]] = {}
GAMES_LOCK = threading.Lock()

# berserk client
CLIENT = None

# ----------------------------
# Logging
# ----------------------------
def log(*args, **kwargs):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(ts, *args, **kwargs)
    sys.stdout.flush()

# ----------------------------
# Load vocab.npz (moves)
# ----------------------------
def load_vocab(path: str = VOCAB_PATH):
    global IDX2MOVE, MOVE2IDX, VOCAB_SIZE, MOVE_FREQ
    if not os.path.exists(path):
        log("[vocab] not found:", path)
        IDX2MOVE, MOVE2IDX, VOCAB_SIZE, MOVE_FREQ = [], {}, 0, Counter()
        return
    data = np.load(path, allow_pickle=True)
    keys = data.files
    log("[vocab] keys:", keys)
    if "moves" in data.files:
        arr = data["moves"]
    elif "vocab" in data.files:
        arr = data["vocab"]
    else:
        arr = data[data.files[0]]
    IDX2MOVE = [str(x) for x in arr.tolist()]
    MOVE2IDX = {m:i for i,m in enumerate(IDX2MOVE)}
    VOCAB_SIZE = len(IDX2MOVE)
    MOVE_FREQ = Counter(IDX2MOVE)
    log(f"[vocab] loaded {VOCAB_SIZE} moves")

# ----------------------------
# Load pos_db.npz
# ----------------------------
def load_pos_db(path: str = POS_DB_PATH):
    global POS_DB, POS_DB_LOADED
    POS_DB = {}
    if not os.path.exists(path):
        log("[pos_db] not found:", path)
        POS_DB_LOADED = False
        return
    data = np.load(path, allow_pickle=True)
    fens = data['fens'].tolist()
    moves_list = data['moves_list'].tolist()
    for fen, moves in zip(fens, moves_list):
        POS_DB[fen] = [(mv, int(cnt)) for mv,cnt in moves]
    POS_DB_LOADED = True
    log("[pos_db] loaded entries:", len(POS_DB))

def get_pos_probs(fen: str) -> Optional[np.ndarray]:
    """Return a length-VOCAB_SIZE probability vector for the exact fen or None"""
    if not POS_DB_LOADED:
        return None
    entry = POS_DB.get(fen)
    if not entry:
        return None
    probs = np.zeros(VOCAB_SIZE, dtype=float)
    total = 0
    for mv, cnt in entry:
        idx = MOVE2IDX.get(mv)
        if idx is not None:
            probs[idx] += cnt
            total += cnt
    if total == 0:
        return None
    probs /= probs.sum()
    return probs

# ----------------------------
# Load Keras model.h5
# ----------------------------
def load_keras(path: str = MODEL_PATH):
    global MODEL, MODEL_INPUT_SHAPE
    if keras_load_model is None:
        log("[model] TensorFlow/Keras not available in this runtime. MODEL disabled.")
        MODEL = None
        MODEL_INPUT_SHAPE = None
        return
    if not os.path.exists(path):
        log("[model] file not found:", path)
        MODEL = None
        MODEL_INPUT_SHAPE = None
        return
    try:
        log("[model] loading", path)
        MODEL = keras_load_model(path)
        try:
            inp = MODEL.inputs[0].shape
            if hasattr(inp, "as_list"):
                MODEL_INPUT_SHAPE = inp.as_list()
            else:
                MODEL_INPUT_SHAPE = tuple(inp)
        except Exception:
            MODEL_INPUT_SHAPE = None
        log("[model] loaded. input_shape:", MODEL_INPUT_SHAPE)
    except Exception as e:
        log("[model] load exception:", e)
        MODEL = None
        MODEL_INPUT_SHAPE = None

# ----------------------------
# Encoders (same as trainer)
# ----------------------------
PIECE_TO_PLANE = {
    "P":0,"N":1,"B":2,"R":3,"Q":4,"K":5,
    "p":6,"n":7,"b":8,"r":9,"q":10,"k":11
}

def encode_planes(board: chess.Board) -> np.ndarray:
    arr = np.zeros((8,8,12), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        r = chess.square_rank(sq)
        c = chess.square_file(sq)
        plane = PIECE_TO_PLANE[piece.symbol()]
        arr[r,c,plane] = 1.0
    return arr.reshape(1, -1).astype(np.float32)

def encode_sequence(board: chess.Board, seq_len: int = SEQ_LEN) -> np.ndarray:
    moves = [m.uci() for m in board.move_stack]
    toks = []
    for mv in moves[-seq_len:]:
        idx = MOVE2IDX.get(mv)
        toks.append((idx + 1) if idx is not None else 0)
    if len(toks) < seq_len:
        toks = [0] * (seq_len - len(toks)) + toks
    return np.array(toks, dtype=np.int32).reshape(1, seq_len)

def encode_onehot(board: chess.Board) -> np.ndarray:
    vec = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
    for mv in [m.uci() for m in board.move_stack]:
        idx = MOVE2IDX.get(mv)
        if idx is not None:
            vec[0, idx] += 1.0
    s = vec.sum()
    if s > 0:
        vec /= s
    return vec

# ----------------------------
# Model inference wrapper
# ----------------------------
def softmax(x):
    x = np.array(x, dtype=np.float64).flatten()
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-12)

def infer_probs(board: chess.Board):
    """Attempt sequence -> model; fallback to planes -> model; final fallback onehot -> model"""
    if MODEL is None:
        raise RuntimeError("No model loaded")
    # sequence
    try:
        seq = encode_sequence(board)
        out = MODEL.predict(seq, verbose=0)
        arr = np.array(out[0]).flatten()
        if arr.size == VOCAB_SIZE:
            return softmax(arr)
    except Exception as e:
        log("[infer] seq failed:", e)
    # planes
    try:
        planes = encode_planes(board)
        out = MODEL.predict(planes, verbose=0)
        arr = np.array(out[0]).flatten()
        if arr.size == VOCAB_SIZE:
            return softmax(arr)
    except Exception as e:
        log("[infer] planes failed:", e)
    # onehot
    try:
        vec = encode_onehot(board)
        out = MODEL.predict(vec, verbose=0)
        arr = np.array(out[0]).flatten()
        if arr.size == VOCAB_SIZE:
            return softmax(arr)
    except Exception as e:
        log("[infer] onehot failed:", e)
    raise RuntimeError("Model inference did not produce a valid-size output")

# ----------------------------
# Choose legal move from prob vector
# ----------------------------
def choose_from_probs(board: chess.Board, probs: np.ndarray, argmax: bool = ARGMAX, top_k: int = TOP_K, temp: float = TEMP) -> Optional[str]:
    legal = [m.uci() for m in board.legal_moves]
    if not legal:
        return None
    items = []
    for mv in legal:
        idx = MOVE2IDX.get(mv)
        if idx is None: continue
        items.append((mv, float(probs[idx]), idx))
    if not items: return None
    items.sort(key=lambda x: -x[1])
    if argmax:
        return items[0][0]
    k = min(top_k, len(items))
    top_scores = np.array([it[1] for it in items[:k]])
    # convert to probabilities with temperature
    logits = top_scores / (temp if temp > 0 else 1e-6)
    logits = logits - np.max(logits)
    e = np.exp(logits)
    p = e / (e.sum() + 1e-12)
    choice = np.random.choice(range(k), p=p)
    return items[choice][0]

# ----------------------------
# Weighted imitation fallback
# ----------------------------
def weighted_imitation(board: chess.Board):
    legal = [m.uci() for m in board.legal_moves]
    phase = "opening" if len(board.move_stack) < 8 else ("endgame" if len(board.piece_map()) < 12 else "midgame")
    bias = {"opening":1.5,"midgame":1.0,"endgame":0.8}[phase]
    weights = []
    for mv in legal:
        freq = MOVE_FREQ.get(mv, 0)
        weights.append((freq + 1)**bias + random.random()*1e-2)
    s = sum(weights)
    probs = [w/s for w in weights]
    chosen = random.choices(legal, weights=probs, k=1)[0]
    thought = {"method":"weighted_imitation", "phase":phase, "chosen_freq":MOVE_FREQ.get(chosen,0)}
    return chosen, thought

# ----------------------------
# Hybrid selection: pos_db + model + fallback
# ----------------------------
def hybrid_select(board: chess.Board, alpha: float = ALPHA):
    # try pos_db
    fen = board.fen()
    pos_probs = get_pos_probs(fen)
    model_probs = None
    chosen = None
    thought = None
    if MODEL is not None:
        try:
            model_probs = infer_probs(board)
        except Exception as e:
            log("[hybrid] model inference failed:", e)
            model_probs = None

    # combine
    if model_probs is not None and pos_probs is not None:
        final = alpha * model_probs + (1 - alpha) * pos_probs
    elif model_probs is not None:
        final = model_probs
    elif pos_probs is not None:
        final = pos_probs
    else:
        final = None

    if final is not None:
        # normalize
        final = final / (final.sum() + 1e-12)
        if DEBUG_LOG_PROBS:
            topk = np.argsort(final)[::-1][:8]
            top_info = [(int(i), IDX2MOVE[i] if i < len(IDX2MOVE) else None, float(final[i])) for i in topk]
            log("[hybrid] final top:", top_info)
        chosen = choose_from_probs(board, final)
        if chosen:
            thought = {"method":"hybrid", "alpha":alpha}
    if not chosen:
        chosen, thought = weighted_imitation(board)
    return chosen, thought

# ----------------------------
# Game handler
# ----------------------------
def handle_game(game_id: str, my_color_is_white: bool):
    log(f"[{game_id}] handler started; my_color={'white' if my_color_is_white else 'black'}")
    # create local berserk client for streaming
    try:
        local_session = berserk.TokenSession(LICHESS_TOKEN)
        local_client = berserk.Client(session=local_session)
    except Exception as e:
        log("[handle_game] berserk client fail:", e)
        local_client = None

    board = chess.Board()
    with GAMES_LOCK:
        GAMES[game_id] = {"moves": [], "white": None, "black": None, "last_thought": None, "result": None}

    try:
        stream = local_client.bots.stream_game_state(game_id)
        for event in stream:
            etype = event.get("type")
            if etype not in ("gameFull", "gameState"):
                continue
            state = event.get("state", event)  # sometimes returned under top-level
            moves_str = state.get("moves", "")
            moves = moves_str.split() if moves_str else []
            # rebuild board
            board = chess.Board()
            for m in moves:
                try:
                    board.push_uci(m)
                except Exception:
                    log(f"[{game_id}] failed to push historic move {m}")

            # update meta
            white_meta = state.get("white")
            black_meta = state.get("black")
            white_id = white_meta.get("id") if isinstance(white_meta, dict) else white_meta
            black_id = black_meta.get("id") if isinstance(black_meta, dict) else black_meta
            with GAMES_LOCK:
                GAMES[game_id].update({"moves": moves, "white": white_id, "black": black_id})

            status = state.get("status")
            if status in ("mate","resign","timeout","stalemate","draw"):
                winner = state.get("winner")
                res = "1-0" if winner == "white" else ("0-1" if winner=="black" else "1/2-1/2")
                with GAMES_LOCK:
                    GAMES[game_id]["result"] = res
                log(f"[{game_id}] finished: {res}")
                break

            is_my_turn = (board.turn == my_color_is_white)
            if is_my_turn and not board.is_game_over():
                log(f"[{game_id}] our turn; moves={len(moves)}")
                if DEBUG_FORCE_RANDOM:
                    chosen = random.choice([m.uci() for m in board.legal_moves])
                    thought = {"method":"debug-random"}
                else:
                    try:
                        chosen, thought = hybrid_select(board, ALPHA)
                    except Exception as e:
                        log(f"[{game_id}] hybrid_select failed: {e}")
                        traceback.print_exc()
                        chosen, thought = weighted_imitation(board)

                # final safety: ensure legal
                if chosen not in [m.uci() for m in board.legal_moves]:
                    log(f"[{game_id}] chosen not legal -> fallback")
                    chosen, thought = weighted_imitation(board)

                # play move
                try:
                    local_client.bots.make_move(game_id, chosen)
                    log(f"[{game_id}] played {chosen} ; thought={thought}")
                except Exception as e:
                    log(f"[{game_id}] make_move failed: {e}")
                    traceback.print_exc()
                # update local board & store
                try:
                    board.push_uci(chosen)
                    with GAMES_LOCK:
                        GAMES[game_id]["moves"].append(chosen)
                        GAMES[game_id]["last_thought"] = thought
                except Exception:
                    pass

    except Exception:
        log(f"[{game_id}] game stream terminated with exception")
        traceback.print_exc()
    finally:
        log(f"[{game_id}] handler exiting")

# ----------------------------
# Events loop (accept challenges)
# ----------------------------
def start_events_loop():
    global CLIENT
    if not LICHESS_TOKEN:
        log("[events] Lichess token not set; events loop will not start")
        return
    session = berserk.TokenSession(LICHESS_TOKEN)
    CLIENT = berserk.Client(session=session)
    log("[events] starting incoming events stream")
    for event in CLIENT.bots.stream_incoming_events():
        try:
            etype = event.get("type")
            if etype == "challenge":
                ch = event.get("challenge", {})
                cid = ch.get("id")
                variant = ch.get("variant", {}).get("key")
                log("[events] challenge", cid, "variant", variant)
                if variant in ("standard", "fromPosition"):
                    try:
                        CLIENT.bots.accept_challenge(cid)
                        log("[events] accepted", cid)
                    except Exception as e:
                        log("[events] accept failed", e)
                else:
                    try:
                        CLIENT.bots.decline_challenge(cid)
                        log("[events] declined nonstandard", cid)
                    except Exception as e:
                        log("[events] decline failed", e)
            elif etype == "gameStart":
                gid = event.get("game", {}).get("id")
                color = event.get("game", {}).get("color")
                my_color = chess.WHITE if color == "white" else chess.BLACK
                log("[events] gameStart", gid, "color", color)
                t = threading.Thread(target=handle_game, args=(gid, my_color), daemon=True)
                t.start()
        except Exception:
            log("[events] exception in incoming loop")
            traceback.print_exc()

# ----------------------------
# Flask dashboard & admin
# ----------------------------
app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html>
<head><title>Imitation Bot Dashboard</title>
<style>
body{font-family:monospace;background:#0b1020;color:#dbe9ee;padding:12px}
.game{border:1px solid #21313b;padding:10px;margin:8px;border-radius:6px}
</style>
<meta http-equiv="refresh" content="5">
</head>
<body>
<h2>Imitation Bot Dashboard</h2>
<p>Model loaded: {{ model_loaded }}, vocab_size: {{ vocab_size }}, pos_db: {{ pos_db_loaded }}</p>
{% for gid, g in games.items() %}
<div class="game">
<b>Game:</b> {{ gid }}<br/>
<b>White:</b> {{ g.white }} | <b>Black:</b> {{ g.black }}<br/>
<b>Moves:</b> <pre>{{ " ".join(g.moves) }}</pre>
<b>Last thought:</b> <pre>{{ g.last_thought }}</pre>
<b>Result:</b> {{ g.result }}
<form method="post" action="/admin/save/{{ gid }}"><button>Save PGN</button></form>
</div>
{% endfor %}
</body>
</html>
"""

@app.route("/")
def index():
    with GAMES_LOCK:
        return render_template_string(INDEX_HTML, games=GAMES, model_loaded=(MODEL is not None), vocab_size=VOCAB_SIZE, pos_db_loaded=POS_DB_LOADED)

@app.route("/admin/reload", methods=["POST"])
def admin_reload():
    try:
        load_vocab(VOCAB_PATH)
        load_pos_db(POS_DB_PATH)
        load_keras(MODEL_PATH)
        return jsonify({"status":"reloaded","vocab_size":VOCAB_SIZE,"model":(MODEL is not None),"pos_db":POS_DB_LOADED})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/admin/stats")
def admin_stats():
    with GAMES_LOCK:
        active = len(GAMES)
    return jsonify({"model": MODEL is not None, "vocab_size": VOCAB_SIZE, "pos_db_loaded": POS_DB_LOADED, "active_games": active, "top_moves_sample": dict(Counter(MOVE2IDX).most_common()[:20])})

@app.route("/admin/debug_explain", methods=["POST"])
def admin_debug_explain():
    """
    Accept JSON: {"fen": "..."} and returns model top-8, pos_db top-8, final mix given ALPHA
    """
    data = request.get_json(force=True)
    fen = data.get("fen")
    if not fen:
        return jsonify({"error": "provide fen"}), 400
    board = chess.Board(fen)
    res = {}
    pos_probs = get_pos_probs(fen)
    model_probs = None
    try:
        model_probs = infer_probs(board) if MODEL is not None else None
    except Exception as e:
        res["model_error"] = str(e)
    if pos_probs is not None:
        top = np.argsort(pos_probs)[::-1][:12]
        res["pos_top"] = [(int(i), IDX2MOVE[i], float(pos_probs[i])) for i in top]
    if model_probs is not None:
        top = np.argsort(model_probs)[::-1][:12]
        res["model_top"] = [(int(i), IDX2MOVE[i], float(model_probs[i])) for i in top]
    if model_probs is not None or pos_probs is not None:
        if model_probs is None:
            final = pos_probs
        elif pos_probs is None:
            final = model_probs
        else:
            final = ALPHA * model_probs + (1 - ALPHA) * pos_probs
        top = np.argsort(final)[::-1][:12]
        res["final_top"] = [(int(i), IDX2MOVE[i], float(final[i])) for i in top]
    return jsonify(res)

@app.route("/admin/save/<game_id>", methods=["POST"])
def admin_save(game_id):
    with GAMES_LOCK:
        g = GAMES.get(game_id)
        if not g:
            return jsonify({"error": "not found"}), 404
        moves = g.get("moves", [])
        white = g.get("white")
        black = g.get("black")
        result = g.get("result")
    # build pgn
    game = chess.pgn.Game()
    game.headers["Event"] = "Saved"
    game.headers["White"] = white or "white"
    game.headers["Black"] = black or "black"
    node = game
    for u in moves:
        try:
            mv = chess.Move.from_uci(u)
            node = node.add_variation(mv)
        except Exception:
            pass
    outdir = "saved_games"
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{game_id}.pgn")
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(game))
    return jsonify({"saved": path})

# ----------------------------
# Boot sequence
# ----------------------------
def boot():
    log("Booting imitation bot app")
    load_vocab(VOCAB_PATH)
    load_pos_db(POS_DB_PATH)
    load_keras(MODEL_PATH)
    # start event loop in separate thread
    if LICHESS_TOKEN:
        t = threading.Thread(target=start_events_loop, daemon=True)
        t.start()
    else:
        log("No Lichess token; events loop disabled")

if __name__ == "__main__":
    boot()
    app.run(host="0.0.0.0", port=PORT, threaded=True)
