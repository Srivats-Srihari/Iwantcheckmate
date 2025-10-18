# app.py
import os
import time
import threading
import argparse
import logging

from flask import Flask, jsonify
import numpy as np
import chess
import berserk

# TensorFlow import (ensure TF is available in the environment)
from tensorflow.keras.models import load_model

# ------------- CONFIG -------------
MODEL_PATH_DEFAULT = "model.h5"
VOCAB_PATH_DEFAULT  = "vocab.npz"
HEALTH_PORT = int(os.environ.get("PORT", 8000))  # Render sets $PORT
# ----------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iwcm-bot")

# --------- load TF model & vocab ----------
def load_model_and_vocab(model_path, vocab_path):
    logger.info("Loading TF model from %s", model_path)
    model = load_model(model_path)
    logger.info("Loading vocab from %s", vocab_path)
    data = np.load(vocab_path, allow_pickle=True)
    if "moves" in data:
        moves = list(data["moves"])
    else:
        # fallback if saved differently
        moves = list(data[list(data.files)[0]])
    move_to_idx = {m: i for i, m in enumerate(moves)}
    idx_to_move = {i: m for m, i in move_to_idx.items()}
    logger.info("Loaded model and vocab (%d moves)", len(moves))
    return model, move_to_idx, idx_to_move

# --------- board encoding helper (must match training) ----------
PIECE_TO_PLANE = {
    "P":0,"N":1,"B":2,"R":3,"Q":4,"K":5,
    "p":6,"n":7,"b":8,"r":9,"q":10,"k":11
}
def fen_to_array(fen):
    board = chess.Board(fen)
    arr = np.zeros((8,8,12), dtype=np.int8)
    for sq, piece in board.piece_map().items():
        r = chess.square_rank(sq)
        c = chess.square_file(sq)
        arr[r, c, PIECE_TO_PLANE[piece.symbol()]] = 1
    return arr.reshape(1, -1).astype("float32")  # shape (1,768)

def softmax(x, temp=1.0):
    x = x.astype("float64")
    x = x / max(temp, 1e-8)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

# --------- choose move from model ----------
def choose_move(board, model, move_to_idx, idx_to_move, argmax=True, top_k=40, temp=1.0):
    legal = [m.uci() for m in board.legal_moves]
    if not legal:
        return None
    X = fen_to_array(board.fen())
    probs = model.predict(X, verbose=0)[0]  # shape (vocab_size,)
    legal_pairs = [(u, float(probs[move_to_idx[u]])) for u in legal if u in move_to_idx]
    if not legal_pairs:
        # fallback: try to pick the legal move with best predicted score if possible,
        # otherwise random legal move.
        return np.random.choice(legal)
    legal_pairs.sort(key=lambda x: -x[1])
    ucis, pvals = zip(*legal_pairs)
    pvals = np.array(pvals, dtype="float64")
    if argmax:
        return ucis[0]
    # sampling
    k = min(top_k, len(pvals))
    top_p = pvals[:k]
    if top_p.sum() <= 0:
        probs_sample = np.ones_like(top_p) / len(top_p)
    else:
        probs_sample = softmax(top_p, temp)
    choice = np.random.choice(range(k), p=probs_sample)
    return ucis[choice]

# --------- game handler (streams game state) ----------
def handle_game(client, game_id, model, move_to_idx, idx_to_move, args, my_id):
    logger.info("[%s] game handler starting", game_id)
    try:
        for state in client.bots.stream_game_state(game_id):
            moves_str = state.get("moves", "").strip()
            moves = moves_str.split() if moves_str else []
            board = chess.Board()
            for mv in moves:
                try:
                    board.push_uci(mv)
                except Exception:
                    pass

            white_id = state.get("white", {}).get("id")
            black_id = state.get("black", {}).get("id")

            # compute if it's our turn
            is_my_turn = (board.turn and white_id == my_id) or (not board.turn and black_id == my_id)

            if is_my_turn:
                chosen_uci = choose_move(board, model, move_to_idx, idx_to_move,
                                         argmax=not args.no_argmax, top_k=args.topk, temp=args.temp)
                if chosen_uci is None:
                    logger.warning("[%s] No legal move found; skipping", game_id)
                else:
                    try:
                        if chosen_uci not in [m.uci() for m in board.legal_moves]:
                            # fallback random legal
                            chosen_uci = np.random.choice([m.uci() for m in board.legal_moves])
                        client.bots.make_move(game_id, chosen_uci)
                        logger.info("[%s] played %s", game_id, chosen_uci)
                    except Exception as e:
                        logger.error("[%s] error sending move: %s", game_id, e)
                time.sleep(0.6)

            # end detection
            status = state.get("status")
            if status in ("mate", "resign", "timeout", "draw", "outoftime", "stalemate"):
                logger.info("[%s] finished: status=%s", game_id, status)
                break
    except Exception as e:
        logger.exception("[%s] game handler crashed: %s", game_id, e)

# --------- main bot event loop ----------
def bot_event_loop(token, model_path, vocab_path, args):
    token = token or os.environ.get("LICHESS_TOKEN")
    if not token:
        logger.error("No LICHESS_TOKEN provided (env var or --token). Exiting bot loop.")
        return

    model, move_to_idx, idx_to_move = load_model_and_vocab(model_path, vocab_path)

    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)
    me = client.account.get()
    my_id = me["id"]
    logger.info("Connected to lichess as %s (%s)", me.get("username"), my_id)

    for event in client.bots.stream_incoming_events():
        etype = event.get("type")
        if etype == "challenge":
            ch = event["challenge"]
            variant = ch.get("variant", {}).get("key")
            challenger = ch.get("challenger", {}).get("name")
            logger.info("Incoming challenge from %s, variant=%s", challenger, variant)
            if variant == "standard":
                try:
                    client.bots.accept_challenge(ch["id"])
                    logger.info("Accepted challenge from %s", challenger)
                except Exception as e:
                    logger.error("Failed to accept challenge: %s", e)
            else:
                try:
                    client.bots.decline_challenge(ch["id"])
                    logger.info("Declined non-standard challenge")
                except Exception:
                    pass
        elif etype == "gameStart":
            game_id = event["game"]["id"]
            logger.info("Game started: %s", game_id)
            th = threading.Thread(target=handle_game, args=(client, game_id, model, move_to_idx, idx_to_move, args, my_id))
            th.daemon = True
            th.start()

# ------------- Flask health app and startup -------------
def start_bot_in_thread(model_path, vocab_path, args):
    t = threading.Thread(target=bot_event_loop, args=(args.token, model_path, vocab_path, args))
    t.daemon = True
    t.start()
    return t

def create_app(model_path, vocab_path, args):
    app = Flask(__name__)

    @app.route("/")
    def health():
        return jsonify({"status": "ok", "bot": "iwcm", "model": os.path.basename(model_path)})

    # start bot thread once app starts
    def _start():
        logger.info("Spawning bot thread...")
        start_bot_in_thread(model_path, vocab_path, args)

    # run once on import
    threading.Thread(target=_start, daemon=True).start()
    return app

# ------------- CLI ----------
if __name__ == "__main__":
    from flask import Flask
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH_DEFAULT)
    parser.add_argument("--vocab",  default=VOCAB_PATH_DEFAULT)
    parser.add_argument("--token",  default=None, help="Lichess token (or set LICHESS_TOKEN env var)")
    parser.add_argument("--no-argmax", action="store_true", help="Sample instead of greedy")
    parser.add_argument("--topk", type=int, default=40)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--logdir", type=str, default="games")
    args = parser.parse_args()

    model_path = args.model
    vocab_path = args.vocab

    app = create_app(model_path, vocab_path, args)

    # If running locally use app.run, otherwise Render/Gunicorn will use the app object.
    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting Flask health server on port %s", port)
    app.run(host="0.0.0.0", port=port)
