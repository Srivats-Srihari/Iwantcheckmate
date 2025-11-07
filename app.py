import os
import time
import json
import random
import threading
import numpy as np
import chess
import berserk
import tensorflow as tf
from tensorflow import keras
from flask import Flask, jsonify

# ===============================
# CONFIGURATION
# ===============================
MODEL_PATH = "./model.h5"
VOCAB_PATH = "./vocab.npz"
SEQ_LEN = 48  # should match training
TOKEN = os.environ.get("Lichess_token")  # Set in Render secrets
THINK_DELAY = 1.2  # seconds to simulate "thinking"

# ===============================
# LOAD MODEL & VOCAB
# ===============================
print("🔁 Loading imitation model and vocab...")

try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

try:
    vocab_data = np.load(VOCAB_PATH, allow_pickle=True)
    moves_vocab = vocab_data["moves"]
    move_to_idx = {m: i for i, m in enumerate(moves_vocab)}
    idx_to_move = {i: m for i, m in enumerate(moves_vocab)}
    print(f"✅ Loaded vocab with {len(moves_vocab)} moves")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load vocab: {e}")

# ===============================
# LICHESS CLIENT
# ===============================
if not TOKEN:
    raise EnvironmentError("❌ Missing Lichess_token environment variable!")

session = berserk.TokenSession(TOKEN)
client = berserk.Client(session=session)

# ===============================
# MOVE PREDICTION
# ===============================
def predict_next_move(board: chess.Board):
    """Predict next move using the imitation model."""
    moves = list(board.move_stack)
    seq = [move_to_idx.get(m.uci(), 0) for m in moves[-SEQ_LEN:]]
    seq = np.pad(seq, (SEQ_LEN - len(seq), 0))
    seq = np.expand_dims(seq, axis=0)

    # Model inference
    preds = model.predict(seq, verbose=0)[0]
    move_idx = np.argmax(preds)
    move_uci = idx_to_move.get(move_idx, None)

    if move_uci not in [m.uci() for m in board.legal_moves]:
        legal = [m.uci() for m in board.legal_moves]
        move_uci = random.choice(legal)
        print(f"⚠️ Model suggested illegal move, picked random {move_uci}")
    return move_uci


# ===============================
# GAME HANDLER
# ===============================
def handle_game(game_id, my_color):
    """Streams a Lichess game and responds to moves."""
    print(f"🎮 Handling game {game_id}")
    board = chess.Board()
    game_stream = client.bots.stream_game_state(game_id)

    for event in game_stream:
        if event["type"] not in ["gameFull", "gameState"]:
            continue

        state = event.get("state", event)
        moves_str = state.get("moves", "")
        moves = moves_str.split() if moves_str else []

        board = chess.Board()
        for mv in moves:
            try:
                board.push_uci(mv)
            except Exception:
                pass

        if board.is_game_over():
            print(f"🏁 Game {game_id} is over: {board.result()}")
            break

        if (board.turn == chess.WHITE and my_color == chess.WHITE) or (
            board.turn == chess.BLACK and my_color == chess.BLACK
        ):
            print(f"🤔 Predicting move for {game_id} ...")
            move_uci = predict_next_move(board)
            time.sleep(THINK_DELAY)
            try:
                client.bots.make_move(game_id, move_uci)
                print(f"✅ Played move {move_uci}")
            except Exception as e:
                print(f"❌ Failed to make move: {e}")
        else:
            print(f"⏳ Waiting for opponent... ({len(moves)} moves played)")


# ===============================
# MAIN LOOP
# ===============================
def main_loop():
    print("🚀 ImitationBot online and awaiting challenges...")
    while True:
        try:
            for event in client.bots.stream_incoming_events():
                if event["type"] == "challenge":
                    variant = event["challenge"]["variant"]["key"]
                    chal_id = event["challenge"]["id"]
                    if variant in ["standard", "fromPosition"]:
                        client.bots.accept_challenge(chal_id)
                        print(f"✅ Accepted challenge {chal_id}")
                    else:
                        client.bots.decline_challenge(chal_id)
                        print(f"❌ Declined non-standard challenge: {variant}")

                elif event["type"] == "gameStart":
                    game_id = event["game"]["id"]
                    color_str = event["game"]["color"]
                    my_color = chess.WHITE if color_str == "white" else chess.BLACK
                    print(f"♟️ Game started {game_id} ({color_str})")
                    threading.Thread(target=handle_game, args=(game_id, my_color), daemon=True).start()
        except Exception as e:
            print(f"⚠️ Stream error: {e}")
            time.sleep(10)


# ===============================
# FLASK SERVER (RENDER KEEPALIVE)
# ===============================
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "bot": "ImitationBot",
        "model": os.path.basename(MODEL_PATH),
        "vocab_size": len(moves_vocab),
        "games": "handled in background"
    })

@app.route("/status")
def status():
    return jsonify({"alive": True, "timestamp": time.time()})


if __name__ == "__main__":
    # Background thread runs the Lichess listener
    threading.Thread(target=main_loop, daemon=True).start()

    port = int(os.environ.get("PORT", 10000))
    print(f"🌐 Starting Flask keepalive on port {port}")
    app.run(host="0.0.0.0", port=port)
