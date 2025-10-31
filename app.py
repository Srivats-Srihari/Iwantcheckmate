import os
import json
import time
import random
import threading
import traceback
import numpy as np
import chess
import chess.engine
import berserk
import tensorflow as tf
from flask import Flask, jsonify


TOKEN = os.environ["Lichess_token"]
MODEL_PATH = "model.h5"
VOCAB_PATH = "vocab.npz"

# === LOAD MODEL + VOCAB ===
print("🔹 Loading model and vocab...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    vocab_data = np.load(VOCAB_PATH, allow_pickle=True)
    move_to_idx = vocab_data["move_to_idx"].item()
    idx_to_move = vocab_data["idx_to_move"].item()
    print(f"✅ Loaded model and vocab — {len(move_to_idx)} moves known.")
except Exception as e:
    print("⚠️ Failed to load model/vocab:", e)
    model, move_to_idx, idx_to_move = None, {}, {}

# === GLOBAL STATE ===
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
session = berserk.TokenSession(TOKEN)
client = berserk.Client(session=session)
app = Flask(__name__)
lock = threading.Lock()
game_data = {}

# === MODEL MOVE PREDICTION ===
def predict_move(board):
    """
    Predicts a move based on FEN using the neural model.
    Falls back to random or worst move if model uncertain.
    """
    try:
        fen = board.fen()
        if model is None or not move_to_idx:
            raise ValueError("Model unavailable.")
        input_vec = np.zeros((1, len(move_to_idx)))
        for token in fen.split():
            if token in move_to_idx:
                input_vec[0, move_to_idx[token]] = 1
        preds = model.predict(input_vec, verbose=0)[0]
        best_idx = np.argmax(preds)
        best_move = idx_to_move.get(best_idx)
        if best_move in [m.uci() for m in board.legal_moves]:
            return chess.Move.from_uci(best_move)
    except Exception as e:
        print("⚠️ Prediction failed:", e)
    return None

# === STOCKFISH WORST MOVE LOGIC ===
def get_worst_survivable_move(board, depth=3):
    legal = list(board.legal_moves)
    if not legal:
        return None

    if board.is_check():
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        return info["pv"][0]

    try:
        cur_eval = engine.analyse(board, chess.engine.Limit(depth=depth))["score"].pov(board.turn).score()
    except Exception:
        cur_eval = 0

    candidates = []
    for mv in legal:
        temp = board.copy()
        temp.push(mv)
        try:
            eval_after = engine.analyse(temp, chess.engine.Limit(depth=depth))["score"].pov(board.turn).score()
            drop = cur_eval - eval_after
            if drop < 900:
                candidates.append((eval_after, mv))
        except Exception:
            continue

    if candidates:
        worst = min(candidates, key=lambda x: x[0])[1]
        return worst
    return random.choice(legal)

# === GAME HANDLER ===
def handle_game(game_id, my_color):
    local_session = berserk.TokenSession(TOKEN)
    local_client = berserk.Client(session=local_session)
    board = chess.Board()

    game_data[game_id] = {
        "moves": [],
        "last_thought": None,
        "result": None,
        "white": None,
        "black": None
    }

    print(f"[{game_id}] Game handler started.")

    try:
        for event in local_client.bots.stream_game_state(game_id):
            if event["type"] not in ["gameFull", "gameState"]:
                continue

            state = event.get("state", event)
            moves = state.get("moves", "")
            board = chess.Board()
            for mv in moves.split():
                board.push_uci(mv)

            with lock:
                game_data[game_id]["moves"] = moves.split()

            if board.is_game_over():
                result = board.result()
                with lock:
                    game_data[game_id]["result"] = result
                print(f"[{game_id}] Game over: {result}")
                break

            if board.turn == my_color:
                print(f"[{game_id}] Thinking...")
                move = predict_move(board)
                thought = None
                if move is None or move not in board.legal_moves:
                    move = get_worst_survivable_move(board)
                    thought = "Model unsure — falling back to survival instinct."
                else:
                    thought = f"Predicted move: {move.uci()} ({len(board.move_stack)} moves in)."

                with lock:
                    game_data[game_id]["last_thought"] = thought
                print(f"[{game_id}] {thought}")

                try:
                    local_client.bots.make_move(game_id, move.uci())
                except Exception as e:
                    print(f"[{game_id}] Move failed: {e}")
                    time.sleep(1)
            else:
                print(f"[{game_id}] Waiting for opponent...")

    except Exception as e:
        print(f"[{game_id}] Stream error:", e)
        traceback.print_exc()
        time.sleep(3)
        print(f"[{game_id}] Reconnecting stream...")
        threading.Thread(target=handle_game, args=(game_id, my_color), daemon=True).start()

# === EVENT LISTENER ===
def listen_events():
    print("♟️ Iwantcheckmate bot online — awaiting battles.")
    for event in client.bots.stream_incoming_events():
        try:
            if event["type"] == "challenge":
                chal = event["challenge"]
                variant = chal["variant"]["key"]
                if variant in ["standard", "fromPosition"]:
                    client.bots.accept_challenge(chal["id"])
                    print(f"✅ Accepted challenge: {chal['id']}")
                else:
                    client.bots.decline_challenge(chal["id"])
                    print(f"❌ Declined {variant} challenge.")
            elif event["type"] == "gameStart":
                gid = event["game"]["id"]
                color_str = event["game"]["color"]
                my_color = chess.WHITE if color_str == "white" else chess.BLACK
                game_data[gid] = {"moves": [], "result": None, "white": color_str, "last_thought": None}
                threading.Thread(target=handle_game, args=(gid, my_color), daemon=True).start()
                print(f"🎮 Game started: {gid} ({color_str})")
        except Exception as e:
            print("⚠️ Event loop error:", e)
            time.sleep(2)

# === FLASK DASHBOARD ===
@app.route("/")
def dashboard():
    with lock:
        return jsonify(game_data)

@app.route("/status")
def status():
    with lock:
        stats = {
            "games_active": len(game_data),
            "loaded_moves": len(move_to_idx),
            "model_loaded": model is not None,
            "engine_alive": engine is not None
        }
        return jsonify(stats)

@app.route("/moves/<game_id>")
def get_moves(game_id):
    with lock:
        if game_id not in game_data:
            return jsonify({"error": "Game not found."}), 404
        return jsonify(game_data[game_id])

# === STARTUP ===
if __name__ == "__main__":
    threading.Thread(target=listen_events, daemon=True).start()
    port = int(os.environ.get("PORT", 10000))
    print(f"🌐 Flask live on port {port}")
    app.run(host="0.0.0.0", port=port)
