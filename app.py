import os
import time
import json
import random
import logging
import threading
import numpy as np
import tensorflow as tf
import chess
import chess.engine
import chess.pgn
import chess.svg
import berserk
import requests
from flask import Flask, jsonify, request

# === SETUP ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LichessBot")

# === ENVIRONMENT ===
LICHESS_TOKEN = os.getenv("LICHESS_TOKEN")
if not LICHESS_TOKEN:
    raise ValueError("Missing LICHESS_TOKEN environment variable!")

client = berserk.Client(session=berserk.TokenSession(LICHESS_TOKEN))

# === MODEL + VOCAB ===
logger.info("Loading model and vocab...")
VOCAB_PATH = "vocab.npz"
MODEL_PATH = "model.h5"

if not os.path.exists(VOCAB_PATH) or not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model or vocab not found!")

vocab_data = np.load(VOCAB_PATH, allow_pickle=True)
idx2move = vocab_data["idx2move"].tolist()
move2idx = vocab_data["move2idx"].tolist()
model = tf.keras.models.load_model(MODEL_PATH)

logger.info(f"Loaded vocab of {len(idx2move)} moves and model '{MODEL_PATH}'")

# === RATE LIMITING & RETRIES ===
def safe_request(func, *args, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                sleep_time = 2 ** attempt
                logger.warning(f"Rate limit hit, sleeping {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                raise
        except Exception as e:
            logger.error(f"Request failed ({e}), retrying...")
            time.sleep(2)
    logger.error("Max retries exceeded!")
    return None

# === MOVE DECISION ===
def predict_move(board):
    """Predict next move from board state using the neural model."""
    fen = board.fen()
    input_vector = np.zeros((1, len(move2idx)))
    for move in board.legal_moves:
        move_str = move.uci()
        if move_str in move2idx:
            input_vector[0, move2idx[move_str]] = 1
    preds = model.predict(input_vector, verbose=0)
    move_idx = np.argmax(preds)
    best_move_str = idx2move[move_idx]
    try:
        move = chess.Move.from_uci(best_move_str)
        if move in board.legal_moves:
            return move
    except:
        pass
    # fallback random move
    return random.choice(list(board.legal_moves))

# === GAME HANDLER ===
def play_game(game_id):
    """Stream moves from a game and respond."""
    logger.info(f"Starting game {game_id}")
    try:
        with safe_request(client.bots.stream_game_state, game_id) as stream:
            board = chess.Board()
            for event in stream:
                if event["type"] == "gameFull":
                    state = event["state"]
                    moves = state.get("moves", "").split()
                    for move_str in moves:
                        board.push_uci(move_str)
                    if board.turn == chess.WHITE:
                        move = predict_move(board)
                        make_move(game_id, move.uci())
                elif event["type"] == "gameState":
                    moves = event.get("moves", "").split()
                    board = chess.Board()
                    for move_str in moves:
                        board.push_uci(move_str)
                    if board.turn == chess.WHITE:
                        move = predict_move(board)
                        make_move(game_id, move.uci())
    except Exception as e:
        logger.error(f"Game stream error: {e}")

def make_move(game_id, move_uci):
    """Submit a move to Lichess."""
    try:
        safe_request(client.bots.make_move, game_id, move_uci)
        logger.info(f"Played move {move_uci} in {game_id}")
    except Exception as e:
        logger.error(f"Failed to play move: {e}")

# === EVENT HANDLER ===
def handle_event(event):
    """Handle incoming Lichess event (challenge/gameStart)."""
    t = event["type"]
    if t == "challenge":
        chal = event["challenge"]
        variant = chal["variant"]["key"]
        chal_id = chal["id"]
        logger.info(f"Challenge received: {variant}")
        if variant in ["standard", "fromPosition"]:
            safe_request(client.bots.accept_challenge, chal_id)
            logger.info(f"Accepted challenge {chal_id}")
        else:
            client.bots.decline_challenge(chal_id)
            logger.info(f"Declined non-standard challenge {chal_id}")
    elif t == "gameStart":
        game_id = event["game"]["id"]
        thread = threading.Thread(target=play_game, args=(game_id,))
        thread.start()
    else:
        logger.debug(f"Ignored event: {t}")

# === STREAM EVENTS (PERSISTENT) ===
def event_loop():
    logger.info("Starting persistent event stream...")
    while True:
        try:
            with client.bots.stream_incoming_events() as stream:
                for event in stream:
                    if event:
                        handle_event(event)
        except requests.exceptions.ChunkedEncodingError:
            logger.warning("Stream interrupted, reconnecting...")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Event loop error: {e}")
            time.sleep(10)

# === FLASK APP ===
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "IwantCheckmate Bot is running!"})

@app.route("/ping")
def ping():
    return jsonify({"pong": True, "uptime": time.time()})

@app.route("/health")
def health():
    try:
        return jsonify({"health": "good", "model_loaded": True})
    except Exception as e:
        return jsonify({"health": "error", "details": str(e)}), 500

@app.route("/simulate", methods=["POST"])
def simulate():
    """Simulate a move prediction from FEN."""
    data = request.get_json()
    fen = data.get("fen")
    board = chess.Board(fen)
    move = predict_move(board)
    return jsonify({"move": move.uci()})

# === START BOT THREAD ===
def start_bot():
    bot_thread = threading.Thread(target=event_loop)
    bot_thread.daemon = True
    bot_thread.start()
    logger.info("Bot started.")

start_bot()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port)
