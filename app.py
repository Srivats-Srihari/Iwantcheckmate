import berserk
import chess
import chess.pgn
import chess.engine
import chess.polyglot
import numpy as np
import tensorflow as tf
import io
import time
import threading

# =========================
# CONFIG
# =========================
LICHESS_TOKEN = "YOUR_LICHESS_TOKEN_HERE"  # replace
MODEL_PATH = "model.h5"
VOCAB_PATH = "vocab.npz"
MAX_LEN = 48  # same as model training seq length

# =========================
# LOAD MODEL + VOCAB
# =========================
print("Loading model and vocab...")
model = tf.keras.models.load_model(MODEL_PATH)
vocab = np.load(VOCAB_PATH, allow_pickle=True)
token_to_id = vocab["token_to_id"].item()
id_to_token = vocab["id_to_token"].item()
print("Loaded model ✅")

# =========================
# CONNECT TO LICHESS
# =========================
session = berserk.TokenSession(LICHESS_TOKEN)
client = berserk.Client(session=session)
print("Connected to Lichess ✅")


# =========================
# TOKENIZER UTILITIES
# =========================
def encode_moves(moves):
    tokens = []
    for mv in moves[-MAX_LEN:]:
        if mv in token_to_id:
            tokens.append(token_to_id[mv])
        else:
            tokens.append(token_to_id.get("<UNK>", 0))
    x = np.zeros((1, MAX_LEN))
    x[0, -len(tokens):] = tokens
    return x


def predict_move(moves, board):
    """Predict next move based on the current move sequence"""
    x = encode_moves(moves)
    preds = model.predict(x, verbose=0)[0]
    token_id = np.argmax(preds)
    move_uci = id_to_token.get(str(token_id))
    if move_uci and chess.Move.from_uci(move_uci) in board.legal_moves:
        return chess.Move.from_uci(move_uci)
    else:
        # fallback: random legal move
        return np.random.choice(list(board.legal_moves))


# =========================
# GAME LOOP
# =========================
def play_game(game_id):
    print(f"Starting game: {game_id}")
    board = chess.Board()
    moves = []
    stream = client.bots.stream_game_state(game_id)

    for event in stream:
        if event["type"] == "gameFull":
            print("Game started.")
            if event["white"]["id"] == client.account.get()["id"]:
                my_color = chess.WHITE
            else:
                my_color = chess.BLACK

        elif event["type"] == "gameState":
            state = event
            moves_san = state["moves"].split()
            board.reset()
            for mv in moves_san:
                board.push_uci(mv)
            moves = moves_san

            if board.turn == my_color:
                move = predict_move(moves, board)
                client.bots.make_move(game_id, move.uci())
                print(f"Played: {move}")
        elif event["type"] == "chatLine":
            print(f"Chat: {event['username']}: {event['text']}")
        elif event["type"] == "gameFinish":
            print("Game finished.")
            break


# =========================
# MAIN LOOP
# =========================
def listen_forever():
    print("Listening for events...")
    for event in client.bots.stream_incoming_events():
        if event["type"] == "challenge":
            ch = event["challenge"]
            if ch["variant"]["key"] == "standard":
                client.bots.accept_challenge(ch["id"])
                print(f"Accepted challenge from {ch['challenger']['name']}")
        elif event["type"] == "gameStart":
            game_id = event["game"]["id"]
            threading.Thread(target=play_game, args=(game_id,)).start()


if __name__ == "__main__":
    listen_forever()
