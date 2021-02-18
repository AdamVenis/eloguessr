# uses https://database.lichess.org/standard/lichess_db_standard_rated_2014-09.pgn.bz2
# September 2014 - 1 million games
# 160MB of data

# PLAN: for each move (s, a), predict distribution of ratings
# where the target is discretized rating buckets

# Then, predict the elo of the player by multiplying the buckets over each move


# NOTES:
# - reweight the samples based on frequency of each elo in the dataset
# A:  the correct prior is the base frequency!
#     although this doesn't correct for the bias of the training itself
# - reweight somehow based on state visit distribution
# - reweight by: for each elo, the propensity to make a move across all moves
#   in a given state should sum to 1


# import collections
# c = collections.Counter()
# with open('/Users/avenis/Downloads/lichess_db_standard_rated_2014-09.pgn') as f:
#     for line in f.readlines():
#         if line.startswith('[Opening'):
#             c[line] += 1
# breakpoint()

# https://www.kaggle.com/c/finding-elo

import chess.pgn
import io
import collections

sample_pgn_string = """[Event "Rated Rapid game"]
[Site "https://lichess.org/OJA7VRXr"]
[Date "2021.02.04"]
[White "pacificosky"]
[Black "Gerbil"]
[Result "0-1"]
[UTCDate "2021.02.04"]
[UTCTime "23:42:37"]
[WhiteElo "1933"]
[BlackElo "1859"]
[WhiteRatingDiff "-7"]
[BlackRatingDiff "+10"]
[Variant "Standard"]
[TimeControl "600+0"]
[ECO "A20"]
[Opening "English Opening: King's English Variation"]
[Termination "Normal"]
[Annotator "lichess.org"]

1. c4 e5 { A20 English Opening: King's English Variation } 2. g3 Nf6 3. Bg2 c6 4. e3 d5 5. cxd5 cxd5 6. Ne2?! { (0.05 → -0.58) Inaccuracy. d4 was best. } (6. d4 e4 7. f3 exf3 8. Nxf3 Be7 9. Qd3 O-O 10. Nc3 Nc6) 6... Be7 7. Qb3 O-O 8. Nbc3 e4?! { (-1.34 → -0.43) Inaccuracy. d4 was best. } (8... d4 9. exd4 exd4 10. Nd5 Nxd5 11. Qxd5 Qxd5 12. Bxd5 Na6 13. O-O Nb4 14. Be4 Bh3 15. Bg2) 9. Nf4 Na6 10. Ncxd5 Nxd5?! { (-0.69 → 0.08) Inaccuracy. g5 was best. } (10... g5) 11. Qxd5 Nc5?! { (0.12 → 0.66) Inaccuracy. g5 was best. } (11... g5 12. Qxd8 Rxd8 13. Ne2 f5 14. O-O Nb4 15. f3 exf3 16. Bxf3 Nd3 17. Rb1 f4 18. gxf4) 12. b4?? { (0.66 → -1.14) Blunder. Qxd8 was best. } (12. Qxd8) 12... Qxd5?? { (-1.14 → 1.83) Blunder. Nd3+ was best. } (12... Nd3+) 13. Nxd5 Nd3+ 14. Ke2 Bg4+ 15. f3 exf3+ 16. Bxf3 Bxf3+ 17. Kxf3 Bxb4 18. Nxb4 Nxb4 19. Ba3?! { (1.65 → 0.75) Inaccuracy. Rb1 was best. } (19. Rb1 a5 20. a3 Nd3 21. Ke2 Nc5 22. Rb5 Ne4 23. Bb2 Nd6 24. Rb6 Rfd8 25. Rc1 Rd7) 19... a5 20. Rab1 Rfd8 21. d4 Nxa2 22. Rxb7 Rdb8 23. Rxb8+? { (1.67 → 0.35) Mistake. Rhb1 was best. } (23. Rhb1) 23... Rxb8 24. Ra1 Nb4?? { (0.16 → 1.76) Blunder. Nc3 was best. } (24... Nc3 25. Bd6 Rb5 26. Ra3 Nb1 27. Ra2 Nc3 28. Rc2 Rb3 29. Bc7 Nd5 30. Rc5 Nxe3 31. Bxa5) 25. Bxb4? { (1.76 → 0.38) Mistake. Rc1 was best. } (25. Rc1 f6) 25... axb4 26. Rb1 f5 27. h3 Kf7 28. g4 g6 29. Kf4 fxg4 30. hxg4 Kf6 31. g5+ Ke6 32. e4 Rb5 33. Kg4 Kd6 34. Kf4 b3 35. Rb2 Rb4 36. d5? { (0.00 → -1.43) Mistake. Ke3 was best. } (36. Ke3 Rb5 37. Ke2 Kc6 38. Kf3 Kd7 39. Rb1 Kd6 40. Ke2 Kc6 41. Rb2 Kd6) 36... Kc5? { (-1.43 → 0.00) Mistake. Rb8 was best. } (36... Rb8) 37. Ke5 Kc4?? { (0.00 → 1.91) Blunder. Rb8 was best. } (37... Rb8 38. Ke6 Re8+ 39. Kf6 Rxe4 40. Rxb3 Kxd5 41. Rd3+ Kc6 42. Rc3+ Kd7 43. Rd3+) 38. d6 Kc3 39. d7 Rb8?? { (2.80 → 6.25) Blunder. Kxb2 was best. } (39... Kxb2 40. d8=Q Ra4 41. Kf6 Rxe4 42. Kg7 Ka3 43. Qf6 Ra4 44. Qa1+ Kb4 45. Qb2 Ra2 46. Qd4+) 40. Rf2 b2 41. Rxb2 Kxb2 42. Ke6 Kc3 43. Ke7?? { (4.99 → -55.67) Blunder. Kd6 was best. } (43. Kd6) 43... Kd4 44. d8=Q+ Rxd8 45. Kxd8 Kxe4 46. Ke7 Kf5 47. Kf7 Kxg5 48. Kg7 h5 { White resigns. } 0-1"""


def all_games(
    filename="/Users/avenis/personal/src/eloguessr/finding-elo/data.pgn", limit=None
):
    f = open(filename)
    games_processed = 0
    while True:
        if games_processed == limit:
            return
        game = chess.pgn.read_game(f)
        if game is None:
            return  # end of file

        yield game
        games_processed += 1


def filter_games(games):
    filtered_games = []
    for game in games: 
        if ('WhiteElo' not in game.headers) or ('BlackElo' not in game.headers):
            continue
        if (1100 < int(game.headers['WhiteElo']) < 2000) and (1100 < int(game.headers['BlackElo']) < 2000):
            filtered_games.append(game)
    return filtered_games

def analyse(): 
    games = all_games()
    filtered_games_all = filter_games(games)
    
    new_pgn = open("C:\\Users\\Anthony\\Documents\\Chess\\filtered_games_all.pgn", "w", encoding="utf-8")
    exporter = chess.pgn.FileExporter(new_pgn)
    for game in filtered_games_all:
        game.accept(exporter)
    new_pgn.close()





game = chess.pgn.read_game(io.StringIO(sample_pgn_string))


ParsedGame = collections.namedtuple("ParsedGame", "white_elo black_elo states actions")


def parse_game(game):
    states = []
    actions = []
    for move in game.mainline():
        states.append(move.board())
        actions.append(move.move)

    return ParsedGame(
        white_elo=int(game.headers["WhiteElo"]),
        black_elo=int(game.headers["BlackElo"]),
        states=states,
        actions=actions,
    )


parse_game(game)

# returns (2246.85104, 2241.89132)
def get_average_elos(games):
    white_elos = []
    black_elos = []
    for game in games:
        white_elos.append(int(game.headers["WhiteElo"]))
        black_elos.append(int(game.headers["BlackElo"]))

    return avg(white_elos), avg(black_elos)

def get_all_elos(games):
    white_elos = []
    black_elos = []
    for game in games:
        white_elos.append(int(game.headers["WhiteElo"]))
        black_elos.append(int(game.headers["BlackElo"]))

    return white_elos, black_elos

white_elos, black_elos = get_all_elos(all_games(limit=1000))

import matplotlib.pyplot as plt 

all_elos = white_elos+black_elos
#plt.hist(x=all_elos, bins='auto')

#winrate as a fucntion of difference between elos of two players

import numpy as np

def get_all_elo_diff_by_winner(games):
    white_elo_diff = []
    black_elo_diff = []
    for game in games:
        if game.headers['Result'] == "1-0":
            white_elo_diff.append(np.subtract(int(game.headers['WhiteElo']),int(game.headers["BlackElo"])))
        elif game.headers['Result'] == "0-1":
            black_elo_diff.append(np.subtract(int(game.headers['BlackElo']),int(game.headers["WhiteElo"])))

    return white_elo_diff, black_elo_diff

#print(get_all_elo_diff_by_winner([game]))

white_elo_diff, black_elo_diff = get_all_elo_diff_by_winner(all_games(limit=1000))
all_elos_diff_by_winners = white_elo_diff+black_elo_diff
#print(white_elo_diff)
plt.hist(x=all_elos_diff_by_winners, bins="auto")


def mse(prediction, target):
    return (prediction - target) ** 2


# returns the benchmark: (215.49107412479475, 218.51960401279564)
def mae(prediction, target):
    return abs(prediction - target)


def avg(values):
    return sum(values) / len(values)


def get_loss(games, white_elo, black_elo, loss=mse):
    white_elo_se = []
    black_elo_se = []
    for game in games:
        white_elo_se.append(loss(white_elo, int(game.headers["WhiteElo"])))
        black_elo_se.append(loss(black_elo, int(game.headers["BlackElo"])))

    return avg(white_elo_se), avg(black_elo_se)


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import models


tf.compat.v1.enable_eager_execution()

NUM_BUCKETS = 12  # < 1000, each 200 up to 3000, > 3000


def build_model():
    input_layer = layers.Input(shape=(64*14,), dtype="int32")
    inputs = [input_layer]

    dense = layers.Dense(256, activation="softmax")(input_layer)
    dense2 = layers.Dense(32, activation="softmax")(dense)
    output_layer = layers.Dense(NUM_BUCKETS, activation="softmax")(dense2)

    model = models.Model(inputs=inputs, outputs=[output_layer])
    learning_rate = 1e-4
    optimizer = optimizers.Adam(learning_rate)

    # model.compile(loss='mse', optimizer=optimizer)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=[
            "categorical_accuracy",
            category_topK,
            true_bucket_accuracy,
        ],
    )

    model.summary()
    return model


def true_bucket_accuracy(y_true, y_pred):
    return K.mean(K.cast(K.max(y_pred * y_true, axis=1), "float32"), axis=0)


def category_topK(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


def split(X, Y, train_percentage=1.0):
    train_samples = int(train_percentage * len(X))
    trainX, testX = X[:train_samples], X[train_samples:]
    trainY, testY = Y[:train_samples], Y[train_samples:]
    return trainX, trainY, testX, testY


def generate_dataset(limit=None):
    X = []
    Y = []
    for game in all_games():
        if limit is not None and len(X) >= limit:
            break

        for move in game.mainline():
            X.append(make_input(move.board(), move.move))
            key = 'WhiteElo' if move.turn() else 'BlackElo'
            Y.append(bucket_index(int(game.headers[key])))
    return split(X, Y)

def bucket_index(elo):
    result = np.zeros(NUM_BUCKETS)
    
    if elo < 1000:
        result[0] = 1
    elif elo > 3000:
        result[-1] = 1
    else:
        clipped_elo = max(1000, min(elo, 3000))
        result[1 + ((clipped_elo - 1000) // 200)] = 1
    return result
    
    
    
def make_input(board, move):
    board_state = board._board_state()
    bits = [
        board_state.pawns & board_state.occupied_w,
        board_state.knights & board_state.occupied_w,
        board_state.bishops & board_state.occupied_w,
        board_state.rooks & board_state.occupied_w,
        board_state.queens & board_state.occupied_w,
        board_state.kings & board_state.occupied_w,
        board_state.pawns & board_state.occupied_b,
        board_state.knights & board_state.occupied_b,
        board_state.bishops & board_state.occupied_b,
        board_state.rooks & board_state.occupied_b,
        board_state.queens & board_state.occupied_b,
        board_state.kings & board_state.occupied_b,
    ]
    result = np.zeros(64 * 14, dtype=np.int64)
    for i, mask in enumerate(bits):
        for j in range(64):
            result[64 * i + j] = 1 if (mask & (1 << j)) > 0 else 0
    result[64 * 12 + move.from_square] = 1
    result[64 * 13 + move.to_square] = 1
    return result
    
    
def train(dataset):
    model = build_model()
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=2,
        callbacks=callbacks,
        verbose=1,
    )
    model.save("model.h5")


def visualize_mask(mask):
    i = 0
    while i < len(mask):
        for j in range(8):
            print(mask[i+8*j:i+8*(j+1)])
        i += 64
        print()


def predict_elo(model, game, white=True):
    buckets = np.ones(NUM_BUCKETS)
    for move in game.mainline():
        if move.turn() == white:
            nn_input = make_input(move.board(), move.move)
            buckets *= model(np.array([nn_input]))
            buckets /= np.sum(buckets)
    return buckets

        

model = build_model()
data = generate_dataset(limit=100_000)
model.fit(
    x=np.array(data[0]), 
    y=np.array(data[1]), 
    epochs=30, 
    validation_split=0.8
)





