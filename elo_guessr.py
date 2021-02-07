# uses https://database.lichess.org/standard/lichess_db_standard_rated_2014-09.pgn.bz2
# September 2014 - 1 million games
# 160MB of data

# PLAN: for each move (s, a), predict distribution of ratings
# where the target is discretized rating buckets

# Then, predict the elo of the player by multiplying the buckets over each move


# NOTES:
# - reweight the samples based on frequency of each elo in the dataset
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

pgn_string = '''[Event "Rated Rapid game"]
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

1. c4 e5 { A20 English Opening: King's English Variation } 2. g3 Nf6 3. Bg2 c6 4. e3 d5 5. cxd5 cxd5 6. Ne2?! { (0.05 → -0.58) Inaccuracy. d4 was best. } (6. d4 e4 7. f3 exf3 8. Nxf3 Be7 9. Qd3 O-O 10. Nc3 Nc6) 6... Be7 7. Qb3 O-O 8. Nbc3 e4?! { (-1.34 → -0.43) Inaccuracy. d4 was best. } (8... d4 9. exd4 exd4 10. Nd5 Nxd5 11. Qxd5 Qxd5 12. Bxd5 Na6 13. O-O Nb4 14. Be4 Bh3 15. Bg2) 9. Nf4 Na6 10. Ncxd5 Nxd5?! { (-0.69 → 0.08) Inaccuracy. g5 was best. } (10... g5) 11. Qxd5 Nc5?! { (0.12 → 0.66) Inaccuracy. g5 was best. } (11... g5 12. Qxd8 Rxd8 13. Ne2 f5 14. O-O Nb4 15. f3 exf3 16. Bxf3 Nd3 17. Rb1 f4 18. gxf4) 12. b4?? { (0.66 → -1.14) Blunder. Qxd8 was best. } (12. Qxd8) 12... Qxd5?? { (-1.14 → 1.83) Blunder. Nd3+ was best. } (12... Nd3+) 13. Nxd5 Nd3+ 14. Ke2 Bg4+ 15. f3 exf3+ 16. Bxf3 Bxf3+ 17. Kxf3 Bxb4 18. Nxb4 Nxb4 19. Ba3?! { (1.65 → 0.75) Inaccuracy. Rb1 was best. } (19. Rb1 a5 20. a3 Nd3 21. Ke2 Nc5 22. Rb5 Ne4 23. Bb2 Nd6 24. Rb6 Rfd8 25. Rc1 Rd7) 19... a5 20. Rab1 Rfd8 21. d4 Nxa2 22. Rxb7 Rdb8 23. Rxb8+? { (1.67 → 0.35) Mistake. Rhb1 was best. } (23. Rhb1) 23... Rxb8 24. Ra1 Nb4?? { (0.16 → 1.76) Blunder. Nc3 was best. } (24... Nc3 25. Bd6 Rb5 26. Ra3 Nb1 27. Ra2 Nc3 28. Rc2 Rb3 29. Bc7 Nd5 30. Rc5 Nxe3 31. Bxa5) 25. Bxb4? { (1.76 → 0.38) Mistake. Rc1 was best. } (25. Rc1 f6) 25... axb4 26. Rb1 f5 27. h3 Kf7 28. g4 g6 29. Kf4 fxg4 30. hxg4 Kf6 31. g5+ Ke6 32. e4 Rb5 33. Kg4 Kd6 34. Kf4 b3 35. Rb2 Rb4 36. d5? { (0.00 → -1.43) Mistake. Ke3 was best. } (36. Ke3 Rb5 37. Ke2 Kc6 38. Kf3 Kd7 39. Rb1 Kd6 40. Ke2 Kc6 41. Rb2 Kd6) 36... Kc5? { (-1.43 → 0.00) Mistake. Rb8 was best. } (36... Rb8) 37. Ke5 Kc4?? { (0.00 → 1.91) Blunder. Rb8 was best. } (37... Rb8 38. Ke6 Re8+ 39. Kf6 Rxe4 40. Rxb3 Kxd5 41. Rd3+ Kc6 42. Rc3+ Kd7 43. Rd3+) 38. d6 Kc3 39. d7 Rb8?? { (2.80 → 6.25) Blunder. Kxb2 was best. } (39... Kxb2 40. d8=Q Ra4 41. Kf6 Rxe4 42. Kg7 Ka3 43. Qf6 Ra4 44. Qa1+ Kb4 45. Qb2 Ra2 46. Qd4+) 40. Rf2 b2 41. Rxb2 Kxb2 42. Ke6 Kc3 43. Ke7?? { (4.99 → -55.67) Blunder. Kd6 was best. } (43. Kd6) 43... Kd4 44. d8=Q+ Rxd8 45. Kxd8 Kxe4 46. Ke7 Kf5 47. Kf7 Kxg5 48. Kg7 h5 { White resigns. } 0-1'''

def all_games():
    f = open("/Users/avenis/Downloads/lichess_db_standard_rated_2014-09.pgn")
    while True:
        game = chess.pgn.read_game(f)
        if game is None:
            return  # end of file

        yield game


game = chess.pgn.read_game(io.StringIO(pgn_string))


ParsedGame = collections.namedtuple('ParsedGame', 'white_elo black_elo states actions')

def parse_game(game):
    states = []
    actions = []
    for move in game.mainline():
        states.append(move.board())
        actions.append(move.move)

    return ParsedGame(
        white_elo=int(game.headers['WhiteElo']),
        black_elo=int(game.headers['BlackElo']),
        states=states,
        actions=actions,
    )

parse_game(game)














