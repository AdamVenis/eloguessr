run_engine.pyimport asyncio
import chess
import chess.engine

WEIGHTS_FILE = '/Users/avenis/personal/src/3p/maia-chess/maia_weights/maia-1100.pb.gz'

async def main() -> None:
    transport, engine = await chess.engine.popen_uci(["/usr/local/bin/lc0", f"--weights={WEIGHTS_FILE}"])

    board = chess.Board()
    print('start one')
    while not board.is_game_over():
        result = await engine.play(board, chess.engine.Limit(time=0.1, nodes=1))
        print(board)
        print(result.move)
        board.push(result.move)


    breakpoint()
    await engine.quit()

asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
asyncio.run(main())
