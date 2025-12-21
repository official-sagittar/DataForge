import chess
from typing import Tuple


def is_quite(board: chess.Board) -> bool:
    if board.is_check():
        return False

    return not any(board.is_capture(m) for m in board.legal_moves)


def quiesce_to_quiet(
    board: chess.Board,
    max_depth: int = 10,
) -> Tuple[bool, chess.Board]:
    """
    Fully convert a position to a quiet position using strict quiescence search.

    Guarantees:
    - If success == True, returned board is strictly quiet.
    - If success == False, no quiet position was reachable within max_depth.

    Parameters
    ----------
    board : chess.Board
        Input position
    max_depth : int
        Maximum quiescence depth (safety bound)

    Returns
    -------
    (success, quiet_board)
    """

    original = board.copy(stack=False)
    board = board.copy(stack=False)

    def qsearch(b: chess.Board, depth: int) -> Tuple[bool, chess.Board]:
        # Success: strictly quiet
        if is_quite(b):
            return True, b

        # Failure: depth exhausted
        if depth == 0:
            return False, b

        # Move generation rules (engine-style):
        # - If in check: ALL legal moves (must escape check)
        # - Else: CAPTURES ONLY
        if b.is_check():
            moves = list(b.legal_moves)
        else:
            moves = [m for m in b.legal_moves if b.is_capture(m)]

        if not moves:
            return False, b

        for move in moves:
            b.push(move)
            success, result = qsearch(b, depth - 1)
            b.pop()

            if success:
                return True, result

        return False, b

    success, quiet = qsearch(board, max_depth)
    return (success, quiet if success else original)
