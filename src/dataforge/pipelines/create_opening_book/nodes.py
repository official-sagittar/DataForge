import random
from pathlib import Path

import chess


def _parse_seed_position(line: str, line_number: int) -> chess.Board:
    board = chess.Board()
    try:
        board.set_epd(line)
    except ValueError as exc:
        raise ValueError(
            f"Invalid EPD in seed openings at line {line_number}: {line!r}"
        ) from exc

    return board


def create_opening_book(
    seed_openings: str,
    output_dir: str,
    size: int,
    random_seed: int,
) -> str:
    if size <= 0:
        raise ValueError("Opening book size must be greater than 0")

    seed_openings_book = Path(seed_openings)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_lines = [
        (line_number, line.strip())
        for line_number, line in enumerate(
            seed_openings_book.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        if line.strip()
    ]

    if not seed_lines:
        raise ValueError(f"No openings found in {seed_openings_book}")

    rng = random.Random(random_seed)
    generated_positions = []
    generated_position_set = set()
    max_attempts = size * 100
    attempts = 0

    while len(generated_positions) < size and attempts < max_attempts:
        attempts += 1
        line_number, line = rng.choice(seed_lines)
        board = _parse_seed_position(line, line_number)

        for _ in range(5):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            board.push(rng.choice(legal_moves))

        fen = board.fen()
        if fen in generated_position_set:
            continue

        generated_positions.append(fen)
        generated_position_set.add(fen)

    if len(generated_positions) < size:
        raise ValueError(
            "Could not generate "
            f"{size} unique FENs after {max_attempts} attempts. "
            "Try using a larger seed opening file, increasing random plies, "
            "or requesting a smaller size."
        )

    output_path = output_dir / f"opening_book_seed{random_seed}_size{size}.epd"
    output_path.write_text("\n".join(generated_positions) + "\n", encoding="utf-8")

    return str(output_path)
