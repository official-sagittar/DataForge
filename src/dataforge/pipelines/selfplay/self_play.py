import subprocess
from pathlib import Path
from datetime import datetime

def self_play(
    tournament_runner_path: str,
    engine1_path: str,
    engine2_path: str,
    opening_book_path: str,
    opening_book_fmt: str,
    tc: str,
    rounds: int,
    output_dir: Path,
    concurrency: int = 4,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pgn_path = output_dir / f"{timestamp}.pgn"

    cmd = (
        f'"{tournament_runner_path}" '
        f'-engine cmd="{engine1_path}" name=engine1 '
        f'-engine cmd="{engine2_path}" name=engine2 '
        f'-openings file="{opening_book_path}" format={opening_book_fmt} order=random '
        f'-each tc={tc} option.Hash=64 -rounds {rounds} -games 1 '
        f'-resign movecount=3 score=400 twosided=true '
        f'-draw movenumber=40 movecount=8 score=10 '
        f'-concurrency {concurrency} '
	f'-recover '
        f'-pgnout file="{pgn_path}"'
    )

    print("Running:", cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)
        raise RuntimeError(f"Tournament Runner failed with code {result.returncode}")

    print("Self-play completed! PGN:", pgn_path)
    return str(pgn_path)
