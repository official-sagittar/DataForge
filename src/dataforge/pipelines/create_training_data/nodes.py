import pandas as pd
from pathlib import Path
from datetime import datetime
from .features import create_features
from .sampling import weighted_sample


def _write_epd(df: pd.DataFrame, output_path: Path) -> None:
    """
    Write training data to an EPD file.

    Format:
        <FEN> ; [WDL]

    Example:
        rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 ; [0.5]
    """
    with output_path.open("w", encoding="utf-8") as f:
        for fen, wdl in zip(df["fen"], df["wdl"]):
            f.write(f"{fen} ; [{wdl}]\n")


def create_training_data(quite_labelled_data_path: str, output_dir: str, size: int) -> str:
    quite_labelled_data_path = Path(quite_labelled_data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read all quite labelled files
    li = []
    for quite_labelled_file in quite_labelled_data_path.glob("*.csv"):
        df = pd.read_csv(quite_labelled_file)
        li.append(df)
    training_data = pd.concat(li, ignore_index=True)

    # Remove duplicate FENs
    training_data.drop_duplicates(subset=['fen'], keep='first', inplace=True)
    print(f"Total Unique FENs = {training_data.shape[0]}")

    # Create features
    create_features(training_data)

    # Tap a sample using Weighted Sampling
    training_data = weighted_sample(training_data, size, n_phase_bins=3, verify=True)

    # Shuffle rows
    training_data = training_data.sample(frac=1).reset_index(drop=True)

    # Write training data as .epd
    training_data = training_data[["fen", "wdl"]]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"Training Data_{timestamp}.epd"
    _write_epd(training_data, output_path)

    return output_path
