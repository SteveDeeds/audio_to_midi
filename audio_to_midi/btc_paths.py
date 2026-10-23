"""Where the BTC chord weights live (importable without torch)."""

from pathlib import Path

CHECKPOINT = Path.home() / ".cache" / "audio_to_midi" / "btc" / "btc_model_large_voca.pt"
# Pinned commit of https://github.com/jayg996/BTC-ISMIR19 (MIT License).
CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/"
    "2682317be668032e6e4b269ded36adaa2ad57df0/test/btc_model_large_voca.pt"
)
CHECKPOINT_BYTES = 12_229_576
