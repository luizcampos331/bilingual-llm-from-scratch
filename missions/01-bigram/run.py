"""Train the Mission 1 bigram model on makemore's names.txt and print its numbers.

Usage (from the repo root):
    uv run python missions/01-bigram/run.py
"""
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src"))

from bigram import Bigram  # noqa: E402

SEED = 2147483647
SPLIT_SEED = 42
TRAIN_FRACTION = 0.8
NAMES_TO_GENERATE = 10


def load_names(path: Path) -> list[str]:
    """Read one name per line from path, skipping blank lines."""
    return path.read_text().split()


def main() -> None:
    """Report corpus size, loss on all names, loss on an 80/20 split, and a few generated names."""
    names = load_names(HERE / "data" / "names.txt")
    model = Bigram(names)
    print(f"names:          {len(names)}")
    print(f"distinct pairs: {len(model.bigram_count_pairs)}")
    print(f"next chars:     {len(model.next_chars)}")
    print(f"loss (all):     {model.loss(names):.4f}")

    shuffled = names[:]
    random.Random(SPLIT_SEED).shuffle(shuffled)
    cut = int(TRAIN_FRACTION * len(shuffled))
    train, val = shuffled[:cut], shuffled[cut:]
    split_model = Bigram(train)
    print(f"loss (train):   {split_model.loss(train):.4f}  ({len(train)} names)")
    print(f"loss (val):     {split_model.loss(val):.4f}  ({len(val)} names)")

    rng = random.Random(SEED)
    print(f"\ngenerated (seed {SEED}):")
    for _ in range(NAMES_TO_GENERATE):
        print(f"  {model.generate(rng)}")


if __name__ == "__main__":
    main()
