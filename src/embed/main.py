import argparse

from src.embed.embed import embed_all
from src.embed.index import build_index, sanity_check


def main() -> None:
    parser = argparse.ArgumentParser(description="Завдання 2 і 3: embeddings і Chroma-індекс")
    parser.add_argument("--stage", choices=["embed", "index", "all"], default="all")
    args = parser.parse_args()

    if args.stage in ("embed", "all"):
        embed_all()
    if args.stage in ("index", "all"):
        build_index()
        sanity_check()


if __name__ == "__main__":
    main()
