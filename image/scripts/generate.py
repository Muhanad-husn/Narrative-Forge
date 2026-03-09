import sys

from ttsgpt.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["image", "generate", *sys.argv[1:]]))
