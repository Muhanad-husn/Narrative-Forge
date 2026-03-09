import sys

from ttsgpt.cli import main
from ttsgpt.tts import validate_transcript_coverage


if __name__ == "__main__":
    raise SystemExit(main(["tts", *sys.argv[1:]]))
