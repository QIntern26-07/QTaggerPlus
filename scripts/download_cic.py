"""Download CIC-MalMem-2022 from Kaggle into data/cic_malmem/.

Requires the Kaggle API token at ~/.kaggle/kaggle.json (chmod 600).
"""
import subprocess
import sys
from pathlib import Path

SLUG = "joebeachcapital/cic-malmem-2022"  # verify slug is live before first run
DEST = Path("data/cic_malmem")


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", "kaggle", "datasets", "download",
           "-d", SLUG, "-p", str(DEST), "--unzip"]
    print("running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
