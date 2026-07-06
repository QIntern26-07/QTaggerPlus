"""Download CIC-MalMem-2022 from Kaggle into data/cic_malmem/.

Requires a Kaggle API token, generated at kaggle.com/settings/api, saved to
~/.kaggle/access_token (chmod 600) or exported as KAGGLE_API_TOKEN.
"""
import subprocess
import sys
from pathlib import Path

SLUG = "luccagodoy/obfuscated-malware-memory-2022-cic"
DEST = Path("data/cic_malmem")


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", "kaggle", "datasets", "download",
           "-d", SLUG, "-p", str(DEST), "--unzip"]
    print("running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
