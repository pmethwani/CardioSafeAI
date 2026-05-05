"""Environment detection + project paths.

Single source of truth for "are we in Colab" and "where does PROJECT live".
Imported by the notebook and every src/ module that touches disk.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _detect_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


IN_COLAB: bool = _detect_colab()


def project_root() -> Path:
    """Project root. In Colab: /content/drive/MyDrive/cardiosafeai-mm.
    Locally: the directory containing this file's parent.parent."""
    env = os.environ.get("CARDIOSAFE_PROJECT")
    if env:
        return Path(env).expanduser().resolve()
    if IN_COLAB:
        return Path("/content/drive/MyDrive/cardiosafeai-mm")
    return Path(__file__).resolve().parent.parent


def mount_drive_if_colab() -> None:
    """No-op outside Colab. Mounts Drive in Colab so PROJECT is writable."""
    if not IN_COLAB:
        return
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive", force_remount=True)


def ensure_project_dirs() -> Path:
    root = project_root()
    for sub in ("data", "data/chemprop", "results", "figures", "paper",
                "src/chem", "src/physio", "src/models", "src/utils"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def device():
    """Return the best available torch device string. Lazy import — torch is heavy."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def add_src_to_path() -> None:
    """Make `import src.x.y` work from a notebook in the project root."""
    root = str(project_root())
    if root not in sys.path:
        sys.path.insert(0, root)
