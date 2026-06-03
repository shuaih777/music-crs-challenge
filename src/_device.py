"""Device-selection helper.

The retrieval scripts in this repo run end-to-end on CPU. If `torch` is
installed and a CUDA device is visible, the heavy linear-algebra steps
(BM25 score lookup, dense matmul) move to GPU automatically — typically
5-25x faster on a single A100 / 4090.

Usage:

    from _device import get_device, has_torch, to_torch, from_torch

    dev = get_device()                  # 'cuda' or 'cpu' or 'mps'
    if has_torch():
        import torch
        t = to_torch(numpy_arr, dev)    # numpy -> torch on the right device
        out = from_torch(t)             # back to numpy

    print(get_device(verbose=True))
"""

from __future__ import annotations

import os
from typing import Any


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def get_device(prefer: str | None = None, verbose: bool = False) -> str:
    """Return the best available compute device as a string.

    Args:
        prefer: If 'cuda' / 'cpu' / 'mps' / None. None = autodetect.
        verbose: If True, print a one-line summary on first call.

    Returns:
        One of 'cuda', 'mps', 'cpu'. If torch is unavailable always 'cpu'.
    """
    env_override = os.environ.get("MUSIC_CRS_DEVICE")
    if env_override:
        prefer = env_override

    if not has_torch():
        if verbose:
            print("[device] torch not installed -> using cpu (numpy only)")
        return "cpu"

    import torch
    if prefer == "cpu":
        if verbose:
            print("[device] forced cpu")
        return "cpu"
    if prefer == "cuda" and torch.cuda.is_available():
        if verbose:
            print(f"[device] using cuda: {torch.cuda.get_device_name(0)}")
        return "cuda"
    if prefer == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        if verbose:
            print("[device] using mps (Apple Silicon)")
        return "mps"

    # autodetect
    if torch.cuda.is_available():
        if verbose:
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            print(f"[device] auto -> cuda ({n} device(s); {name})")
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        if verbose:
            print("[device] auto -> mps")
        return "mps"
    if verbose:
        print("[device] auto -> cpu (no GPU visible)")
    return "cpu"


def to_torch(arr, device: str = "cpu", dtype: Any = None):
    """numpy -> torch on `device`, optionally casting dtype."""
    if not has_torch():
        raise RuntimeError("torch is not installed")
    import torch
    t = torch.as_tensor(arr)
    if dtype is not None:
        t = t.to(dtype)
    return t.to(device)


def from_torch(t):
    """torch -> numpy (cpu)."""
    return t.detach().cpu().numpy()


if __name__ == "__main__":
    print("torch installed:", has_torch())
    dev = get_device(verbose=True)
    print(f"selected device: {dev}")
    if has_torch() and dev != "cpu":
        import torch
        x = torch.randn(2048, 512, device=dev)
        y = torch.randn(512, device=dev)
        z = x @ y
        print(f"sanity: {x.shape} @ {y.shape} -> {z.shape}")
