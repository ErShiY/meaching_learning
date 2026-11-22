import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

try:
    from numpy._core.multiarray import _reconstruct
except ImportError:
    from numpy.core.multiarray import _reconstruct


RELEASE_SHA256 = "f42201232ad35ead24bce136cfd989a56d09db555d1842d1a1cd730e971c64f9"


def reconstruct_legacy_array(*args):
    return _reconstruct(*args)


# NumPy 2 moved the function. PyTorch 2.5 uses these attributes for its allowlist.
reconstruct_legacy_array.__module__ = "numpy.core.multiarray"
reconstruct_legacy_array.__name__ = "_reconstruct"
reconstruct_legacy_array.__qualname__ = "_reconstruct"


def main():
    parser = argparse.ArgumentParser(description="Convert the published Bestfitting fold 0 checkpoint.")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    with args.source.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != RELEASE_SHA256:
            raise ValueError("Source does not match the published Bestfitting checkpoint.")
        source.seek(0)
        # The release stores a float64 NumPy array of scores alongside its tensors.
        allowed = [reconstruct_legacy_array, np.ndarray, np.dtype, type(np.dtype(np.float64))]
        with torch.serialization.safe_globals(allowed):
            released = torch.load(source, map_location="cpu", weights_only=True)["state_dict"]

    state = {
        "encoder." + key.removeprefix("backbone.features."): value
        for key, value in released.items()
        if key.startswith("backbone.features.")
    }
    state.update({
        key: value for key, value in released.items()
        if key.startswith(("bn1.", "fc1.", "bn2.", "logit."))
    })
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("xb") as destination:
        torch.save(state, destination)
    print(f"Converted {len(state)} tensors to {args.destination}")


if __name__ == "__main__":
    main()
