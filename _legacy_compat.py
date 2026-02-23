#!/usr/bin/env python
"""Legacy compatibility wrapper for run_inference.py.

Older checkpoints use FullLobPredModel (merging='projected') which lacks
__call_ar__, __call_rnn__, and initialize_carry. This script monkey-patches
the runtime to force merging='padded' — the core weights (message_encoder,
book_encoder, fused_s5, decoder) are shape-compatible since both architectures
concatenate along the feature axis producing (seq_len, 2*d_model) for fused_s5.
The projection layer weights (message_out_proj, book_out_proj) are loaded but
unused.

Run from ${REPO_DIR}/LOBS5/:
    python /path/to/_legacy_compat.py [run_inference.py args...]

Environment variables:
    FORCE_MERGING_PADDED=1   Force merging='padded' (set by batch scripts)
    INFER_INIT_GLOBAL_BSZ=N  Override global_bsz during init (default: 1)
    INFER_DEFAULT_MERGING    Default merging mode when absent (default: padded)
"""

import argparse
import json
import os
import runpy
import sys

# Ensure cwd is on sys.path so that modules like `preproc` (which live in
# LOBS5/) are importable.  When this wrapper is invoked by absolute path,
# Python sets sys.path[0] to the *script's* directory, not the cwd.
_cwd = os.getcwd()
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)

import lob.init_train as _init_train

_orig_parse_args = argparse.ArgumentParser.parse_args
_orig_parse_known_args = argparse.ArgumentParser.parse_known_args
_orig_init_train_state = _init_train.init_train_state
_orig_load_metadata = _init_train.load_metadata
_orig_load_checkpoint = _init_train.load_checkpoint

_force_padded = os.environ.get("FORCE_MERGING_PADDED", "0") == "1"


def _ensure_global_bsz(ns):
    if not hasattr(ns, "global_bsz"):
        bsz = getattr(ns, "bsz", None)
        if bsz is None:
            bsz = getattr(ns, "batch_size", 1)
        num_devices = getattr(ns, "num_devices", 1)
        ns.global_bsz = max(1, int(bsz) * int(num_devices))
    if not hasattr(ns, "merging"):
        ns.merging = os.environ.get("INFER_DEFAULT_MERGING", "padded")
        print(f"[*] Runtime patch: merging missing in metadata, defaulting to {ns.merging}")
    # Legacy compat: force projected -> padded (weights are shape-compatible)
    if _force_padded and getattr(ns, "merging", "") != "padded":
        print(f"[*] Runtime patch: overriding merging='{ns.merging}' -> 'padded' (legacy compat)")
        ns.merging = "padded"
    return ns


def _parse_args_with_global_bsz(self, *args, **kwargs):
    ns = _orig_parse_args(self, *args, **kwargs)
    return _ensure_global_bsz(ns)


def _parse_known_args_with_global_bsz(self, *args, **kwargs):
    ns, extras = _orig_parse_known_args(self, *args, **kwargs)
    return _ensure_global_bsz(ns), extras


def _load_metadata_robust(path):
    """Fallback metadata loader that tolerates trailing garbage after JSON."""
    from argparse import Namespace
    json_path = path + '/metadata/_ROOT_METADATA'
    with open(json_path, 'r') as f:
        raw = f.read()
    # Brace-depth parser: find the end of the first complete JSON object
    depth, end = 0, 0
    for i, c in enumerate(raw):
        if c == '{': depth += 1
        elif c == '}': depth -= 1
        if depth == 0 and i > 0:
            end = i + 1
            break
    metadata = json.loads(raw[:end] if end > 0 else raw)
    if 'custom' in metadata:
        return Namespace(**metadata['custom'])
    elif 'custom_metadata' in metadata:
        return Namespace(**metadata['custom_metadata'])
    return Namespace(**metadata)


def _load_metadata_with_global_bsz(*args, **kwargs):
    try:
        ns = _orig_load_metadata(*args, **kwargs)
    except Exception as e:
        print(f"[*] Runtime patch: load_metadata failed ({e}), using robust fallback")
        ns = _load_metadata_robust(*args, **kwargs)
    ns = _ensure_global_bsz(ns)
    ns.global_bsz = max(1, int(os.environ.get("INFER_INIT_GLOBAL_BSZ", "1")))
    if not hasattr(ns, "token_mode"):
        ns.token_mode = 22
        print("[*] Runtime patch: token_mode missing in metadata, defaulting to legacy token_mode=22")
    print(f"[*] Runtime patch: forcing inference init global_bsz={ns.global_bsz}")
    return ns


def _load_checkpoint_compat(state, path, step=None, train=True, partial_restore=False):
    """Wrapper that falls back to _load_ocdbt_direct on any restore error.

    When forcing merging='padded' on a 'projected' checkpoint, the param tree
    has extra keys (message_out_proj, book_out_proj).  StandardRestore may
    reject the mismatch, so we catch broadly and fall back to the direct
    tensorstore loader which is structure-agnostic.
    """
    try:
        return _orig_load_checkpoint(state, path, step=step, train=train,
                                     partial_restore=partial_restore)
    except Exception as first_err:
        if not _force_padded:
            raise
        print(f"[*] Runtime patch: load_checkpoint failed ({first_err})")
        print(f"[*] Runtime patch: retrying with _load_ocdbt_direct (legacy compat)")
        import jax
        import orbax.checkpoint as ocp

        if step is None:
            mngr = ocp.CheckpointManager(os.path.abspath(path),
                                          item_names=('state', 'metadata'))
            step = mngr.latest_step()
        params = _init_train._load_ocdbt_direct(os.path.abspath(path), step)
        loaded_state = state.replace(params=params, step=step)

        # Load per-step metadata
        metadata_path = os.path.join(os.path.abspath(path), str(step),
                                     "metadata", "metadata")
        ckpt = {}
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                raw = f.read()
            try:
                ckpt = json.loads(raw)
            except json.JSONDecodeError:
                # Tolerate trailing garbage (same as _load_metadata_robust)
                depth, end = 0, 0
                for i, c in enumerate(raw):
                    if c == '{': depth += 1
                    elif c == '}': depth -= 1
                    if depth == 0 and i > 0:
                        end = i + 1
                        break
                ckpt = json.loads(raw[:end]) if end > 0 else {}

        if train:
            from lob.sharding_utils import get_global_mesh, create_state_shardings
            mesh = get_global_mesh()
            state_shardings = create_state_shardings(loaded_state, mesh)
            ckpt['model'] = jax.device_put(loaded_state, state_shardings)
        else:
            ckpt['model'] = loaded_state
        return ckpt


def _init_train_state_with_global_bsz(args, *iargs, **ikwargs):
    args = _ensure_global_bsz(args)
    return _orig_init_train_state(args, *iargs, **ikwargs)


# Install monkey-patches
argparse.ArgumentParser.parse_args = _parse_args_with_global_bsz
argparse.ArgumentParser.parse_known_args = _parse_known_args_with_global_bsz
_init_train.load_metadata = _load_metadata_with_global_bsz
_init_train.init_train_state = _init_train_state_with_global_bsz
_init_train.load_checkpoint = _load_checkpoint_compat

# Run the actual inference script
runpy.run_path("run_inference.py", run_name="__main__")
