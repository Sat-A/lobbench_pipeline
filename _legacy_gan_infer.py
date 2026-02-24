#!/usr/bin/env python
"""Legacy inference using LOBS5-gan codebase.

For checkpoints that are incompatible with the new inference pipeline
(e.g., models trained with the original LOBS5-gan codebase using
BatchFullLobPredModel/projected merging without __call_ar__ support).

Uses the LOBS5-gan codebase for model loading and token-by-token generation
via full S5 forward passes (slower but compatible with any S5 model).

Run from the LOBS5-gan directory:
    python /path/to/pipeline/_legacy_gan_infer.py --stock GOOG --ckpt_path ... [OPTIONS]

Or from anywhere with LOBS5_GAN_DIR set:
    LOBS5_GAN_DIR=/path/to/LOBS5-gan python pipeline/_legacy_gan_infer.py ...

Environment variables:
    LOBS5_GAN_DIR    Path to LOBS5-gan directory (default: sibling of pipeline/)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from time import time

# === Resolve LOBS5-gan directory ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
LOBS5_GAN_DIR = os.environ.get(
    'LOBS5_GAN_DIR',
    os.path.join(REPO_ROOT, 'LOBS5-gan')
)

if not os.path.isdir(LOBS5_GAN_DIR):
    print(f"ERROR: LOBS5-gan directory not found: {LOBS5_GAN_DIR}")
    print("  Set LOBS5_GAN_DIR or ensure LOBS5-gan/ is a sibling of pipeline/")
    sys.exit(1)

# Add LOBS5-gan root and lob/ to path (legacy code uses relative imports)
for p in [LOBS5_GAN_DIR, os.path.join(LOBS5_GAN_DIR, 'lob')]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Add AlphaTrade submodule (needed for OrderBook simulator)
# Search order: LOBS5-gan's own submodule, sibling dir, deployed pipeline copy
for candidate in [
    os.path.join(LOBS5_GAN_DIR, 'AlphaTrade'),
    os.path.join(os.path.dirname(LOBS5_GAN_DIR), 'AlphaTrade'),
    '/projects/s5e/lob_pipeline/LOBS5/AlphaTrade',
]:
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

# Set JAX memory defaults (can be overridden by env)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

# ============================================================
# Imports (all from LOBS5-gan codebase)
# ============================================================
import jax
import jax.numpy as jnp
import numpy as onp
from tqdm import tqdm

from lob.encoding import Vocab, Message_Tokenizer
from lob.init_train import (
    init_train_state,
    load_checkpoint,
    load_args_from_checkpoint,
    deduplicate_trainstate,
)
import lob.inference_no_errcorr as inference


# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(
    description="Legacy inference via LOBS5-gan codebase"
)
parser.add_argument('--stock', type=str, required=True,
                    help='Stock symbol (e.g., GOOG)')
parser.add_argument('--ckpt_path', type=str, required=True,
                    help='Path to checkpoint directory')
parser.add_argument('--data_dir', type=str, required=True,
                    help='Path to preprocessed data directory (raw .npy format)')
parser.add_argument('--save_dir', type=str, required=True,
                    help='Directory to save inference results')
parser.add_argument('--n_cond_msgs', type=int, default=500,
                    help='Number of conditioning messages')
parser.add_argument('--n_gen_msgs', type=int, default=500,
                    help='Number of messages to generate')
parser.add_argument('--n_sequences', type=int, default=1024,
                    help='Total sequences to generate (before rank splitting)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size for inference')
parser.add_argument('--checkpoint_step', type=int, default=None,
                    help='Checkpoint step to load (default: latest)')
parser.add_argument('--test_split', type=float, default=1.0,
                    help='Fraction of data to use (default: 1.0 = all)')
parser.add_argument('--rank', type=int, default=0,
                    help='Rank of this process (0-indexed)')
parser.add_argument('--world_size', type=int, default=1,
                    help='Total number of processes')
parser.add_argument('--sample_indices_file', type=str, default=None,
                    help='File with dataset indices (one per line) for HF-matched mode')


def load_checkpoint_config(ckpt_path, step=None):
    """Load checkpoint config, trying both old and new Orbax formats."""
    # Try 1: Old format via flax.training.checkpoints (has 'config' key)
    try:
        args = load_args_from_checkpoint(ckpt_path, step=step)
        print(f"[*] Loaded config via load_args_from_checkpoint (old format)")
        return args
    except Exception as e1:
        print(f"[*] load_args_from_checkpoint failed: {e1}")

    # Try 2: New Composite format via JSON metadata
    try:
        from lob.init_train import load_metadata
        args = load_metadata(ckpt_path)
        print(f"[*] Loaded config via load_metadata (new format)")
        return args
    except Exception as e2:
        print(f"[*] load_metadata failed: {e2}")

    # Try 3: Read raw JSON from metadata/_ROOT_METADATA
    try:
        from argparse import Namespace
        json_path = os.path.join(ckpt_path, 'metadata', '_ROOT_METADATA')
        with open(json_path, 'r') as f:
            raw = f.read()
        # Brace-depth parser for potentially corrupted JSON
        depth, end = 0, 0
        for i, c in enumerate(raw):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            if depth == 0 and i > 0:
                end = i + 1
                break
        metadata = json.loads(raw[:end] if end > 0 else raw)
        # Unwrap 'custom' key if present
        if 'custom' in metadata and isinstance(metadata['custom'], dict):
            metadata = metadata['custom']
        if 'config' in metadata and isinstance(metadata['config'], dict):
            metadata = metadata['config']
        print(f"[*] Loaded config from raw metadata JSON")
        return Namespace(**metadata)
    except Exception as e3:
        print(f"[*] Raw metadata read failed: {e3}")

    raise RuntimeError(
        f"Cannot load checkpoint config from {ckpt_path}. "
        f"Tried load_args_from_checkpoint, load_metadata, and raw JSON."
    )


def main():
    args = parser.parse_args()

    rank = args.rank
    world_size = args.world_size

    # ============================================================
    # Parameters
    # ============================================================
    n_messages = args.n_cond_msgs
    n_gen_msgs = args.n_gen_msgs
    n_vol_series = 500
    tick_size = 100
    sample_top_n = -1  # sample from full distribution

    # Legacy uses 22-token mode
    v = Vocab()
    n_classes = len(v)
    seq_len = n_messages * Message_Tokenizer.MSG_LEN
    book_dim = 501
    book_seq_len = n_messages

    # Per-rank RNG for generation diversity
    rng = jax.random.PRNGKey(42 + rank)
    rng, rng_ = jax.random.split(rng)

    # ============================================================
    # Load checkpoint
    # ============================================================
    print(f"[Rank {rank}/{world_size}] Loading checkpoint from {args.ckpt_path}")
    ckpt_args = load_checkpoint_config(args.ckpt_path, step=args.checkpoint_step)

    # Override for single-GPU inference
    ckpt_args.bsz = 1
    ckpt_args.num_devices = 1

    print(f"[Rank {rank}/{world_size}] Initializing model...")
    print(f"  d_model={ckpt_args.d_model}, n_layers={getattr(ckpt_args, 'n_layers', '?')}")
    print(f"  use_book_data={getattr(ckpt_args, 'use_book_data', '?')}")

    new_train_state, model_cls = init_train_state(
        ckpt_args,
        n_classes=n_classes,
        seq_len=seq_len,
        book_dim=book_dim,
        book_seq_len=book_seq_len,
    )

    print(f"[Rank {rank}/{world_size}] Loading checkpoint weights...")
    ckpt = load_checkpoint(
        new_train_state,
        args.ckpt_path,
        step=args.checkpoint_step,
        train=False,
    )
    state = ckpt['model']

    # Deduplicate if trained on multiple GPUs (replicated params)
    try:
        # Quick test: if params are nested device arrays, deduplicate
        emb = state.params['message_encoder']['encoder']['embedding']
        if emb.ndim > 2:
            print(f"[*] Deduplicating multi-GPU params (shape {emb.shape})")
            state = deduplicate_trainstate(state)
    except (KeyError, AttributeError):
        pass

    model = model_cls(training=False, step_rescale=1.0)
    batchnorm = ckpt_args.batchnorm

    # ============================================================
    # Load dataset
    # ============================================================
    print(f"[Rank {rank}/{world_size}] Loading dataset from {args.data_dir}")
    ds = inference.get_dataset(
        args.data_dir,
        n_messages,
        n_gen_msgs,
    )
    print(f"[Rank {rank}/{world_size}] Dataset size: {len(ds)}")

    # ============================================================
    # Compute rank's indices
    # ============================================================
    if args.sample_indices_file is not None:
        with open(args.sample_indices_file, 'r') as f:
            all_indices = [int(line.strip()) for line in f if line.strip()]
        print(f"[Rank {rank}/{world_size}] Loaded {len(all_indices)} indices from {args.sample_indices_file}")
    else:
        rng_idx = jax.random.PRNGKey(42)
        all_indices = jax.random.choice(
            rng_idx,
            jnp.arange(len(ds), dtype=jnp.int32),
            shape=(args.n_sequences,),
            replace=False
        ).tolist()

    # Interleaved split across ranks
    rank_indices = all_indices[rank::world_size]
    n_samples = len(rank_indices)

    batch_size = args.batch_size

    # Pad to batch_size multiple
    if n_samples > 0 and n_samples % batch_size != 0:
        n_padded = ((n_samples + batch_size - 1) // batch_size) * batch_size
        n_pad = n_padded - n_samples
        rank_indices = (rank_indices * (n_padded // n_samples + 1))[:n_padded]
        print(f"[Rank {rank}/{world_size}] Padded {n_pad} indices ({n_padded} total)")

    # Batch the indices
    batched_indices = [
        rank_indices[i:i + batch_size]
        for i in range(0, len(rank_indices), batch_size)
    ]

    print(f"[Rank {rank}/{world_size}] Processing {n_samples} sequences in {len(batched_indices)} batches")
    print(f"[Rank {rank}/{world_size}] GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    if rank_indices:
        print(f"[Rank {rank}/{world_size}] First indices: {rank_indices[:5]}")

    # ============================================================
    # Create output directories
    # ============================================================
    save_dir = args.save_dir
    Path(f'{save_dir}/data_cond').mkdir(parents=True, exist_ok=True)
    Path(f'{save_dir}/data_real').mkdir(parents=True, exist_ok=True)
    Path(f'{save_dir}/data_gen').mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Inference loop (replicated from LOBS5-gan sample_new with
    # index support and rank sharding)
    # ============================================================
    from preproc import transform_L2_state

    transform_L2_state_batch = jax.jit(
        jax.vmap(transform_L2_state, in_axes=(0, None, None)),
        static_argnums=(1, 2)
    )

    start = time()

    for batch_i in tqdm(batched_indices, desc=f"Rank {rank}"):
        print(f'[Rank {rank}] BATCH {batch_i}')

        # Load data
        m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
        m_seq = jnp.array(m_seq)
        b_seq_pv = jnp.array(b_seq_pv)
        msg_seq_raw = jnp.array(msg_seq_raw)
        book_l2_init = jnp.array(book_l2_init)

        # Transform book to volume image representation
        b_seq = transform_L2_state_batch(b_seq_pv, n_vol_series, tick_size)

        # Split into input and eval
        m_seq_inp = m_seq[:, :seq_len]
        b_seq_inp = b_seq[:, :n_messages]

        # True L2 data (remove price change column)
        b_seq_pv_inp = onp.array(b_seq_pv[:, :n_messages, 1:])
        b_seq_pv_eval = onp.array(b_seq_pv[:, n_messages:, 1:])

        # Raw LOBSTER data
        m_seq_raw_inp = msg_seq_raw[:, :n_messages]
        m_seq_raw_eval = msg_seq_raw[:, n_messages:]

        # Initialize simulators (batched)
        sim_init, sim_states_init = inference.get_sims_vmap(
            book_l2_init,
            m_seq_raw_inp,
        )

        # Generate messages (batched)
        m_seq_gen, b_seq_gen, msgs_decoded, l2_book_states, num_errors = \
            inference.generate_batched(
                sim_init,
                state,
                model,
                batchnorm,
                v.ENCODING,
                sample_top_n,
                tick_size,
                m_seq_inp,
                b_seq_inp,
                n_gen_msgs,
                sim_states_init,
                jax.random.split(rng_, len(batch_i)),
            )
        rng, rng_ = jax.random.split(rng)
        print(f'[Rank {rank}] num_errors: {num_errors}')

        # Save output CSVs
        for i, cond_msg, cond_book, real_msg, real_book, gen_msg, gen_book \
            in zip(
                batch_i,
                m_seq_raw_inp, b_seq_pv_inp,
                m_seq_raw_eval, b_seq_pv_eval,
                msgs_decoded, l2_book_states,
            ):
            date = ds.get_date(i)
            stock = args.stock

            # Conditioning data
            inference.msg_to_lobster_format(cond_msg).to_csv(
                f'{save_dir}/data_cond/{stock}_{date}_message_real_id_{i}.csv',
                index=False, header=False,
            )
            inference.book_to_lobster_format(cond_book).to_csv(
                f'{save_dir}/data_cond/{stock}_{date}_orderbook_real_id_{i}.csv',
                index=False, header=False,
            )

            # Real data
            inference.msg_to_lobster_format(real_msg).to_csv(
                f'{save_dir}/data_real/{stock}_{date}_message_real_id_{i}.csv',
                index=False, header=False,
            )
            inference.book_to_lobster_format(real_book).to_csv(
                f'{save_dir}/data_real/{stock}_{date}_orderbook_real_id_{i}.csv',
                index=False, header=False,
            )

            # Generated data
            inference.msg_to_lobster_format(gen_msg).to_csv(
                f'{save_dir}/data_gen/{stock}_{date}_message_real_id_{i}_gen_id_0.csv',
                index=False, header=False,
            )
            inference.book_to_lobster_format(gen_book).to_csv(
                f'{save_dir}/data_gen/{stock}_{date}_orderbook_real_id_{i}_gen_id_0.csv',
                index=False, header=False,
            )

    elapsed = time() - start
    print(f"[Rank {rank}/{world_size}] Done: {n_samples} sequences in {elapsed:.1f}s")


if __name__ == '__main__':
    import logging
    fhandler = logging.FileHandler(filename='legacy_gan_inference.log', mode='w')
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fhandler)
    logger.setLevel(logging.WARNING)

    main()
