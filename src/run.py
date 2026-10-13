"""Command-line entry point for compact WiTra training and evaluation.

New runs record the source hashes of this renamed package. Evaluate older
runs with their original source files to preserve the integrity checks.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import trainer as base

_resolve_reference = base.resolve_config
_source_hashes_reference = base.source_hashes


def resolve_compact_config(args, metadata):
    config = _resolve_reference(args, metadata)
    values = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    ratio = values.get("mlp_ratio", 4)
    if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio < 1:
        raise ValueError("mlp_ratio must be a positive integer")
    config["model"]["mlp_ratio"] = ratio
    return config


def compact_source_hashes():
    hashes = _source_hashes_reference()
    hashes["run.py"] = base.sha256_file(Path(__file__).resolve())
    # Bind the exploratory sweep implementation to the new run as well.
    hashes["checkpoint_sweep.py"] = base.sha256_file(base.ROOT / "checkpoint_sweep.py")
    return hashes


def install_compact_hooks():
    # The original runner's functions look up these two helpers at runtime.
    # Only this process is affected; no original source file is changed.
    base.resolve_config = resolve_compact_config
    base.source_hashes = compact_source_hashes


def parser():
    result = base.parser()
    commands = next(action for action in result._actions
                    if isinstance(action, argparse._SubParsersAction))
    commands.choices["train"].add_argument(
        "--mlp-ratio", type=int, default=4,
        help="FFN hidden width / embed_dim (positive integer; reference is 4)")
    sweep = commands.add_parser("sweep", help="Exploratory test of the retained MAIN checkpoints")
    sweep.add_argument("--run-dir", required=True)
    sweep.add_argument("--output", required=True)
    sweep.add_argument("--device", help="CPU/CUDA override; saved precision is preserved")
    return result


def main():
    args = parser().parse_args()
    install_compact_hooks()
    if args.command == "sweep":
        from checkpoint_sweep import sweep
        return sweep(args.run_dir, args.output, args.device)
    return {"train": base.train, "evaluate": base.evaluate,
            "aggregate": base.aggregate}[args.command](args)


if __name__ == "__main__":
    main()
