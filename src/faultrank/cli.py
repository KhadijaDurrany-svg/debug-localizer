"""Command-line interface for faultrank.

Subcommands
-----------
* ``bench`` -- run the full benchmark; writes ``benchmarks/results.json`` and
  ``benchmarks/manifest.json`` unless ``--no-write`` is given.
* ``gen``   -- emit one synthetic spectrum as JSON (for ``rank`` or demos).
* ``rank``  -- localize a spectrum JSON with a chosen formula / the heuristic.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from faultrank import __version__
from faultrank.benchmark import run_benchmark
from faultrank.formulas import FORMULAS
from faultrank.generate import GenConfig, generate_dataset
from faultrank.heuristic import coverage_aware
from faultrank.ranking import rank_order
from faultrank.spectrum import Spectrum


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def _spectrum_to_dict(spectrum: Spectrum) -> dict[str, Any]:
    return {
        "coverage": spectrum.coverage.astype(int).tolist(),
        "outcomes": spectrum.outcomes.astype(int).tolist(),
        "faults": list(spectrum.faults),
        "element_names": list(spectrum.element_names or ()),
    }


def _spectrum_from_dict(data: dict[str, Any]) -> Spectrum:
    cov = np.asarray(data["coverage"], dtype=bool)
    outcomes = np.asarray(data["outcomes"], dtype=bool)
    faults = tuple(int(f) for f in data["faults"])
    names = data.get("element_names") or None
    return Spectrum(
        coverage=cov,
        outcomes=outcomes,
        faults=faults,
        element_names=tuple(names) if names else None,
    )


def _cmd_bench(args: argparse.Namespace) -> int:
    cfg = GenConfig(
        n_programs=args.programs,
        n_tests=args.tests,
        label_noise=args.label_noise,
    )
    result = run_benchmark(
        cfg, seed=args.seed, n_boot=args.boot, n_perm=args.perm
    )

    print(f"faultrank {__version__} benchmark")
    print(f"  programs={cfg.n_programs} tests={cfg.n_tests} seed={args.seed}")
    print(f"  dataset checksum: {result['dataset_checksum'][:16]}")
    print("\n  method          MRR     hit@1   hit@5   wasted   (95% CI on MRR)")
    for row in result["leaderboard"]:
        m = row["method"]
        s = result["methods"][m]["metrics"]
        tag = "*" if result["methods"][m]["is_baseline"] else " "
        print(
            f"  {tag}{m:<13} "
            f"{s['reciprocal_rank']['mean']:.3f}  "
            f"{s['hit1']['mean']:.3f}  "
            f"{s['hit5']['mean']:.3f}  "
            f"{s['wasted_effort']['mean']:6.2f}  "
            f"[{s['reciprocal_rank']['ci_lo']:.3f}, {s['reciprocal_rank']['ci_hi']:.3f}]"
        )
    print("  (* = baseline)")
    print("\n  paired permutation tests (coverage_aware vs X, on reciprocal rank):")
    for comp in result["comparisons"]:
        print(
            f"    vs {comp['versus']:<10} "
            f"diff={comp['mean_diff']:+.4f}  p={comp['p_value']:.4f}"
        )

    if args.no_write:
        print("\n  --no-write: results not persisted.")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    manifest_path = out_dir / "manifest.json"
    results_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    manifest = {
        "tool": "faultrank",
        "version": __version__,
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_checksum": result["dataset_checksum"],
        "config": result["config"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\n  wrote {results_path}")
    print(f"  wrote {manifest_path}")
    return 0


def _cmd_gen(args: argparse.Namespace) -> int:
    cfg = GenConfig(n_programs=1, n_tests=args.tests)
    prog = generate_dataset(cfg, seed=args.seed)[0]
    payload = _spectrum_to_dict(prog.spectrum)
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote spectrum to {args.out} (fault at {prog.spectrum.faults})")
    else:
        print(text)
    return 0


def _cmd_rank(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.spectrum).read_text())
    spectrum = _spectrum_from_dict(data)
    if args.formula == "coverage_aware":
        scores = coverage_aware(spectrum)
    else:
        scores = FORMULAS[args.formula](spectrum.counts())
    order = rank_order(scores)
    names = spectrum.element_names or tuple(f"e{i}" for i in range(spectrum.n_elements))
    faults = set(spectrum.faults)
    print(f"ranking by {args.formula} (fault(s): {sorted(faults)})")
    for pos, idx in enumerate(order[: args.top], start=1):
        mark = " <== FAULT" if idx in faults else ""
        print(f"  {pos:>3}. {names[idx]:<8} score={scores[idx]:.4f}{mark}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="faultrank",
        description="Offline spectrum-based fault localization and benchmarking.",
    )
    p.add_argument("--version", action="version", version=f"faultrank {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bench", help="run the benchmark over synthetic programs")
    b.add_argument("--programs", type=int, default=200)
    b.add_argument("--tests", type=int, default=100)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    b.add_argument("--perm", type=int, default=10000, help="permutations")
    b.add_argument("--label-noise", type=float, default=0.0)
    b.add_argument("--no-write", action="store_true", help="do not persist results")
    b.add_argument("--out", default="benchmarks", help="output directory")
    b.set_defaults(func=_cmd_bench)

    g = sub.add_parser("gen", help="emit one synthetic spectrum as JSON")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--tests", type=int, default=100)
    g.add_argument("--out", default=None, help="file to write (default: stdout)")
    g.set_defaults(func=_cmd_gen)

    r = sub.add_parser("rank", help="localize a spectrum JSON file")
    r.add_argument("spectrum", help="path to a spectrum JSON (see `gen`)")
    r.add_argument(
        "--formula",
        default="coverage_aware",
        choices=[*FORMULAS.keys(), "coverage_aware"],
    )
    r.add_argument("--top", type=int, default=10, help="rows to display")
    r.set_defaults(func=_cmd_rank)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
