from __future__ import annotations

import json
from pathlib import Path

from faultrank.cli import main


def test_gen_and_rank(tmp_path, capsys):
    spec_path = tmp_path / "spec.json"
    rc = main(["gen", "--seed", "1", "--tests", "50", "--out", str(spec_path)])
    assert rc == 0
    data = json.loads(spec_path.read_text())
    assert "coverage" in data and "faults" in data

    rc = main(["rank", str(spec_path), "--formula", "ochiai", "--top", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ranking by ochiai" in out


def test_rank_coverage_aware(tmp_path):
    spec_path = tmp_path / "spec.json"
    main(["gen", "--seed", "2", "--out", str(spec_path)])
    assert main(["rank", str(spec_path), "--formula", "coverage_aware"]) == 0


def test_bench_no_write(capsys):
    rc = main(
        ["bench", "--programs", "20", "--tests", "40", "--boot", "100",
         "--perm", "200", "--no-write"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "benchmark" in out
    assert "coverage_aware" in out


def test_bench_writes_outputs(tmp_path):
    out_dir = tmp_path / "bench_out"
    rc = main(
        ["bench", "--programs", "20", "--tests", "40", "--boot", "100",
         "--perm", "200", "--out", str(out_dir)]
    )
    assert rc == 0
    results = json.loads((out_dir / "results.json").read_text())
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert "methods" in results
    assert manifest["dataset_checksum"] == results["dataset_checksum"]
    assert "python" in manifest
    assert "git_sha" in manifest


def test_manifest_git_sha_present(tmp_path):
    out_dir = tmp_path / "o"
    main(["bench", "--programs", "15", "--tests", "30", "--boot", "50",
          "--perm", "100", "--out", str(out_dir)])
    manifest = json.loads((Path(out_dir) / "manifest.json").read_text())
    # either a sha or the documented fallback
    assert isinstance(manifest["git_sha"], str) and manifest["git_sha"]
