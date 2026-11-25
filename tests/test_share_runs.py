"""Sharing runs on GitHub (scripts/share_runs.py): small results for the repository, whole runs as
archives, and nothing that ties a run to the machines it used."""

import importlib.util
import json
import tarfile
from pathlib import Path

spec = importlib.util.spec_from_file_location("share_runs", Path(__file__).resolve().parent.parent / "scripts" / "share_runs.py")
share_runs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(share_runs)

MACHINE = {"host": "lab", "pid": 7, "where": "server", "follower_pid": 9, "follower_host": "mac", "server_host": "10.0.0.5"}


def _run(root: Path) -> Path:
    run = root / "demo"
    files = {
        "config.json": "{}", "metrics.jsonl": '{"event": "start"}\n', "train.log": "trained\n",
        "status.json": json.dumps({"state": "finished", "generation": 2, **MACHINE}),
        "launch.json": json.dumps({"where": "server", "host": "10.0.0.5", "folder": "/home/someone/cz"}),
        "server.json": "{}", "server_run.log": "Started on 10.0.0.5\n", "control.json": "{}", ".DS_Store": "",
        "report/strength.json": json.dumps({"state": "done", "results": {}, **MACHINE}),
        "report/benchmark.json": json.dumps({"state": "done", "pairs": {}, **MACHINE}),
        "report/server_job.log": "Started the ladder on 10.0.0.5\n", "report/ladder.log": "Rating ladder\n",
        "checkpoints/gen_0002.pt": "weights", "games/gen_0002.jsonl.gz": "games",
        "games/gen_0003.partial.42.jsonl.gz": "half a generation", "status.json.tmp": "",
    }
    for name, text in files.items():
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text(text)
    return run


def test_the_repository_gets_the_small_results_without_the_machines(tmp_path):
    run = _run(tmp_path)
    written = share_runs.export_results(run, tmp_path / "results" / "demo")
    assert written == ["config.json", "metrics.jsonl", "train.log", "report/strength.json", "report/benchmark.json"]
    strength = json.loads((tmp_path / "results" / "demo" / "report" / "strength.json").read_text())
    assert strength == {"state": "done", "results": {}}


def test_an_archive_holds_the_whole_run_but_what_belongs_to_its_machines(tmp_path):
    run = _run(tmp_path)
    share_runs.pack(run, tmp_path / "dist" / "demo.tar")
    with tarfile.open(tmp_path / "dist" / "demo.tar") as tar:
        names = set(tar.getnames())
        assert names == {f"demo/{n}" for n in ("config.json", "metrics.jsonl", "train.log", "status.json",
                                               "report/strength.json", "report/benchmark.json", "report/ladder.log",
                                               "checkpoints/gen_0002.pt", "games/gen_0002.jsonl.gz")}
        status = json.loads(tar.extractfile("demo/status.json").read())
        assert status == {"state": "finished", "generation": 2}  # a finished run, wherever it is unpacked
        assert tar.extractfile("demo/checkpoints/gen_0002.pt").read() == b"weights"
        tar.extractall(tmp_path / "restored", filter="data")
    assert (tmp_path / "restored" / "demo" / "config.json").exists()
