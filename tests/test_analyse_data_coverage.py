"""Tests for analyse_data_coverage.py using path lists and weak labels."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _create_dummy_npz(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        embeddings=np.zeros((1, 1), dtype=np.float32),
        start_sec=np.array([0.0], dtype=np.float32),
        end_sec=np.array([1.0], dtype=np.float32),
    )
    return path


def test_analyse_data_coverage_with_path_lists(tmp_path):
    weak_labels_csv = tmp_path / "weak_labels.csv"
    pd.DataFrame(
        [
            {
                "MONITORING_SITE": "SITE_A",
                "AUDIO_FILE_ID": "REC1",
                "SPECIES_Species_A": 1,
                "SPECIES_Species_B": 0,
            },
            {
                "MONITORING_SITE": "SITE_A",
                "AUDIO_FILE_ID": "REC2",
                "SPECIES_Species_A": 0,
                "SPECIES_Species_B": 1,
            },
            {
                "MONITORING_SITE": "SITE_B",
                "AUDIO_FILE_ID": "REC3",
                "SPECIES_Species_A": 1,
                "SPECIES_Species_B": 1,
            },
        ]
    ).to_csv(weak_labels_csv, index=False)

    emb_dir = tmp_path / "embeddings"
    train_paths = [
        _create_dummy_npz(emb_dir / "REC1.embeddings.npz"),
        _create_dummy_npz(emb_dir / "REC3.embeddings.npz"),
    ]
    test_paths = [_create_dummy_npz(emb_dir / "REC2.embeddings.npz")]

    train_txt = tmp_path / "train.txt"
    train_txt.write_text("\n".join(str(p) for p in train_paths))

    test_txt = tmp_path / "test.txt"
    test_txt.write_text("\n".join(str(p) for p in test_paths))

    output_json = tmp_path / "coverage.json"
    script_path = Path(__file__).parent.parent / "scripts" / "analyse_data_coverage.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--train-labels",
            str(train_txt),
            "--test-labels",
            str(test_txt),
            "--weak-labels-csv",
            str(weak_labels_csv),
            "--output",
            str(output_json),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert output_json.exists()

    with open(output_json, "r") as f:
        report = json.load(f)

    train_counts = {c["class"]: c["count"] for c in report["train"]["y_true"]["per_class"]}
    test_counts = {c["class"]: c["count"] for c in report["test"]["y_true"]["per_class"]}

    assert report["train"]["y_true"]["total_samples"] == 2
    assert report["test"]["y_true"]["total_samples"] == 1
    assert train_counts["Species_A"] == 2
    assert train_counts["Species_B"] == 1
    assert test_counts["Species_B"] == 1
