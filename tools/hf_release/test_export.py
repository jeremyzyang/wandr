#!/usr/bin/env python3
"""End-to-end offline verification for the WANDR Hugging Face export."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import pyarrow.parquet as pq
from datasets import load_dataset

from export import (
    EXPECTED_SCORED_NODES,
    EXPECTED_SMOKE_TASKS,
    EXPECTED_TEST_TASKS,
    EXPECTED_TREE_NODES,
    REPO_ROOT,
    SOURCE_COMMIT,
    SUBMISSION_CONTRACT,
    export,
)
from publish_private import publish_private


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def verify_rows(output: Path) -> None:
    test = pq.read_table(output / "data" / "test-00000-of-00001.parquet").to_pylist()
    smoke = pq.read_table(output / "data" / "smoke-00000-of-00001.parquet").to_pylist()
    assert len(test) == EXPECTED_TEST_TASKS
    assert len(smoke) == EXPECTED_SMOKE_TASKS
    rows = test + smoke
    assert len({row["task_id"] for row in rows}) == len(rows)
    assert [row["task_id"] for row in test] == sorted(row["task_id"] for row in test)
    assert smoke[0]["task_id"] == "smoke"
    assert all(row["source_commit"] == SOURCE_COMMIT for row in rows)
    assert all(json.loads(row["submission_contract_json"]) == SUBMISSION_CONTRACT for row in rows)

    node_count = 0
    scored_node_count = 0
    for row in rows:
        nodes = json.loads(row["task_tree_json"])
        node_count += len(nodes)
        scored_node_count += len(nodes) if row["scored"] else 0
        assert nodes[0]["task_id"] == row["task_id"]
        assert [node["order"] for node in nodes] == list(range(len(nodes)))
        assert row["required_output_files"] == [node["required_output_file"] for node in nodes]
        names = {node["task_id"] for node in nodes}
        for node in nodes:
            assert node["parent_id"] is None or node["parent_id"] in names
            expected_fields = []
            for key in node["key_hierarchy"]:
                if key["name"] == "url":
                    continue
                expected_fields.extend(
                    field for field in key["fields"] if field not in expected_fields
                )
            assert node["item_fields"] == expected_fields
    assert node_count == EXPECTED_TREE_NODES
    assert scored_node_count == EXPECTED_SCORED_NODES

    withheld = {row["task_id"] for row in rows if row["instruction"] is None}
    assert withheld == {
        "forbes_250_claims",
        "forbes_250_cross",
        "forbes_250_errors",
        "hbcu_proxy_directors",
    }
    for row in rows:
        instruction = (
            REPO_ROOT / "datasets" / "wandr" / row["task_id"].replace("_", "-") / "instruction.md"
        ).read_bytes()
        assert row["instruction_sha256"] == hashlib.sha256(instruction).hexdigest()
        if row["instruction"] is not None:
            assert row["instruction"].encode() == instruction
    affected = {row["task_id"] for row in rows if not row["self_contained"]}
    assert affected == {
        "forbes_250_claims",
        "forbes_250_cross",
        "forbes_250_errors",
        "hbcu_proxy_directors",
        "mozambique_districts",
        "portugal_municipalities",
    }

    answer_schema = SUBMISSION_CONTRACT["properties"]["answer"]
    assert answer_schema == {"type": "object", "additionalProperties": True}


def verify_assets(output: Path) -> None:
    secret_patterns = (
        re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        re.compile(rb"\bhf_[A-Za-z0-9]{30,}\b"),
        re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
        re.compile(rb"\bAKIA[A-Z0-9]{16}\b"),
    )

    def assert_no_secrets(value: bytes, label: object) -> None:
        assert not any(pattern.search(value) for pattern in secret_patterns), label

    excluded = _jsonl(output / "auxiliary" / "excluded-artifacts.jsonl")
    excluded_paths = {record["path"] for record in excluded}
    actual_paths = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in REPO_ROOT.glob("reference/wandr_tasks/*/artifacts/**/*")
        if path.is_file()
    }
    assert excluded_paths == actual_paths
    assert all(record["included"] is False for record in excluded)

    package_files = [path for path in output.rglob("*") if path.is_file()]
    for record in excluded:
        artifact = REPO_ROOT / record["path"]
        assert record["sha256"] == _sha256(artifact)
        assert record["size"] == artifact.stat().st_size
        artifact_bytes = artifact.read_bytes()
        assert all(artifact_bytes not in path.read_bytes() for path in package_files)

    archive = output / "auxiliary" / "wandr-task-sources.tar.gz"
    with tarfile.open(archive, "r:gz") as bundle:
        names = bundle.getnames()
        assert names == sorted(names)
        for member in bundle.getmembers():
            pure = PurePosixPath(member.name)
            assert member.isfile()
            assert not pure.is_absolute()
            assert ".." not in pure.parts
            assert "artifacts" not in pure.parts
            assert member.mtime == 0
            extracted = bundle.extractfile(member)
            assert extracted is not None
            assert_no_secrets(extracted.read(), member.name)

    source_manifest = _jsonl(output / "auxiliary" / "source-files.jsonl")
    assert all(
        _sha256(REPO_ROOT / record["path"]) == record["sha256"] for record in source_manifest
    )
    evaluator = _jsonl(output / "evaluator" / "index.jsonl")
    assert len(evaluator) == EXPECTED_TREE_NODES
    source_paths = {record["path"] for record in source_manifest}
    assert all(set(record["spec_paths"]).issubset(source_paths) for record in evaluator)

    for path in package_files:
        assert_no_secrets(path.read_bytes(), path)


def verify_card_and_stock_load(output: Path) -> None:
    card = (output / "README.md").read_text(encoding="utf-8")
    for required in (
        "path: data/test-00000-of-00001.parquet",
        "path: data/smoke-00000-of-00001.parquet",
        "https://arxiv.org/abs/2608.14747",
        "official article",
        SOURCE_COMMIT,
        "free-form `answer` object",
        "reference-free specifications",
    ):
        assert required in card

    loaded_test = load_dataset(str(output), split="test")
    loaded_smoke = load_dataset(str(output), split="smoke")
    assert len(loaded_test) == EXPECTED_TEST_TASKS
    assert len(loaded_smoke) == EXPECTED_SMOKE_TASKS
    streamed_test = load_dataset(str(output), split="test", streaming=True)
    streamed_smoke = load_dataset(str(output), split="smoke", streaming=True)
    assert sum(1 for _ in streamed_test) == EXPECTED_TEST_TASKS
    assert sum(1 for _ in streamed_smoke) == EXPECTED_SMOKE_TASKS


class FakeHubApi:
    def __init__(self, private_states: list[bool]) -> None:
        self.private_states = iter(private_states)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def create_repo(self, **kwargs: object) -> None:
        self.calls.append(("create_repo", kwargs))

    def repo_info(self, **kwargs: object) -> object:
        self.calls.append(("repo_info", kwargs))
        return SimpleNamespace(private=next(self.private_states))

    def upload_folder(self, **kwargs: object) -> None:
        self.calls.append(("upload_folder", kwargs))


def verify_private_publish_guard(output: Path) -> None:
    api = FakeHubApi([True, True])
    publish_private(api, output, "perplexity-ai/wandr")
    assert api.calls[0][0] == "create_repo"
    assert api.calls[0][1]["private"] is True
    assert [name for name, _ in api.calls] == [
        "create_repo",
        "repo_info",
        "upload_folder",
        "repo_info",
    ]
    public_api = FakeHubApi([False])
    try:
        publish_private(public_api, output, "perplexity-ai/wandr")
    except RuntimeError:
        pass
    else:
        raise AssertionError("publish helper accepted a public repository")
    assert "upload_folder" not in [name for name, _ in public_api.calls]


def verify_determinism(output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="wandr-hf-repeat-") as directory:
        repeated = Path(directory) / "release"
        export(repeated)
        assert _file_hashes(output) == _file_hashes(repeated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    export(output)
    verify_rows(output)
    verify_assets(output)
    verify_card_and_stock_load(output)
    verify_private_publish_guard(output)
    verify_determinism(output)
    shutil.rmtree(output / "cache", ignore_errors=True)
    print(f"HF export verification OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
