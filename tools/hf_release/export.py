#!/usr/bin/env python3
"""Export the pinned WANDR task corpus as a deterministic HF dataset repo."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_SRC = REPO_ROOT / "adapters" / "wandr" / "src"
if str(ADAPTER_SRC) not in sys.path:
    sys.path.insert(0, str(ADAPTER_SRC))

from wandr.adapter import (  # noqa: E402
    _clear_task_source_modules,
    _render_instruction,
    _slug,
)
from wandr.adapter import REFERENCE_TASKS_DIR  # noqa: E402
from src.config import flatten_tasks, load_task_config  # noqa: E402


SOURCE_COMMIT = "ccb0baeb96f1c77a48e47f92122c57479ee99700"
SOURCE_REPO = "https://github.com/perplexityai/wandr"
HF_REPO_ID = "perplexity-ai/wandr"
EXPORT_VERSION = "1"
PAPER_TITLE = "WANDR: A Benchmark for Wide and Deep Research"
PAPER_AUTHORS = (
    "Vitaliy Polshkov",
    "Marcin Pitera",
    "Jeremy Yang",
    "Kirill Priemko",
    "Maksim Gaiduk",
    "Aleksandr Nikolenko",
    "Denis Bykov",
    "Clare Southern",
    "Denis Yarats",
    "Jerry Ma",
)
ARXIV_ID = "2608.14747"
PAPER_DOI = f"10.48550/arXiv.{ARXIV_ID}"
EXPECTED_TEST_TASKS = 500
EXPECTED_SMOKE_TASKS = 1
EXPECTED_TREE_NODES = 609
EXPECTED_SCORED_NODES = 608
EXCLUDED_ARTIFACT_GLOB = "reference/wandr_tasks/**/artifacts/**"
INPUT_PATHS = (
    "reference/wandr_tasks",
    "datasets/wandr",
    "adapters/wandr/src/wandr/origin/instruction_macro.md.jinja",
    "adapters/wandr/src/wandr/origin/wandr_core/src/config.py",
    "adapters/wandr/src/wandr/origin/wandr_core/src/markup.py",
    "adapters/wandr/src/wandr/origin/wandr_core/src/submissions.py",
)

SUBMISSION_CONTRACT = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["item", "url", "excerpts", "answer"],
    "properties": {
        "item": {"type": "object", "additionalProperties": {"type": "string"}},
        "url": {"type": "string", "format": "uri", "pattern": "^https?://"},
        "excerpts": {"type": "array", "items": {"type": "string"}},
        "answer": {"type": "object", "additionalProperties": True},
    },
    "additionalProperties": True,
}

ROW_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("source_commit", pa.string(), nullable=False),
        pa.field("source_url", pa.string(), nullable=False),
        pa.field("instruction", pa.string()),
        pa.field("instruction_sha256", pa.string(), nullable=False),
        pa.field("instruction_status", pa.string(), nullable=False),
        pa.field("instruction_source_url", pa.string(), nullable=False),
        pa.field("required_output_files", pa.list_(pa.string()), nullable=False),
        pa.field("task_tree_json", pa.string(), nullable=False),
        pa.field("metadata_json", pa.string(), nullable=False),
        pa.field("submission_contract_json", pa.string(), nullable=False),
        pa.field("evaluator_index_path", pa.string(), nullable=False),
        pa.field("source_bundle_path", pa.string(), nullable=False),
        pa.field("external_dependencies_json", pa.string(), nullable=False),
        pa.field("self_contained", pa.bool_(), nullable=False),
        pa.field("scored", pa.bool_(), nullable=False),
    ]
)

CITATION_BIB = f"""@misc{{polshkov2026wandr,
  title={{{PAPER_TITLE}}},
  author={{Polshkov, Vitaliy and Pitera, Marcin and Yang, Jeremy and Priemko, Kirill and Gaiduk, Maksim and Nikolenko, Aleksandr and Bykov, Denis and Southern, Clare and Yarats, Denis and Ma, Jerry}},
  year={{2026}},
  eprint={{{ARXIV_ID}}},
  archivePrefix={{arXiv}},
  primaryClass={{cs.LG}},
  doi={{{PAPER_DOI}}},
  url={{https://arxiv.org/abs/{ARXIV_ID}}},
}}
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _source_url(path: str) -> str:
    return f"{SOURCE_REPO}/blob/{SOURCE_COMMIT}/{path}"


def _source_tree_url(path: str) -> str:
    return f"{SOURCE_REPO}/tree/{SOURCE_COMMIT}/{path}"


def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def verify_pinned_inputs() -> None:
    _run_git("cat-file", "-e", f"{SOURCE_COMMIT}^{{commit}}")
    result = subprocess.run(
        ["git", "diff", "--quiet", SOURCE_COMMIT, "--", *INPUT_PATHS],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("release inputs differ from pinned source commit " + SOURCE_COMMIT)
    for path in INPUT_PATHS:
        if not (REPO_ROOT / path).exists():
            raise FileNotFoundError(f"missing pinned release input: {path}")


def _root_task_dirs() -> list[Path]:
    return sorted(
        path
        for path in REFERENCE_TASKS_DIR.iterdir()
        if path.is_dir() and (path / "config.py").is_file()
    )


def _artifact_paths() -> list[Path]:
    return sorted(path for path in REFERENCE_TASKS_DIR.glob("*/artifacts/**/*") if path.is_file())


def _artifact_markers(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8", errors="strict")
    markers = {line.strip() for line in text.splitlines() if len(line.strip()) >= 48}
    if path.suffix in {".json", ".jsonl"}:
        for line in text.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                markers.update(
                    item for item in value.values() if isinstance(item, str) and len(item) >= 24
                )
    return markers


def _instruction_artifact_dependencies(
    instruction: str, root_name: str, artifacts: list[Path]
) -> list[str]:
    dependencies = []
    for path in artifacts:
        if path.relative_to(REFERENCE_TASKS_DIR).parts[0] != root_name:
            continue
        artifact_text = path.read_text(encoding="utf-8", errors="strict")
        exact_match = bool(artifact_text) and artifact_text in instruction
        marker_matches = sum(marker in instruction for marker in _artifact_markers(path))
        if exact_match or marker_matches >= 3:
            dependencies.append(path.relative_to(REPO_ROOT).as_posix())
    return dependencies


def _included_source_paths() -> list[Path]:
    paths = []
    for path in sorted(REFERENCE_TASKS_DIR.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(REFERENCE_TASKS_DIR)
        if "artifacts" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        paths.append(path)
    return paths


def _source_role(path: Path) -> str:
    name = path.name
    if name == "task_template.md.jinja":
        return "solver_template"
    if name == "config.py":
        return "task_and_evaluator_config"
    if "prompts" in path.parts or "schemas" in path.parts:
        return "evaluator_spec"
    return "task_source_support"


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = "".join(_json(record) + "\n" for record in records)
    path.write_text(value, encoding="utf-8", newline="\n")


def _write_source_bundle(output: Path, source_paths: list[Path]) -> str:
    relative_output = "auxiliary/wandr-task-sources.tar.gz"
    archive_path = output / relative_output
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in [REPO_ROOT / "LICENSE", REPO_ROOT / "NOTICE", *source_paths]:
            relative = path.relative_to(REPO_ROOT).as_posix()
            info = tarfile.TarInfo(relative)
            data = path.read_bytes()
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    with archive_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(buffer.getvalue())
    return relative_output


def _task_metadata(task_slug: str) -> tuple[dict[str, Any], list[str]]:
    task_toml = REPO_ROOT / "datasets" / "wandr" / task_slug / "task.toml"
    data = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    return metadata, list(metadata["required_file_paths"])


def _node_source_dir(root_name: str, node_name: str) -> Path:
    suffix = node_name.removeprefix(root_name).lstrip(".")
    return (
        REFERENCE_TASKS_DIR / root_name / Path(*suffix.split("."))
        if suffix
        else REFERENCE_TASKS_DIR / root_name
    )


def _task_tree(config: Any, *, withhold_text: bool) -> list[dict[str, Any]]:
    nodes = []
    for order, node in enumerate(flatten_tasks(config)):
        parent_id = node.name.rpartition(".")[0] or None
        rendered_task = node.task
        nodes.append(
            {
                "task_id": node.name,
                "parent_id": parent_id,
                "order": order,
                "task_text": None if withhold_text else rendered_task,
                "task_text_sha256": _sha256_bytes(rendered_task.encode()),
                "task_text_status": "excluded_embedded_asset" if withhold_text else "included",
                "item_fields": node.item_fields,
                "key_hierarchy": [
                    {
                        "name": key.name,
                        "fields": list(key.key_fields),
                        "required": key.required,
                    }
                    for key in node.key_hierarchy
                ],
                "required_output_file": f"results_{node.name}.jsonl",
                "task_fingerprint": node.fingerprint,
                "eval_fingerprint": node.eval.fingerprint,
            }
        )
    return nodes


def _evaluator_records(config: Any, source_paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    root_name = config.name
    root_config = f"reference/wandr_tasks/{root_name}/config.py"
    for node in flatten_tasks(config):
        node_dir = _node_source_dir(root_name, node.name)
        paths = [
            path.relative_to(REPO_ROOT).as_posix()
            for path in source_paths
            if path == REPO_ROOT / root_config
            or (
                path.is_relative_to(node_dir)
                and path.name != "task_template.md.jinja"
                and _source_role(path) in {"evaluator_spec", "task_source_support"}
            )
        ]
        records.append(
            {
                "task_id": node.name,
                "root_task_id": root_name,
                "eval_fingerprint": node.eval.fingerprint,
                "spec_paths": sorted(set(paths)),
                "runtime": {
                    "included": False,
                    "reason": "Evaluator runtime remains authoritative on GitHub/Harbor.",
                    "source_url": _source_url(
                        "adapters/wandr/src/wandr/origin/wandr_core/src/pipeline.py"
                    ),
                },
            }
        )
    return records


def _artifact_manifest(artifacts: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "root_task_id": path.relative_to(REFERENCE_TASKS_DIR).parts[0],
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
            "included": False,
            "classification": "excluded_third_party_evidence_or_derived_asset",
            "source_url": _source_url(path.relative_to(REPO_ROOT).as_posix()),
        }
        for path in artifacts
    ]


def _row(
    config: Any,
    instruction: str,
    metadata: dict[str, Any],
    required_outputs: list[str],
    artifact_records: list[dict[str, Any]],
    embedded_paths: list[str],
    source_bundle_path: str,
) -> dict[str, Any]:
    task_id = config.name
    task_slug = _slug(task_id)
    instruction_path = f"datasets/wandr/{task_slug}/instruction.md"
    dependencies = [record for record in artifact_records if record["root_task_id"] == task_id]
    withheld = bool(embedded_paths)
    return {
        "task_id": task_id,
        "source_commit": SOURCE_COMMIT,
        "source_url": _source_tree_url(f"reference/wandr_tasks/{task_id}"),
        "instruction": None if withheld else instruction,
        "instruction_sha256": _sha256_bytes(instruction.encode()),
        "instruction_status": "excluded_embedded_asset" if withheld else "included",
        "instruction_source_url": _source_url(instruction_path),
        "required_output_files": required_outputs,
        "task_tree_json": _json(_task_tree(config, withhold_text=withheld)),
        "metadata_json": _json(metadata),
        "submission_contract_json": _json(SUBMISSION_CONTRACT),
        "evaluator_index_path": "evaluator/index.jsonl",
        "source_bundle_path": source_bundle_path,
        "external_dependencies_json": _json(dependencies),
        "self_contained": not dependencies,
        "scored": task_id != "smoke",
    }


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=ROW_SCHEMA)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
        version="2.6",
    )


def _dataset_card() -> str:
    authors = ", ".join(PAPER_AUTHORS)
    return f'''---
pretty_name: WANDR
license: other
license_name: Apache-2.0 for Perplexity-owned material; see NOTICE
license_link: LICENSE
language:
- en
task_categories:
- question-answering
- information-retrieval
tags:
- research-agents
- benchmark
- web-research
- arxiv:{ARXIV_ID}
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-00000-of-00001.parquet
  - split: smoke
    path: data/smoke-00000-of-00001.parquet
---

# WANDR

WANDR (Wide ANd Deep Research) is a benchmark of 500 realistic, structured,
high-volume web research tasks. This dataset is a task-and-verification corpus,
not a question/answer collection: it contains no solver outputs or reference
answer sets. WANDR evaluation refetches cited pages and judges submitted records
against task-specific, reference-free specifications.

This private-staging export is pinned exactly to the public GitHub source
snapshot [`{SOURCE_COMMIT}`]({SOURCE_REPO}/tree/{SOURCE_COMMIT}). GitHub and
Harbor remain authoritative for executable evaluation. The snapshot was audited
for this release, but it has not been established as the exact historical
snapshot used for every result in the paper.

See the [paper](https://arxiv.org/abs/2608.14747), the
[official article](https://www.perplexity.ai/hub/blog/wandr-benchmark-evaluating-research-agents-that-must-search-wide-and-deep),
and the [evaluation repository]({SOURCE_REPO}/tree/{SOURCE_COMMIT}).

## Paper and citation

**{PAPER_TITLE}** (2026)

Authors, in publication order: {authors}.

[arXiv:{ARXIV_ID}](https://arxiv.org/abs/{ARXIV_ID}) ·
[DOI:{PAPER_DOI}](https://doi.org/{PAPER_DOI})

Download [`CITATION.bib`](CITATION.bib), or copy the [BibTeX citation](#citation)
below.

## Splits

- `test`: 500 scored root tasks, in canonical order.
- `smoke`: one unscored framework task. It is not a validation split and is not
  part of benchmark scoring.

The 501 roots expand to 609 ordered task-tree nodes, 608 of which are scored.
Eighty-five scored roots have subtasks. Rows preserve ordered hierarchies,
required counts, item fields, output filenames, source metadata, and exact
instruction/task-text hashes.

## Loading

While this dataset repository is private, loading or streaming requires a
Hugging Face account with access. Authenticate first with `hf auth login`, then
allow `datasets` to use that saved token. No paid inference or retrieval API
keys are needed, and the dataset uses no custom remote code:

```python
from datasets import load_dataset

tasks = load_dataset(
    "{HF_REPO_ID}",
    split="test",
    revision="<private-release-tag-or-commit>",
    token=True,
)

# Pass only solver-facing fields to an agent. Evaluator assets are not inputs.
task = tasks[0]
if task["instruction"] is None:
    raise ValueError(
        "Instruction text is withheld; review the pinned source under its applicable terms: "
        + task["instruction_source_url"]
    )

solver_input = {{
    "instruction": task["instruction"],
    "required_output_files": task["required_output_files"],
}}
```

Heterogeneous task structures and metadata are losslessly represented as JSON
strings (`task_tree_json`, `metadata_json`, `submission_contract_json`, and
`external_dependencies_json`) so stock `datasets` can load one stable Parquet
schema. Parse them with `json.loads`.

Each submission JSONL row has an `item` object, HTTP(S) `url`, `excerpts` as a
list of strings, and a free-form `answer` object. The `answer` object
intentionally has no fixed field vocabulary; each task defines the answer it
asks for.

## Evaluation specifications and sources

`evaluator/index.jsonl` indexes task-specific judgment schemas, rubrics,
canonicalization, and deduplication sources by node ID. The lossless,
deterministic source bundle is `auxiliary/wandr-task-sources.tar.gz`.
`auxiliary/source-files.jsonl` records every included source hash. Evaluator
assets are deliberately separate from the loadable Parquet files and must not
be passed to solvers. Relay, Docker, provider integrations, generated Harbor
runtimes, and common evaluator runtime code are not duplicated here.

Run scoring through the pinned GitHub/Harbor workflow. It can incur paid model
and retrieval calls; loading this dataset never does.

## Redistribution boundary and non-self-contained tasks

The upstream Apache-2.0 license and NOTICE apply only to material for which
Perplexity holds the necessary rights. Third-party material retains its own
terms. This release does not claim that all linked or task-required material is
Apache-2.0 licensed.

Every file matching `{EXCLUDED_ARTIFACT_GLOB}` is excluded. The complete path,
size, SHA-256, provenance link, and inclusion status are in
`auxiliary/excluded-artifacts.jsonl`. A link is provenance, not legal clearance.

Four instructions embed excluded evidence and therefore have a null
`instruction`: `forbes_250_claims`, `forbes_250_cross`, `forbes_250_errors`, and
`hbcu_proxy_directors`. Their exact hashes and pinned instruction links are
retained; no edited substitute is supplied. `mozambique_districts` and
`portugal_municipalities` have included solver instructions but depend on
excluded canonical evaluator assets. All six affected roots are explicitly
marked `self_contained=false`. Use the authoritative pinned repository only
under the applicable source terms.

## Limitations

WANDR depends on live web pages. Availability, content, and facts change over
time, and results can vary with retrieval date. Required volume and hierarchy
depth make tasks expensive. The dataset does not include answers, guarantee
that external dependencies remain reachable, or make the evaluator runnable
offline.

## Citation

```bibtex
{CITATION_BIB.rstrip()}
```
'''


def _safe_archive_members(path: Path) -> list[str]:
    with tarfile.open(path, "r:gz") as archive:
        names = [member.name for member in archive.getmembers()]
        for member in archive.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or not member.isfile():
                raise RuntimeError(f"unsafe source archive member: {member.name}")
        return names


def _manifest(output: Path, counts: dict[str, int]) -> dict[str, Any]:
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "release-manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "format_version": EXPORT_VERSION,
        "dataset_repo_id": HF_REPO_ID,
        "intended_visibility": "private",
        "source_commit": SOURCE_COMMIT,
        "source_repository": SOURCE_REPO,
        "counts": counts,
        "excluded_artifact_glob": EXCLUDED_ARTIFACT_GLOB,
        "files": files,
    }


def export(output: Path) -> dict[str, Any]:
    verify_pinned_inputs()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    source_paths = _included_source_paths()
    artifacts = _artifact_paths()
    artifact_records = _artifact_manifest(artifacts)
    source_bundle_path = _write_source_bundle(output, source_paths)
    _safe_archive_members(output / source_bundle_path)

    source_records = [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "role": _source_role(path),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
            "source_url": _source_url(path.relative_to(REPO_ROOT).as_posix()),
        }
        for path in source_paths
    ]
    _write_jsonl(output / "auxiliary" / "source-files.jsonl", source_records)
    _write_jsonl(output / "auxiliary" / "excluded-artifacts.jsonl", artifact_records)

    test_rows = []
    smoke_rows = []
    evaluator_records = []
    instruction_blockers = []
    roots_with_subtasks = 0
    tree_nodes = 0
    scored_nodes = 0
    for task_dir in _root_task_dirs():
        root_name = task_dir.name
        _clear_task_source_modules()
        config = load_task_config(root_name, task_dir)
        slug = _slug(root_name)
        instruction_path = REPO_ROOT / "datasets" / "wandr" / slug / "instruction.md"
        instruction = instruction_path.read_text(encoding="utf-8")
        rendered = _render_instruction(config)
        if rendered != instruction:
            raise RuntimeError(f"generated instruction drift for {root_name}")
        embedded_paths = _instruction_artifact_dependencies(instruction, root_name, artifacts)
        if embedded_paths:
            instruction_blockers.append(root_name)
        metadata, required_outputs = _task_metadata(slug)
        nodes = flatten_tasks(config)
        tree_nodes += len(nodes)
        if root_name != "smoke":
            scored_nodes += len(nodes)
            roots_with_subtasks += int(len(nodes) > 1)
        row = _row(
            config,
            instruction,
            metadata,
            required_outputs,
            artifact_records,
            embedded_paths,
            source_bundle_path,
        )
        (smoke_rows if root_name == "smoke" else test_rows).append(row)
        evaluator_records.extend(_evaluator_records(config, source_paths))

    expected_blockers = {
        "forbes_250_claims",
        "forbes_250_cross",
        "forbes_250_errors",
        "hbcu_proxy_directors",
    }
    if set(instruction_blockers) != expected_blockers:
        raise RuntimeError(
            f"instruction artifact blocker set changed: {sorted(instruction_blockers)}"
        )
    counts = {
        "test_tasks": len(test_rows),
        "smoke_tasks": len(smoke_rows),
        "tree_nodes": tree_nodes,
        "scored_nodes": scored_nodes,
        "scored_roots_with_subtasks": roots_with_subtasks,
        "source_files": len(source_records),
        "excluded_artifacts": len(artifact_records),
        "withheld_instructions": len(instruction_blockers),
    }
    expected = {
        "test_tasks": EXPECTED_TEST_TASKS,
        "smoke_tasks": EXPECTED_SMOKE_TASKS,
        "tree_nodes": EXPECTED_TREE_NODES,
        "scored_nodes": EXPECTED_SCORED_NODES,
        "scored_roots_with_subtasks": 85,
        "excluded_artifacts": 13,
        "withheld_instructions": 4,
    }
    for name, value in expected.items():
        if counts[name] != value:
            raise RuntimeError(f"{name}: expected {value}, got {counts[name]}")

    _write_parquet(output / "data" / "test-00000-of-00001.parquet", test_rows)
    _write_parquet(output / "data" / "smoke-00000-of-00001.parquet", smoke_rows)
    _write_jsonl(output / "evaluator" / "index.jsonl", evaluator_records)
    (output / "README.md").write_text(_dataset_card(), encoding="utf-8", newline="\n")
    (output / "CITATION.bib").write_text(CITATION_BIB, encoding="utf-8", newline="\n")
    shutil.copyfile(REPO_ROOT / "LICENSE", output / "LICENSE")
    shutil.copyfile(REPO_ROOT / "NOTICE", output / "NOTICE")
    third_party = (
        "# Redistribution boundary\n\n"
        "The Apache-2.0 LICENSE and upstream NOTICE cover only material for which "
        "Perplexity holds the necessary rights. Third-party material retains its "
        "own terms. See `auxiliary/excluded-artifacts.jsonl`; none of those bytes "
        "are included in this release. Pinned links are provenance, not clearance.\n"
    )
    (output / "THIRD_PARTY.md").write_text(third_party, encoding="utf-8", newline="\n")

    manifest = _manifest(output, counts)
    (output / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = export(args.output.resolve())
    print(_json({"output": str(args.output.resolve()), "counts": manifest["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
