#!/usr/bin/env python
"""Index Codex conversations related to this workspace.

The index is intentionally project-scoped. It includes conversations whose
recorded cwd is the project root or a child path, plus rollout summaries that
explicitly mention the project path.
"""

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


INDEX_JSON = "type10_7_conversations.json"
INDEX_MD = "type10_7_conversations.md"
DEFAULT_MAX_SESSION_LINES = 400
DEFAULT_MAX_SESSION_MESSAGES = 8


@dataclass
class ConversationEntry:
    thread_id: str
    source_kind: str
    title: str
    summary: str
    cwd: str
    updated_at: str
    rollout_path: str
    source_path: str
    search_text: str


@dataclass
class SearchResult:
    entry: ConversationEntry
    score: int


def script_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME")
    if configured:
        return Path(configured)
    return Path.home() / ".codex"


def canonical_windows_path(value: object) -> str:
    text = str(value or "").strip().strip('"').strip("'")
    text = text.replace("/", "\\")
    if text.startswith("\\\\?\\"):
        text = text[4:]
    while "\\\\" in text:
        text = text.replace("\\\\", "\\")
    text = text.rstrip("\\")
    return text.casefold()


def is_project_path(candidate: object, project_root: Path) -> bool:
    candidate_text = canonical_windows_path(candidate)
    root_text = canonical_windows_path(project_root)
    return bool(candidate_text) and (
        candidate_text == root_text or candidate_text.startswith(root_text + "\\")
    )


def mentions_project(text: str, project_root: Path) -> bool:
    canonical_text = canonical_windows_path(text)
    return canonical_windows_path(project_root) in canonical_text


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def first_heading(lines: Sequence[str], fallback: str) -> str:
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def compact_summary(lines: Sequence[str], max_chars: int = 900) -> str:
    useful: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(thread_id|updated_at|rollout_path|cwd):", stripped, re.I):
            continue
        if stripped.startswith("# "):
            continue
        useful.append(stripped)
        if sum(len(item) for item in useful) >= max_chars:
            break
    text = " ".join(useful)
    return text[:max_chars].strip()


def parse_summary_metadata(lines: Sequence[str]) -> dict:
    metadata = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            break
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        metadata[key.strip().casefold()] = value.strip()
    return metadata


def parse_rollout_summary(path: Path, project_root: Path) -> Optional[ConversationEntry]:
    text = read_text(path)
    lines = text.splitlines()
    metadata = parse_summary_metadata(lines)
    cwd = metadata.get("cwd", "")
    if not (is_project_path(cwd, project_root) or mentions_project(text, project_root)):
        return None

    title = first_heading(lines, path.stem)
    summary = compact_summary(lines)
    thread_id = metadata.get("thread_id") or path.stem
    rollout_path = metadata.get("rollout_path", "")
    updated_at = metadata.get("updated_at", "")
    search_text = "\n".join([title, summary, text[:5000], cwd, rollout_path])
    return ConversationEntry(
        thread_id=thread_id,
        source_kind="summary",
        title=title,
        summary=summary,
        cwd=cwd,
        updated_at=updated_at,
        rollout_path=rollout_path,
        source_path=str(path),
        search_text=search_text,
    )


def message_text_from_payload(payload: dict) -> str:
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    chunks = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("input_text") or item.get("output_text")
        if isinstance(text, str):
            chunks.append(text)
    return " ".join(chunks)


def strip_codex_context_injection(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("# AGENTS.md instructions for "):
        return stripped

    marker = "</environment_context>"
    if marker in stripped:
        return stripped.split(marker, 1)[1].strip()
    return ""


def parse_session_file(
    path: Path,
    project_root: Path,
    max_lines: int = DEFAULT_MAX_SESSION_LINES,
    max_messages: int = DEFAULT_MAX_SESSION_MESSAGES,
) -> Optional[ConversationEntry]:
    thread_id = ""
    cwd = ""
    updated_at = ""
    snippets: List[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle):
            if line_no >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            record_type = record.get("type")
            if record_type == "session_meta":
                thread_id = str(payload.get("id") or thread_id)
                cwd = str(payload.get("cwd") or cwd)
                updated_at = str(payload.get("timestamp") or updated_at)
                continue

            if len(snippets) >= max_messages:
                continue
            if record_type != "response_item":
                continue
            if payload.get("type") != "message":
                continue
            role = str(payload.get("role") or "")
            if role not in {"user", "assistant"}:
                continue
            text = message_text_from_payload(payload)
            text = strip_codex_context_injection(text)
            if text:
                snippets.append(f"{role}: {text[:400]}")

    if not is_project_path(cwd, project_root):
        return None

    title = snippets[0].replace("user: ", "", 1)[:120] if snippets else path.stem
    summary = " ".join(snippets)[:900]
    search_text = "\n".join([title, summary, cwd, str(path)])
    return ConversationEntry(
        thread_id=thread_id or path.stem,
        source_kind="session",
        title=title,
        summary=summary,
        cwd=cwd,
        updated_at=updated_at,
        rollout_path=str(path),
        source_path=str(path),
        search_text=search_text,
    )


def iter_rollout_summary_paths(codex_home: Path) -> Iterable[Path]:
    root = codex_home / "memories" / "rollout_summaries"
    if not root.exists():
        return []
    return sorted(root.glob("*.md"))


def iter_session_paths(codex_home: Path) -> Iterable[Path]:
    root = codex_home / "sessions"
    if not root.exists():
        return []
    return sorted(root.rglob("*.jsonl"))


def entry_sort_key(entry: ConversationEntry) -> tuple:
    return (entry.updated_at or "", entry.thread_id)


def write_json(entries: Sequence[ConversationEntry], path: Path) -> None:
    path.write_text(
        json.dumps([asdict(entry) for entry in entries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_markdown(entries: Sequence[ConversationEntry], path: Path, project_root: Path) -> None:
    lines = [
        "# E:\\type10-7 Conversation Index",
        "",
        f"- Project root: `{project_root}`",
        f"- Entries: {len(entries)}",
        "- Refresh: `conda activate ssr-gpu; python tools/conversation_index.py build`",
        "- Search: `conda activate ssr-gpu; python tools/conversation_index.py search \"keyword\"`",
        "",
        "| Updated | Thread | Source | Title | Path |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        path_value = entry.source_path or entry.rollout_path
        lines.append(
            "| {updated} | {thread} | {source} | {title} | `{path}` |".format(
                updated=escape_md(entry.updated_at),
                thread=escape_md(entry.thread_id),
                source=escape_md(entry.source_kind),
                title=escape_md(entry.title),
                path=escape_md(path_value),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_index(project_root: Path, codex_home: Path, output_dir: Path) -> List[ConversationEntry]:
    entries_by_key = {}

    for path in iter_rollout_summary_paths(codex_home):
        entry = parse_rollout_summary(path, project_root)
        if entry is None:
            continue
        key = entry.thread_id or entry.rollout_path or entry.source_path
        entries_by_key[key] = entry

    for path in iter_session_paths(codex_home):
        entry = parse_session_file(path, project_root)
        if entry is None:
            continue
        key = entry.thread_id or entry.rollout_path or entry.source_path
        if key in entries_by_key:
            continue
        entries_by_key[key] = entry

    entries = sorted(entries_by_key.values(), key=entry_sort_key, reverse=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(entries, output_dir / INDEX_JSON)
    write_markdown(entries, output_dir / INDEX_MD, project_root)
    return entries


def load_index(path: Path) -> List[ConversationEntry]:
    data = json.loads(read_text(path))
    return [ConversationEntry(**item) for item in data]


def tokenize(query: str) -> List[str]:
    return [token.casefold() for token in re.findall(r"[\w:\\.-]+", query) if token.strip()]


def search_entries(entries: Sequence[ConversationEntry], query: str, limit: int = 10) -> List[SearchResult]:
    terms = tokenize(query)
    results: List[SearchResult] = []
    if not terms:
        return results

    for entry in entries:
        title = entry.title.casefold()
        summary = entry.summary.casefold()
        body = entry.search_text.casefold()
        score = 0
        for term in terms:
            score += 6 * title.count(term)
            score += 3 * summary.count(term)
            score += body.count(term)
        if score > 0:
            results.append(SearchResult(entry=entry, score=score))

    results.sort(key=lambda item: (item.score, item.entry.updated_at), reverse=True)
    return results[:limit]


def render_search_results(results: Sequence[SearchResult]) -> str:
    if not results:
        return "No matching E:\\type10-7 conversation entries found."

    blocks = []
    for item in results:
        entry = item.entry
        blocks.append(
            "\n".join(
                [
                    f"[score {item.score}] {entry.title}",
                    f"  thread_id: {entry.thread_id}",
                    f"  updated_at: {entry.updated_at}",
                    f"  source: {entry.source_kind}",
                    f"  path: {entry.source_path or entry.rollout_path}",
                    f"  summary: {entry.summary[:500]}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=script_project_root())
    parser.add_argument("--codex-home", type=Path, default=default_codex_home())
    parser.add_argument("--output-dir", type=Path, default=script_project_root() / "conversation_index")
    subparsers = parser.add_subparsers(dest="command")

    build = subparsers.add_parser("build", help="Build the project-scoped conversation index")
    build.set_defaults(command="build")

    search = subparsers.add_parser("search", help="Search the conversation index")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--refresh", action="store_true")
    search.set_defaults(command="search")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "build"
    index_path = args.output_dir / INDEX_JSON

    if command == "build":
        entries = build_index(args.project_root, args.codex_home, args.output_dir)
        print(f"Indexed {len(entries)} E:\\type10-7 conversation entries.")
        print(f"JSON: {args.output_dir / INDEX_JSON}")
        print(f"Markdown: {args.output_dir / INDEX_MD}")
        return 0

    if command == "search":
        if args.refresh or not index_path.exists():
            build_index(args.project_root, args.codex_home, args.output_dir)
        entries = load_index(index_path)
        print(render_search_results(search_entries(entries, args.query, args.limit)))
        return 0

    parser.error(f"Unknown command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
