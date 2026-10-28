"""Read-only Full WiSig ZIP/NumPy validation; never constructs experiment roles.

CRC is checked while reading each complete ZIP member. Official counts are a
reference, not evidence of historical packet independence or model eligibility.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from analyze_wisig_full_eq_capacity import CountsOnly, OLD, NEW, TARGET_RX


MEMBER = re.compile(r"pkl_wifi_eq_(2021_03_\d{2})/dataset_\1_node([\d-]+)\.pkl")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def unique_rows(x):
    """Count exact rows without packet hashes or persistent identity claims."""
    if not len(x):
        return 0
    y = np.ascontiguousarray(x).reshape(len(x), 512)
    return int(len(np.unique(y.view(np.dtype((np.void, y.dtype.itemsize * 512))).reshape(-1))))


def capacity(matrix, days, receivers, transmitters, names):
    indices = [transmitters.index(t) for t in OLD + names]
    selected = np.take(np.take(matrix, [receivers.index(r) for r in TARGET_RX], axis=1), indices, axis=2)
    required = np.array([520, 464, 540, 572], dtype=np.int64)
    shortages = []
    for di, ri, ti in np.argwhere(selected < required[:, None, None]):
        shortages.append(dict(date=days[di], receiver=TARGET_RX[ri], transmitter=(OLD + names)[ti],
                              available=int(selected[di, ri, ti]), required=int(required[di])))
    return dict(minimum_per_date=selected.min(axis=(1, 2)).astype(int).tolist(),
                minimum_four_date_total=int(selected.sum(axis=0).min()),
                total_selected_records=int(selected.sum()),
                r07_required_per_date=required.tolist(), r07_capacity_feasible=not shortages,
                shortage_cells=len(shortages), shortages=shortages)


def validate(directory, metadata, planning, output):
    output.mkdir(parents=True, exist_ok=False)
    with metadata.open("rb") as f:
        reference = CountsOnly(f).load()
    days = list(map(str, reference["capture_date_list"]))
    receivers = list(map(str, reference["rx_list"]))
    transmitters = list(map(str, reference["tx_list"]))
    expected = np.asarray(reference["mat_date_eq"], dtype=np.int64)
    assert expected.shape == (len(days), len(receivers), len(transmitters))
    proposal = json.loads(planning.read_text(encoding="utf-8"))["primary_candidate"]["proposed_new_classes"]
    checked_classes = set(OLD + NEW + proposal)
    files = sorted(directory.glob("*.zip"))
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
    pending_before = sorted(p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() != ".zip")
    actual = np.zeros_like(expected)
    distinct = np.zeros_like(expected)
    seen = Counter()
    errors = []
    archives = []
    rows = []
    duplicate_rows = []
    total_records = 0
    expected_members = {(days[d], receivers[r]) for d, r in np.argwhere(expected.sum(axis=2) > 0)}
    with (output / "members.jsonl").open("w", encoding="utf-8") as log:
        for path in files:
            archive = dict(file=path.name, compressed_bytes=before[path.name][0], members=0, uncompressed_bytes=0, errors=0)
            try:
                with zipfile.ZipFile(path) as z:
                    for info in z.infolist():
                        if info.is_dir():
                            continue
                        row = dict(archive=path.name, member=info.filename, uncompressed_bytes=info.file_size)
                        archive["members"] += 1
                        archive["uncompressed_bytes"] += info.file_size
                        try:
                            match = MEMBER.fullmatch(info.filename)
                            if not match:
                                raise ValueError("Unexpected member path/layout")
                            day, receiver = match.groups()
                            di, ri = days.index(day), receivers.index(receiver)
                            seen[(day, receiver)] += 1
                            if seen[(day, receiver)] != 1:
                                raise ValueError("Duplicate date/receiver member")
                            # Reading to EOF verifies ZIP decompression and its stored CRC.
                            raw = z.read(info)
                            stream = io.BytesIO(raw)
                            data = CountsOnly(stream).load()
                            if stream.read(1):
                                raise ValueError("Trailing bytes after pickle object")
                            del stream, raw
                            nodes, arrays = data["node_list"], data["data"]
                            if not isinstance(nodes, list) or not isinstance(arrays, list) or len(nodes) != len(arrays):
                                raise ValueError("Unequal node_list/data lists")
                            if len(set(nodes)) != len(nodes):
                                raise ValueError("Duplicate transmitter entries")
                            dtypes = Counter()
                            n = zero = bad = 0
                            max_abs = 0.0
                            squared_sum = 0.0
                            for node, x in zip(nodes, arrays):
                                ti = transmitters.index(str(node))
                                if not isinstance(x, np.ndarray) or x.ndim != 3 or x.shape[1:] != (256, 2) or x.dtype.kind != "f":
                                    raise ValueError(f"Unexpected IQ shape/dtype: {node} {getattr(x, 'shape', None)} {getattr(x, 'dtype', None)}")
                                actual[di, ri, ti] = len(x)
                                n += len(x)
                                dtypes[str(x.dtype)] += len(x)
                                bad += int(np.count_nonzero(~np.isfinite(x)))
                                zero += int(np.count_nonzero(~np.any(x != 0, axis=(1, 2))))
                                if len(x):
                                    max_abs = max(max_abs, float(np.max(np.abs(x))))
                                    squared_sum += float(np.sum(np.square(x), dtype=np.float64))
                                if receiver in TARGET_RX and node in checked_classes:
                                    nd = unique_rows(x)
                                    distinct[di, ri, ti] = nd
                                    if nd != len(x):
                                        duplicate_rows.append(dict(date=day, receiver=receiver, transmitter=node,
                                                                   records=len(x), exact_distinct_rows=nd))
                            if bad or zero:
                                raise ValueError(f"Invalid IQ: nonfinite_values={bad}, all_zero_records={zero}")
                            if max_abs > np.finfo(np.float32).max:
                                raise ValueError("IQ overflows FP32")
                            mismatch = np.flatnonzero(actual[di, ri] != expected[di, ri])
                            row.update(date=day, receiver=receiver, transmitter_entries=len(nodes), records=n,
                                       dtype_records=dict(dtypes), shape="N,256,2", nonfinite_values=bad,
                                       all_zero_records=zero, max_abs=max_abs,
                                       mean_squared_component=squared_sum / (n * 512) if n else None,
                                       count_mismatch_TX=[transmitters[i] for i in mismatch],
                                       CRC="PASS", status="PASS" if not len(mismatch) else "COUNT_MISMATCH")
                            if len(mismatch):
                                raise ValueError(f"Official count mismatch for {len(mismatch)} TX")
                            total_records += n
                            del data, nodes, arrays, x
                            gc.collect()
                        except Exception as exc:
                            row.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
                            errors.append(dict(archive=path.name, member=info.filename, error=row["error"]))
                            archive["errors"] += 1
                        rows.append(row)
                        log.write(json.dumps(row, ensure_ascii=False) + "\n")
                        log.flush()
                        print(json.dumps(dict(checked_members=len(rows), expected_members=len(expected_members),
                                              date=row.get("date"), receiver=row.get("receiver"),
                                              status=row["status"], records=total_records)), flush=True)
            except Exception as exc:
                errors.append(dict(archive=path.name, error=f"{type(exc).__name__}: {exc}"))
                archive["errors"] += 1
            archives.append(archive)
    after_files = sorted(directory.glob("*.zip"))
    after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in after_files}
    pending_after = sorted(p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() != ".zip")
    changed = sorted(set(before) ^ set(after) | {n for n in set(before) & set(after) if before[n] != after[n]})
    missing = sorted(expected_members - set(seen))
    extra = sorted(set(seen) - expected_members)
    count_mismatches = int(np.count_nonzero(actual != expected))
    success = not (errors or pending_before or pending_after or changed or missing or extra or count_mismatches)
    report = dict(schema="wisig_full_eq_archive_validation_v1", time_UTC=datetime.now(timezone.utc).isoformat(),
                  status="VERIFIED" if success else "INCOMPLETE_OR_FAILED", dataset_directory=str(directory),
                  scope="All completed ZIP members and all IQ values; exact-row duplicate checks only within each selected target RX/TX/date cell",
                  archives=len(files), compressed_bytes=sum(a["compressed_bytes"] for a in archives),
                  uncompressed_bytes=sum(a["uncompressed_bytes"] for a in archives),
                  members_checked=len(rows), expected_nonempty_RX_date_members=len(expected_members),
                  official_receiver_universe=len(receivers), official_transmitter_universe=len(transmitters), dates=days,
                  successful_records=total_records, expected_records=int(expected.sum()),
                  records_per_date=actual.sum(axis=(1, 2)).tolist(), expected_records_per_date=expected.sum(axis=(1, 2)).tolist(),
                  dtype_records=dict(sum((Counter(r.get("dtype_records", {})) for r in rows), Counter())),
                  missing_members=missing, extra_members=extra, duplicate_member_keys=[list(k) for k, v in seen.items() if v > 1],
                  pending_before=pending_before, pending_after=pending_after, changed_archives=changed,
                  count_mismatch_cells=count_mismatches, errors=errors,
                  exact_duplicate_selected_cells=duplicate_rows,
                  original_classes_capacity=capacity(actual, days, receivers, transmitters, NEW),
                  candidate_classes_capacity=capacity(actual, days, receivers, transmitters, proposal),
                  candidate_classes_exact_distinct_capacity=capacity(distinct, days, receivers, transmitters, proposal),
                  new_classes_candidate=proposal, archive_results=archives,
                  limitations=["Official metadata comparison is not a publisher-signed content authenticity proof.",
                               "Full per-RX PKL has data/node_list only; do not pass it to the compact loader with invented singleton RX/date/eq roles.",
                               "No cross-RX/date/TX or legacy ManyTx physical-packet independence has been established by this count/format check.",
                               "Byte-distinct IQ rows do not establish statistical independence or unique transmissions across receiver captures.",
                               "No source roles, selected class list, support/query allocation, normalization or scientific benchmark are changed."],
                  experiment_launches=0, model_reads=0, model_predictions=0, channel_generation=0,
                  training_allowed=False, user_hold_preserved=True)
    write_json(output / "summary.json", report)
    write_json(output / "counts.json", dict(dates=days, receivers=receivers, transmitters=transmitters, records=actual.tolist()))
    print(json.dumps(dict(status=report["status"], output=str(output), records=total_records,
                          errors=len(errors), count_mismatch_cells=count_mismatches), ensure_ascii=False), flush=True)
    return 0 if success else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--planning", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    raise SystemExit(validate(a.directory, a.metadata, a.planning, a.output))


if __name__ == "__main__":
    main()
