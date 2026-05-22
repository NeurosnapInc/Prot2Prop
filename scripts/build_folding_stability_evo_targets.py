#!/usr/bin/env python3

import argparse
import csv
import difflib
import gzip
import json
import math
from pathlib import Path

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}


def open_text(path, mode="rt"):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def clean_raw_sequence(seq):
    out = []
    for ch in str(seq).strip():
        if ch in "-.":
            continue
        if ch.isalpha():
            out.append(ch.upper())
    return "".join(out)


def clean_aligned_sequence(seq):
    out = []
    for ch in str(seq).strip():
        if ch == ".":
            continue
        if ch == "-":
            out.append("-")
        elif ch.isalpha():
            out.append(ch.upper())
    return "".join(out)


def clean_match_sequence(seq):
    out = []
    for ch in str(seq).strip():
        if ch == ".":
            continue
        if ch == "-":
            out.append("-")
        elif ch.isupper() and ch.isalpha():
            out.append(ch)
    return "".join(out)


def ungap(aligned_seq):
    return aligned_seq.replace("-", "")


def read_rows(path):
    with open_text(path, "rt") as handle:
        return list(csv.DictReader(handle))


def read_fasta(path):
    records = []
    name = None
    chunks = []

    with open_text(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(chunks)))
                name = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)

    if name is not None:
        records.append((name, "".join(chunks)))

    return records


def parse_int_or_none(value):
    if value is None or str(value).strip() == "":
        return None
    return int(float(value))


def find_existing_files(root, filenames):
    root = Path(root)
    filenames = {str(x) for x in filenames if str(x)}
    found = {}

    for name in filenames:
        for candidate in (root / name, root / f"{name}.gz"):
            if candidate.exists():
                found[name] = candidate
                break

    missing = filenames - set(found)
    if missing:
        for path in root.rglob("*"):
            if not path.is_file():
                continue

            name = path.name
            raw_name = name[:-3] if name.endswith(".gz") else name
            if raw_name in missing:
                found[raw_name] = path
                missing.remove(raw_name)
                if not missing:
                    break

    return found


def load_homolog_groups(path):
    groups = {}

    with open_text(path, "rt") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            profile_id = row.get("profile_id") or row.get("DMS_id")
            wt_sequence = clean_raw_sequence(row.get("wt_sequence", ""))

            if not profile_id:
                raise ValueError(f"Missing profile_id/DMS_id on homolog JSONL line {line_no}")
            if not wt_sequence:
                raise ValueError(f"Missing wt_sequence on homolog JSONL line {line_no}")

            groups[profile_id] = {
                "task_name": row.get("task_name", "folding_stability"),
                "profile_id": profile_id,
                "source": row.get("source") or f"ProteinGym/folding_stability:{profile_id}",
                "wt_sequence": wt_sequence,
                "msa_path": row.get("msa_path"),
                "aligned_wt": row.get("aligned_wt"),
                "homologs": row.get("homologs", []),
                "msa_start": parse_int_or_none(row.get("MSA_start") or row.get("msa_start")),
                "msa_end": parse_int_or_none(row.get("MSA_end") or row.get("msa_end")),
            }

    return groups


def load_aligned_rows_for_group(group):
    wt_sequence = clean_raw_sequence(group["wt_sequence"])
    msa_start = group.get("msa_start") or 1
    msa_end = group.get("msa_end") or len(wt_sequence)
    expected_msa_sequence = wt_sequence[msa_start - 1:msa_end]

    if group.get("msa_path"):
        records = read_fasta(group["msa_path"])
        if not records:
            raise ValueError(f"MSA file has no records: {group['msa_path']}")

        query_idx = None
        for i, (_, seq) in enumerate(records):
            if ungap(clean_aligned_sequence(seq)) == expected_msa_sequence:
                query_idx = i
                break

        if query_idx is None:
            raise ValueError(
                f"Could not find aligned WT row in MSA for profile_id={group['profile_id']} "
                f"covering MSA_start={msa_start} MSA_end={msa_end}"
            )

        aligned_wt = records[query_idx][1]
        aligned_homologs = [seq for i, (_, seq) in enumerate(records) if i != query_idx]
        return aligned_wt, aligned_homologs, msa_start - 1

    if group.get("aligned_wt"):
        aligned_wt = group["aligned_wt"]
        if ungap(clean_aligned_sequence(aligned_wt)) != expected_msa_sequence:
            raise ValueError(
                f"aligned_wt does not ungap to expected MSA-covered sequence for profile_id={group['profile_id']}"
            )
        aligned_homologs = list(group.get("homologs", []))
        if not aligned_homologs:
            raise ValueError(f"No aligned homologs for profile_id={group['profile_id']}")
        return aligned_wt, aligned_homologs, msa_start - 1

    raise ValueError(
        f"profile_id={group['profile_id']} must provide either msa_path or aligned_wt plus aligned homologs"
    )


def build_wt_position_nll_profile(group, pseudocount=0.5):
    wt_sequence = clean_raw_sequence(group["wt_sequence"])
    wt_len = len(wt_sequence)
    msa_start = group.get("msa_start") or 1
    msa_end = group.get("msa_end") or wt_len
    expected_msa_sequence = wt_sequence[msa_start - 1:msa_end]

    aligned_wt, aligned_homologs, start_offset = load_aligned_rows_for_group(group)

    if ungap(clean_aligned_sequence(aligned_wt)) != expected_msa_sequence:
        raise ValueError(f"Aligned WT does not match WT sequence for profile_id={group['profile_id']}")

    core_col_to_wt_pos = []
    wt_pos = start_offset - 1

    for ch in str(aligned_wt).strip():
        if ch == ".":
            continue
        if ch == "-":
            core_col_to_wt_pos.append(None)
        elif ch.isalpha():
            wt_pos += 1
            if wt_pos >= wt_len:
                raise ValueError(f"Aligned WT exceeds WT length for profile_id={group['profile_id']}")
            if ch.isupper():
                core_col_to_wt_pos.append(wt_pos)

    if wt_pos != msa_end - 1:
        raise ValueError(f"WT alignment length mismatch for profile_id={group['profile_id']}")

    counts = [[float(pseudocount) for _ in AA] for _ in range(wt_len)]
    observed = [0 for _ in range(wt_len)]

    for pos in range(msa_start - 1, msa_end):
        aa = wt_sequence[pos]
        if aa in AA_TO_IDX:
            counts[pos][AA_TO_IDX[aa]] += 1.0
            observed[pos] += 1

    used_homologs = 0
    skipped_homologs = 0

    for homolog in aligned_homologs:
        aligned_homolog = clean_match_sequence(homolog)

        if len(aligned_homolog) != len(core_col_to_wt_pos):
            skipped_homologs += 1
            continue

        used_this_homolog = False

        for col, hom_aa in enumerate(aligned_homolog):
            wt_position = core_col_to_wt_pos[col]
            if wt_position is None:
                continue
            if hom_aa in AA_TO_IDX:
                counts[wt_position][AA_TO_IDX[hom_aa]] += 1.0
                observed[wt_position] += 1
                used_this_homolog = True

        if used_this_homolog:
            used_homologs += 1
        else:
            skipped_homologs += 1

    nll_matrix = []
    position_mask = []

    for pos in range(wt_len):
        total = sum(counts[pos])
        probs = [c / total for c in counts[pos]]
        nll_matrix.append([-math.log(max(p, 1e-12)) for p in probs])
        position_mask.append(observed[pos] > 0 and wt_sequence[pos] in AA_TO_IDX)

    return {
        "nll_matrix": nll_matrix,
        "position_mask": position_mask,
        "used_homologs": used_homologs,
        "skipped_homologs": skipped_homologs,
    }


def variant_position_mapping(target_seq, mutated_sequence):
    target_seq = clean_raw_sequence(target_seq)
    mutated_sequence = clean_raw_sequence(mutated_sequence)

    if len(target_seq) == len(mutated_sequence):
        return list(range(len(mutated_sequence)))

    mapping = []
    matcher = difflib.SequenceMatcher(a=target_seq, b=mutated_sequence, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(j2 - j1):
                mapping.append(i1 + offset)
        elif tag == "delete":
            continue
        elif tag == "insert":
            for _ in range(j1, j2):
                mapping.append(None)
        elif tag == "replace":
            wt_len = i2 - i1
            mut_len = j2 - j1
            paired = min(wt_len, mut_len)

            for offset in range(paired):
                mapping.append(i1 + offset)
            for _ in range(mut_len - paired):
                mapping.append(None)

    if len(mapping) != len(mutated_sequence):
        raise ValueError(f"Variant mapping length mismatch: {len(mapping)} != {len(mutated_sequence)}")

    return mapping


def map_variant_to_alignment_targets(target_seq, mutated_sequence, profile):
    mutated_sequence = clean_raw_sequence(mutated_sequence)
    mapping = variant_position_mapping(target_seq, mutated_sequence)

    nll_matrix = profile["nll_matrix"]
    position_mask = profile["position_mask"]

    alignment_nll = []
    alignment_mask = []

    for mut_aa, wt_pos in zip(mutated_sequence, mapping):
        if wt_pos is None or mut_aa not in AA_TO_IDX:
            alignment_nll.append(0.0)
            alignment_mask.append(False)
            continue

        if wt_pos >= len(nll_matrix) or not position_mask[wt_pos]:
            alignment_nll.append(0.0)
            alignment_mask.append(False)
            continue

        alignment_nll.append(float(nll_matrix[wt_pos][AA_TO_IDX[mut_aa]]))
        alignment_mask.append(True)

    if len(alignment_nll) != len(mutated_sequence):
        raise ValueError("alignment_nll length mismatch")
    if len(alignment_mask) != len(mutated_sequence):
        raise ValueError("alignment_mask length mismatch")

    return alignment_nll, alignment_mask


def write_targets_for_csv(csv_path, group, out_handle, pseudocount):
    rows = read_rows(csv_path)
    if not rows:
        return 0, 0

    if "mutated_sequence" not in rows[0]:
        raise ValueError(f"{csv_path} missing required column: mutated_sequence")

    target_seq = clean_raw_sequence(group["wt_sequence"])
    profile = build_wt_position_nll_profile(group, pseudocount=pseudocount)

    written_rows = 0
    for row in rows:
        row_target = clean_raw_sequence(row.get("target_seq") or target_seq)
        if row_target != target_seq:
            raise ValueError(f"WT mismatch in {csv_path}: row target_seq != reference target_seq")

        mutated_sequence = clean_raw_sequence(row["mutated_sequence"])
        alignment_nll, alignment_mask = map_variant_to_alignment_targets(
            target_seq=target_seq,
            mutated_sequence=mutated_sequence,
            profile=profile,
        )

        out_row = {
            "task_name": "folding_stability",
            "DMS_id": row.get("DMS_id") or group["profile_id"],
            "profile_id": group["profile_id"],
            "source": group["source"],
            "sequence": mutated_sequence,
            "target_seq": target_seq,
            "mutant": row.get("mutant", ""),
            "DMS_score": row.get("DMS_score", ""),
            "DMS_score_bin": row.get("DMS_score_bin", ""),
            "aa_order": AA,
            "alignment_nll": alignment_nll,
            "alignment_mask": alignment_mask,
        }

        out_handle.write(json.dumps(out_row, separators=(",", ":")) + "\n")
        written_rows += 1

    return len(rows), written_rows


def build_targets_from_homologs(csv_path, homologs_path, out_path, pseudocount):
    homolog_groups = load_homolog_groups(homologs_path)
    rows = read_rows(csv_path)

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    written_rows = 0
    skipped_rows = 0

    with open_text(out_path, "wt") as out_handle:
        for row in rows:
            profile_id = row.get("DMS_id") or row.get("profile_id")
            if not profile_id and len(homolog_groups) == 1:
                profile_id = next(iter(homolog_groups))

            if profile_id not in homolog_groups:
                skipped_rows += 1
                continue

            group = homolog_groups[profile_id]
            group_rows = [row]
            tmp_path = None

            target_seq = clean_raw_sequence(row.get("target_seq") or group["wt_sequence"])
            if target_seq != clean_raw_sequence(group["wt_sequence"]):
                raise ValueError(f"WT mismatch for profile_id={profile_id}")

            profile = build_wt_position_nll_profile(group, pseudocount=pseudocount)
            mutated_sequence = clean_raw_sequence(row["mutated_sequence"])
            alignment_nll, alignment_mask = map_variant_to_alignment_targets(target_seq, mutated_sequence, profile)

            out_row = {
                "task_name": "folding_stability",
                "DMS_id": profile_id,
                "profile_id": profile_id,
                "source": group["source"],
                "sequence": mutated_sequence,
                "target_seq": target_seq,
                "mutant": row.get("mutant", ""),
                "DMS_score": row.get("DMS_score", ""),
                "DMS_score_bin": row.get("DMS_score_bin", ""),
                "aa_order": AA,
                "alignment_nll": alignment_nll,
                "alignment_mask": alignment_mask,
            }

            out_handle.write(json.dumps(out_row, separators=(",", ":")) + "\n")
            total_rows += 1
            written_rows += 1

    print(f"Read rows: {total_rows + skipped_rows}")
    print(f"Wrote rows: {written_rows}")
    print(f"Skipped rows: {skipped_rows}")
    print(f"Wrote: {out_path}")


def reference_by_filename(path):
    return {row["DMS_filename"]: row for row in read_rows(path)}


def build_targets_from_proteingym(
    manifest_path,
    substitutions_dir,
    indels_dir,
    substitutions_reference,
    indels_reference,
    msa_dir,
    out_path,
    pseudocount,
):
    manifest_rows = read_rows(manifest_path)
    substitutions_ref = reference_by_filename(substitutions_reference)
    indels_ref = reference_by_filename(indels_reference)

    substitution_files = {
        row["DMS_filename"] for row in manifest_rows
        if row["mutation_type"] == "substitutions"
    }
    indel_files = {
        row["DMS_filename"] for row in manifest_rows
        if row["mutation_type"] == "indels"
    }

    refs = []
    for row in manifest_rows:
        filename = row["DMS_filename"]
        mutation_type = row["mutation_type"]

        if mutation_type == "substitutions":
            ref = substitutions_ref.get(filename)
        elif mutation_type == "indels":
            ref = indels_ref.get(filename)
        else:
            raise ValueError(f"Unsupported mutation_type={mutation_type} for {filename}")

        if ref is None:
            raise ValueError(f"No ProteinGym reference row found for {filename}")

        refs.append((row, ref))

    substitution_paths = find_existing_files(substitutions_dir, substitution_files)
    indel_paths = find_existing_files(indels_dir, indel_files)
    msa_paths = find_existing_files(msa_dir, {ref["MSA_filename"] for _, ref in refs})

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    written_rows = 0
    processed_files = 0

    with open_text(out_path, "wt") as out_handle:
        for manifest_row, ref in refs:
            filename = manifest_row["DMS_filename"]
            mutation_type = manifest_row["mutation_type"]
            msa_filename = ref["MSA_filename"]

            if mutation_type == "substitutions":
                csv_path = substitution_paths.get(filename)
            else:
                csv_path = indel_paths.get(filename)

            if csv_path is None:
                raise FileNotFoundError(f"Could not find ProteinGym DMS CSV: {filename}")
            if msa_filename not in msa_paths:
                raise FileNotFoundError(f"Could not find ProteinGym MSA file: {msa_filename}")

            group = {
                "task_name": "folding_stability",
                "profile_id": ref["DMS_id"],
                "source": f"ProteinGym/folding_stability:{Path(filename).stem}",
                "wt_sequence": clean_raw_sequence(ref["target_seq"]),
                "msa_path": str(msa_paths[msa_filename]),
                "msa_start": parse_int_or_none(ref.get("MSA_start")),
                "msa_end": parse_int_or_none(ref.get("MSA_end")),
            }

            n_read, n_written = write_targets_for_csv(
                csv_path=csv_path,
                group=group,
                out_handle=out_handle,
                pseudocount=pseudocount,
            )

            total_rows += n_read
            written_rows += n_written
            processed_files += 1

    print(f"Processed files: {processed_files}")
    print(f"Read rows: {total_rows}")
    print(f"Wrote rows: {written_rows}")
    print(f"Wrote: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--homologs")
    parser.add_argument("--manifest")
    parser.add_argument("--substitutions-dir")
    parser.add_argument("--indels-dir")
    parser.add_argument("--substitutions-reference")
    parser.add_argument("--indels-reference")
    parser.add_argument("--msa-dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pseudocount", type=float, default=0.5)
    args = parser.parse_args()

    if args.manifest:
        required = [
            args.substitutions_dir,
            args.indels_dir,
            args.substitutions_reference,
            args.indels_reference,
            args.msa_dir,
        ]
        if any(x is None for x in required):
            parser.error(
                "--manifest mode requires --substitutions-dir, --indels-dir, "
                "--substitutions-reference, --indels-reference, and --msa-dir"
            )

        build_targets_from_proteingym(
            manifest_path=args.manifest,
            substitutions_dir=args.substitutions_dir,
            indels_dir=args.indels_dir,
            substitutions_reference=args.substitutions_reference,
            indels_reference=args.indels_reference,
            msa_dir=args.msa_dir,
            out_path=args.out,
            pseudocount=args.pseudocount,
        )
    else:
        if not args.csv or not args.homologs:
            parser.error("manual mode requires --csv and --homologs")

        build_targets_from_homologs(
            csv_path=args.csv,
            homologs_path=args.homologs,
            out_path=args.out,
            pseudocount=args.pseudocount,
        )


if __name__ == "__main__":
    main()