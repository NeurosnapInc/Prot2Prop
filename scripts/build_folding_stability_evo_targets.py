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


def clean_raw_sequence(seq: str) -> str:
    out = []
    for ch in str(seq).strip():
        if ch in "-.":
            continue
        if ch.isalpha():
            out.append(ch.upper())
    return "".join(out)


def clean_aligned_sequence(seq: str) -> str:
    out = []
    for ch in str(seq).strip():
        if ch == ".":
            continue
        if ch == "-":
            out.append("-")
        elif ch.isupper() and ch.isalpha():
            out.append(ch)
        elif ch.islower():
            continue
    return "".join(out)


def ungap(aligned_seq: str) -> str:
    return aligned_seq.replace("-", "")


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
                "wt_sequence": wt_sequence,
                "msa_path": row.get("msa_path"),
                "aligned_wt": row.get("aligned_wt"),
                "homologs": row.get("homologs", []),
            }

    return groups


def load_aligned_rows_for_group(group):
    wt_sequence = clean_raw_sequence(group["wt_sequence"])

    if group.get("msa_path"):
        records = read_fasta(group["msa_path"])
        if not records:
            raise ValueError(f"MSA file has no records: {group['msa_path']}")

        aligned_records = [(name, clean_aligned_sequence(seq)) for name, seq in records]
        query_idx = None

        for i, (_, aligned_seq) in enumerate(aligned_records):
            if ungap(aligned_seq) == wt_sequence:
                query_idx = i
                break

        if query_idx is None:
            raise ValueError(
                f"Could not find a query/aligned WT row in MSA for profile_id={group['profile_id']}"
            )

        aligned_wt = aligned_records[query_idx][1]
        aligned_homologs = [
            seq for i, (_, seq) in enumerate(aligned_records)
            if i != query_idx
        ]
        return aligned_wt, aligned_homologs

    if group.get("aligned_wt"):
        aligned_wt = clean_aligned_sequence(group["aligned_wt"])
        if ungap(aligned_wt) != wt_sequence:
            raise ValueError(
                f"aligned_wt does not ungap to wt_sequence for profile_id={group['profile_id']}"
            )
        aligned_homologs = [clean_aligned_sequence(x) for x in group.get("homologs", [])]
        return aligned_wt, aligned_homologs

    raise ValueError(
        f"profile_id={group['profile_id']} must provide either msa_path or aligned_wt plus aligned homologs"
    )


def build_wt_position_nll_profile(group, pseudocount=0.5):
    wt_sequence = clean_raw_sequence(group["wt_sequence"])
    wt_len = len(wt_sequence)

    aligned_wt, aligned_homologs = load_aligned_rows_for_group(group)

    if ungap(clean_aligned_sequence(aligned_wt)) != wt_sequence:
        raise ValueError(f"Aligned WT does not match WT sequence for profile_id={group['profile_id']}")

    col_to_wt_pos = []
    wt_pos = -1

    for ch in clean_aligned_sequence(aligned_wt):
        if ch == "-":
            col_to_wt_pos.append(None)
        else:
            wt_pos += 1
            col_to_wt_pos.append(wt_pos)

    if wt_pos + 1 != wt_len:
        raise ValueError(f"WT alignment length mismatch for profile_id={group['profile_id']}")

    counts = [[float(pseudocount) for _ in AA] for _ in range(wt_len)]
    observed = [0 for _ in range(wt_len)]

    for pos, aa in enumerate(wt_sequence):
        if aa in AA_TO_IDX:
            counts[pos][AA_TO_IDX[aa]] += 1.0
            observed[pos] += 1

    used_homologs = 0
    skipped_homologs = 0

    for homolog in aligned_homologs:
        aligned_homolog = clean_aligned_sequence(homolog)

        if len(aligned_homolog) != len(col_to_wt_pos):
            raise ValueError(f"Aligned homolog length mismatch for profile_id={group['profile_id']}")

        used_this_homolog = False

        for col, hom_aa in enumerate(aligned_homolog):
            wt_position = col_to_wt_pos[col]
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

    matcher = difflib.SequenceMatcher(
        a=target_seq,
        b=mutated_sequence,
        autojunk=False,
    )

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(j2 - j1):
                mapping.append(i1 + offset)

        elif tag == "delete":
            continue

        elif tag == "insert":
            # Mutated residues with no WT position.
            for _ in range(j1, j2):
                mapping.append(None)

        elif tag == "replace":
            wt_len = i2 - i1
            mut_len = j2 - j1
            paired = min(wt_len, mut_len)
            # Paired replace positions are substitutions.
            for offset in range(paired):
                mapping.append(i1 + offset)
            # Extra mutated residues are insertions.
            for _ in range(mut_len - paired):
                mapping.append(None)
            # Extra WT residues are deletions.

    if len(mapping) != len(mutated_sequence):
        raise ValueError(
            f"Variant mapping length mismatch: {len(mapping)} != {len(mutated_sequence)}"
        )

    return mapping


def map_variant_to_alignment_targets(target_seq, mutated_sequence, profile):
    mutated_sequence = clean_raw_sequence(mutated_sequence)
    mapping = variant_position_mapping(target_seq, mutated_sequence)

    nll_matrix = profile["nll_matrix"]
    position_mask = profile["position_mask"]

    alignment_nll = []
    alignment_mask = []

    for mut_aa, wt_pos in zip(mutated_sequence, mapping):
        if wt_pos is None:
            alignment_nll.append(0.0)
            alignment_mask.append(False)
            continue

        if mut_aa not in AA_TO_IDX:
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


def build_targets(csv_path, homologs_path, out_path, pseudocount):
    homolog_groups = load_homolog_groups(homologs_path)
    profile_cache = {}

    total_rows = 0
    written_rows = 0
    skipped_rows = 0

    with open_text(csv_path, "rt") as in_handle, open_text(out_path, "wt") as out_handle:
        reader = csv.DictReader(in_handle)

        required = {"DMS_id", "target_seq", "mutated_sequence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            total_rows += 1

            profile_id = row["DMS_id"]
            target_seq = clean_raw_sequence(row["target_seq"])
            mutated_sequence = clean_raw_sequence(row["mutated_sequence"])

            if profile_id not in homolog_groups:
                skipped_rows += 1
                print(f"Skipping row {total_rows}: no homolog group for DMS_id={profile_id}")
                continue

            group = homolog_groups[profile_id]
            group_wt = clean_raw_sequence(group["wt_sequence"])

            if group_wt != target_seq:
                raise ValueError(
                    f"WT mismatch for {profile_id}: CSV target_seq != homolog wt_sequence"
                )

            if profile_id not in profile_cache:
                profile_cache[profile_id] = build_wt_position_nll_profile(
                    group=group,
                    pseudocount=pseudocount,
                )

            profile = profile_cache[profile_id]

            alignment_nll, alignment_mask = map_variant_to_alignment_targets(
                target_seq=target_seq,
                mutated_sequence=mutated_sequence,
                profile=profile,
            )

            # Long out_row just in case the extra data is helpful (might need trimming)
            out_row = {
                "task_name": "folding_stability",
                "DMS_id": profile_id,
                "profile_id": profile_id,
                "source": f"ProteinGym/folding_stability:{profile_id}",

                "sequence": mutated_sequence,

                "target_seq": target_seq,
                "mutant": row.get("mutant", ""),
                "DMS_score": row.get("DMS_score", ""),
                "DMS_score_bin": row.get("DMS_score_bin", ""),

                "aa_order": AA,
                "alignment_nll": alignment_nll,
                "alignment_mask": alignment_mask,
                "sequence_length": len(mutated_sequence),
                "num_valid_positions": int(sum(alignment_mask)),
            }

            out_handle.write(json.dumps(out_row, separators=(",", ":")) + "\n")
            written_rows += 1

    print(f"Read rows: {total_rows}")
    print(f"Wrote rows: {written_rows}")
    print(f"Skipped rows: {skipped_rows}")
    print(f"Wrote: {out_path}")

    for profile_id, profile in profile_cache.items():
        print(
            f"{profile_id}: used_homologs={profile['used_homologs']} "
            f"skipped_homologs={profile['skipped_homologs']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--homologs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pseudocount", type=float, default=0.5)
    args = parser.parse_args()

    build_targets(
        csv_path=args.csv,
        homologs_path=args.homologs,
        out_path=args.out,
        pseudocount=args.pseudocount,
    )

if __name__ == "__main__":
    main()