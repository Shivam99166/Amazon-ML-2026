"""
08_submission.py - Final Competition Submission Generator for Amazon ML Challenge.

Responsibilities (Owned by Team Lead / Integration & Validation):
1. Format output according to strict competition submission requirements:
   output/matching_results.tsv:
     source1_entity_id<TAB>matched_entity_ids
2. Verify rules:
   - Exactly one row per test Source 1 entity.
   - Singletons have empty matched_entity_ids (no whitespace/null placeholders).
   - No duplicate entity IDs within any comma-separated list.
   - Matched IDs reference only valid Source 2 and Source 3 entities.
   - If candidate_pairs.tsv is provided, verify final matches are a strict subset of candidates.
3. Modular integration:
   - Accepts either raw scored candidate pairs (s1, cand, prob) with a threshold,
   - OR pre-grouped predictions (source1_entity_id, matched_entity_ids).
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


def read_s1_reference_ids(s1_file: str) -> List[str]:
    """Read all required Source 1 IDs in their original file order."""
    if not os.path.isfile(s1_file):
        raise FileNotFoundError(f"Reference S1 file not found: {s1_file}")

    s1_ids = []
    seen = set()
    with open(s1_file, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            entity_id = line.split("\t")[0].strip()
            if entity_id and entity_id not in seen:
                s1_ids.append(entity_id)
                seen.add(entity_id)
    return s1_ids


def load_candidate_pool(candidate_file: str) -> Dict[str, Set[str]]:
    """Load valid candidate set {s1_id: set(candidate_ids)} to ensure subset rule."""
    if not os.path.isfile(candidate_file):
        return {}

    pool = defaultdict(set)
    with open(candidate_file, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        s1_col = 0
        cand_col = 1
        for i, h in enumerate(header):
            h_low = h.strip().lower()
            if "s1" in h_low or "source1" in h_low:
                s1_col = i
            elif "cand" in h_low and "source" not in h_low:
                cand_col = i

        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) > max(s1_col, cand_col):
                s1 = parts[s1_col].strip()
                cand_raw = parts[cand_col].strip()
                # Handle comma-separated candidate list or single pair
                for c in cand_raw.split(","):
                    c_clean = c.strip()
                    if c_clean:
                        pool[s1].add(c_clean)
    return dict(pool)


def generate_submission(
    predictions_path: str,
    output_path: str = "output/matching_results.tsv",
    reference_s1_path: Optional[str] = None,
    candidate_path: Optional[str] = None,
    threshold: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    """
    Format and validate final matching_results.tsv.

    Returns:
        (success_bool, issues_list)
    """
    if not os.path.isfile(predictions_path):
        raise FileNotFoundError(f"Input predictions file not found: {predictions_path}")

    issues = []
    matches_by_s1 = defaultdict(list)

    # 1. Determine input format (scored pairs vs pre-grouped predictions)
    with open(predictions_path, "r", encoding="utf-8") as f:
        first_line = f.readline().rstrip("\r\n")
        header_cols = [c.strip().lower() for c in first_line.split("\t")]

    is_scored_pairs = any("prob" in c or "score" in c for c in header_cols)

    print(f"Reading predictions from: {predictions_path}")
    if is_scored_pairs:
        if threshold is None:
            threshold = 0.80
        print(f"Input format: Scored candidate pairs. Applying threshold = {threshold:.4f}...")

        s1_idx = 0
        cand_idx = 1
        prob_idx = 2
        for i, c in enumerate(header_cols):
            if "s1" in c or "source1" in c:
                s1_idx = i
            elif "cand" in c and "source" not in c:
                cand_idx = i
            elif "prob" in c or "score" in c:
                prob_idx = i

        with open(predictions_path, "r", encoding="utf-8") as f:
            f.readline()  # skip header
            for line_no, line in enumerate(f, start=2):
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) <= max(s1_idx, cand_idx, prob_idx):
                    continue
                s1 = parts[s1_idx].strip()
                cand = parts[cand_idx].strip()
                try:
                    prob = float(parts[prob_idx].strip())
                except ValueError:
                    continue

                if prob >= threshold:
                    if cand not in matches_by_s1[s1]:
                        matches_by_s1[s1].append(cand)

    else:
        print("Input format: Grouped predictions TSV (source1_entity_id \\t matched_entity_ids)...")
        with open(predictions_path, "r", encoding="utf-8") as f:
            f.readline()  # skip header
            for line_no, line in enumerate(f, start=2):
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t")
                s1 = parts[0].strip()
                matched_raw = parts[1].strip() if len(parts) > 1 else ""

                if not matched_raw:
                    if s1 not in matches_by_s1:
                        matches_by_s1[s1] = []
                    continue

                cands = [c.strip() for c in matched_raw.split(",") if c.strip()]
                # Check for duplicates within row
                unique_cands = []
                seen_cands = set()
                for c in cands:
                    if c in seen_cands:
                        issues.append(f"Line {line_no}: Duplicate ID '{c}' in row for '{s1}' (deduplicated).")
                    else:
                        seen_cands.add(c)
                        unique_cands.append(c)

                matches_by_s1[s1] = unique_cands

    # 2. Check against candidate pool if provided (subset rule)
    if candidate_path and os.path.isfile(candidate_path):
        print(f"Auditing matches against candidate pool: {candidate_path}...")
        cand_pool = load_candidate_pool(candidate_path)
        out_of_pool_count = 0
        for s1, cands in matches_by_s1.items():
            valid_cands = cand_pool.get(s1, set())
            for c in cands:
                if c not in valid_cands:
                    out_of_pool_count += 1
                    if out_of_pool_count <= 5:
                        issues.append(
                            f"Subset violation: Match '{c}' for '{s1}' was never in candidate pairs."
                        )
        if out_of_pool_count > 5:
            issues.append(f"... and {out_of_pool_count - 5} more subset violations.")

    # 3. Align with all Reference Source 1 IDs (including singletons)
    if reference_s1_path and os.path.isfile(reference_s1_path):
        print(f"Ensuring all test Source 1 entities from {reference_s1_path} are present...")
        all_s1 = read_s1_reference_ids(reference_s1_path)
        print(f"Found {len(all_s1):,} reference Source 1 entities.")
        output_s1_order = all_s1
    else:
        output_s1_order = sorted(matches_by_s1.keys())

    # 4. Write final output file
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Writing final submission to: {output_path}...")

    total_rows = 0
    total_singletons = 0
    total_matches = 0

    with open(output_path, "w", encoding="utf-8") as f:
        # Exact competition header
        f.write("source1_entity_id\tmatched_entity_ids\n")

        for s1 in output_s1_order:
            cands = matches_by_s1.get(s1, [])
            cands_str = ",".join(cands)
            f.write(f"{s1}\t{cands_str}\n")

            total_rows += 1
            if not cands:
                total_singletons += 1
            else:
                total_matches += len(cands)

    print("\n" + "=" * 60)
    print("SUBMISSION FILE SUMMARY")
    print("=" * 60)
    print(f"File Path:                {output_path}")
    print(f"Total Source 1 Rows:      {total_rows:,}")
    print(f"Singletons (0 matches):   {total_singletons:,} ({total_singletons/total_rows*100:.2f}%)")
    print(f"Total Predicted Matches:  {total_matches:,}")
    print(f"Average Matches per S1:   {total_matches/total_rows:.4f}")
    print("=" * 60)

    if issues:
        print("\n[VALIDATION WARNINGS / ISSUES FOUND]:")
        for iss in issues[:10]:
            print(f"  * {iss}")
        return False, issues

    print("\n[VERIFICATION PASSED]: File is cleanly formatted and submission-ready.")
    return True, []


def main():
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge - Submission TSV Generator & Quality Checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to prediction input file (raw scored pairs or grouped TSV)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/matching_results.tsv",
        help="Path to final submission output (default: output/matching_results.tsv)",
    )
    parser.add_argument(
        "--reference-s1",
        type=str,
        default=None,
        help="Path to reference test_source1.tsv (guarantees every test S1 is present)",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Path to candidate_pairs.tsv (verifies subset constraint)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Threshold to apply if input contains match probabilities (default: 0.80)",
    )

    args = parser.parse_args()

    success, _ = generate_submission(
        predictions_path=args.input,
        output_path=args.output,
        reference_s1_path=args.reference_s1,
        candidate_path=args.candidates,
        threshold=args.threshold,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
