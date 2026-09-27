"""
01b_gt_analysis.py - Ground Truth Analysis for Amazon ML Challenge.

Analyzes dataset/train/train_ground_truth.tsv in a streaming, memory-conscious
fashion without loading unnecessary datasets or exploding RAM.

Outputs summary metrics to stdout and saves output/ground_truth_analysis.txt.
"""

import os
import sys
from collections import Counter
from pathlib import Path


def analyze_ground_truth(gt_path: str, output_path: str = None) -> dict:
    """Stream ground-truth TSV and compute entity matching statistics."""
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    total_s1 = 0
    zero_matches = 0
    match_count_freq = Counter()

    s1_with_s2 = 0
    s1_with_s3 = 0
    s1_with_both = 0

    unique_s2 = set()
    unique_s3 = set()

    total_matches_sum = 0

    print(f"Reading ground truth from: {gt_path}")
    print("Streaming rows for memory efficiency...")

    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        if len(header) < 2 or "source1_entity_id" not in header[0]:
            print(f"Warning: Unexpected header format: {header}")

        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue

            parts = line.split("\t")
            s1_id = parts[0].strip()
            total_s1 += 1

            matched_raw = parts[1].strip() if len(parts) > 1 else ""

            if not matched_raw:
                zero_matches += 1
                match_count_freq[0] += 1
                continue

            # Parse comma-separated match IDs
            matches = [m.strip() for m in matched_raw.split(",") if m.strip()]
            num_matches = len(matches)
            match_count_freq[num_matches] += 1
            total_matches_sum += num_matches

            has_s2 = False
            has_s3 = False

            for m in matches:
                if m.startswith("S2-"):
                    has_s2 = True
                    unique_s2.add(m)
                elif m.startswith("S3-"):
                    has_s3 = True
                    unique_s3.add(m)
                else:
                    # In case prefix is lowercase or custom
                    if "S2" in m:
                        has_s2 = True
                        unique_s2.add(m)
                    elif "S3" in m:
                        has_s3 = True
                        unique_s3.add(m)

            if has_s2:
                s1_with_s2 += 1
            if has_s3:
                s1_with_s3 += 1
            if has_s2 and has_s3:
                s1_with_both += 1

    # Compute statistics
    zero_match_pct = (zero_matches / total_s1 * 100) if total_s1 > 0 else 0.0
    exactly_1 = match_count_freq[1]
    exactly_2 = match_count_freq[2]
    exactly_3 = match_count_freq[3]
    avg_matches = (total_matches_sum / total_s1) if total_s1 > 0 else 0.0
    max_matches = max(match_count_freq.keys()) if match_count_freq else 0

    # Compute median from frequency distribution
    median_matches = 0.0
    if total_s1 > 0:
        cum_count = 0
        sorted_counts = sorted(match_count_freq.items())
        mid_left = (total_s1 + 1) // 2
        mid_right = (total_s1 + 2) // 2
        left_val = None
        right_val = None

        for count_val, freq in sorted_counts:
            cum_count += freq
            if left_val is None and cum_count >= mid_left:
                left_val = count_val
            if right_val is None and cum_count >= mid_right:
                right_val = count_val
                break
        median_matches = (left_val + right_val) / 2.0

    report_lines = [
        "GROUND TRUTH ANALYSIS",
        "=====================",
        f"Total S1 entities:       {total_s1:,}",
        f"Zero matches:            {zero_matches:,}",
        f"Zero-match percentage:   {zero_match_pct:.2f}%",
        f"Exactly 1:               {exactly_1:,}",
        f"Exactly 2:               {exactly_2:,}",
        f"Exactly 3:               {exactly_3:,}",
        f"Average matches:         {avg_matches:.4f}",
        f"Median matches:          {median_matches:.1f}",
        f"Maximum matches:         {max_matches}",
        f"S1 with S2 matches:      {s1_with_s2:,}",
        f"S1 with S3 matches:      {s1_with_s3:,}",
        f"S1 with both:            {s1_with_both:,}",
        f"Unique S2 IDs:           {len(unique_s2):,}",
        f"Unique S3 IDs:           {len(unique_s3):,}",
        "",
        "Match Count Distribution:",
        "-------------------------",
    ]

    for count_val in sorted(match_count_freq.keys()):
        freq = match_count_freq[count_val]
        pct = (freq / total_s1 * 100) if total_s1 > 0 else 0.0
        report_lines.append(f"  {count_val:2d} matches: {freq:10,d} ({pct:6.2f}%)")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # Save summary file if requested
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
        print(f"\nSaved analysis summary to: {output_path}")

    return {
        "total_s1": total_s1,
        "zero_matches": zero_matches,
        "zero_match_pct": zero_match_pct,
        "exactly_1": exactly_1,
        "exactly_2": exactly_2,
        "exactly_3": exactly_3,
        "avg_matches": avg_matches,
        "median_matches": median_matches,
        "max_matches": max_matches,
        "s1_with_s2": s1_with_s2,
        "s1_with_s3": s1_with_s3,
        "s1_with_both": s1_with_both,
        "unique_s2": len(unique_s2),
        "unique_s3": len(unique_s3),
        "distribution": dict(match_count_freq),
    }


def main():
    base_dir = Path(__file__).resolve().parent.parent
    gt_file = base_dir / "data" / "train" / "train_ground_truth.tsv"
    out_file = base_dir / "output" / "ground_truth_analysis.txt"

    if len(sys.argv) > 1:
        gt_file = Path(sys.argv[1])
    if len(sys.argv) > 2:
        out_file = Path(sys.argv[2])

    analyze_ground_truth(str(gt_file), str(out_file))


if __name__ == "__main__":
    main()
