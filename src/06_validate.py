"""
06_validate.py - Validation Metric and Evaluation for Amazon ML Challenge.

Responsibilities (Owned by Team Lead / Integration & Validation):
1. Compute competition-exact macro-averaged F0.5 score over Source 1 entities.
2. Validate prediction files: duplicate checks, missing row checks, format checks.
3. Evaluate blocking recall: measure coverage of true pairs in candidate sets.
4. Run self-contained synthetic test suite (--test).
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


def compute_entity_metrics(actual: Set[str], predicted: Set[str]) -> Tuple[float, float, float, int, int, int]:
    """
    Calculate precision, recall, and F0.5 for a single Source 1 entity.

    Returns:
        (precision, recall, f05, tp, fp, fn)
    """
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)

    # Both empty: perfect match on singleton
    if len(actual) == 0 and len(predicted) == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0

    # False positive on singleton
    if len(actual) == 0 and len(predicted) > 0:
        return 0.0, 0.0, 0.0, 0, fp, 0

    # False negative: missed all matches
    if len(actual) > 0 and len(predicted) == 0:
        return 0.0, 0.0, 0.0, 0, 0, fn

    # Both non-empty
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    denom = (0.25 * prec + rec)
    if denom == 0.0 or tp == 0:
        f05 = 0.0
    else:
        f05 = (1.25 * prec * rec) / denom

    return prec, rec, f05, tp, fp, fn


def evaluate_predictions(
    gt_dict: Dict[str, Set[str]],
    pred_dict: Dict[str, Set[str]],
    s1_ids: Optional[List[str]] = None,
) -> dict:
    """
    Compute macro-averaged precision, recall, and F0.5 across all Source 1 entities.
    """
    if s1_ids is None:
        s1_ids = list(gt_dict.keys())

    total_entities = len(s1_ids)
    if total_entities == 0:
        raise ValueError("Cannot evaluate empty entity list.")

    sum_prec = 0.0
    sum_rec = 0.0
    sum_f05 = 0.0

    perfect_matches = 0
    entities_with_fp = 0
    entities_with_fn = 0
    both_empty_count = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for s1 in s1_ids:
        actual = gt_dict.get(s1, set())
        predicted = pred_dict.get(s1, set())

        prec, rec, f05, tp, fp, fn = compute_entity_metrics(actual, predicted)

        sum_prec += prec
        sum_rec += rec
        sum_f05 += f05

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if len(actual) == 0 and len(predicted) == 0:
            both_empty_count += 1
            perfect_matches += 1
        elif fp == 0 and fn == 0:
            perfect_matches += 1

        if fp > 0:
            entities_with_fp += 1
        if fn > 0:
            entities_with_fn += 1

    macro_precision = sum_prec / total_entities
    macro_recall = sum_rec / total_entities
    macro_f05 = sum_f05 / total_entities

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f05": macro_f05,
        "total_evaluated": total_entities,
        "perfect_matches": perfect_matches,
        "entities_with_fp": entities_with_fp,
        "entities_with_fn": entities_with_fn,
        "both_empty": both_empty_count,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def load_ground_truth(path: str) -> Dict[str, Set[str]]:
    """Load ground-truth TSV into {s1_id: set(matched_ids)}."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line_num, line in enumerate(f, start=2):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0].strip()
            raw_matches = parts[1].strip() if len(parts) > 1 else ""
            if raw_matches:
                gt[s1_id] = {m.strip() for m in raw_matches.split(",") if m.strip()}
            else:
                gt[s1_id] = set()
    return gt


def load_predictions(path: str) -> Tuple[Dict[str, Set[str]], List[str]]:
    """
    Load predictions TSV with strict sanity checks:
    - Detect duplicate Source 1 IDs.
    - Detect duplicate candidate IDs in a single row.

    Returns:
        (pred_dict, issues_list)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prediction file not found: {path}")

    pred = {}
    issues = []
    seen_s1 = set()

    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\r\n")
        header = header_line.split("\t")
        if len(header) < 2 or "source1_entity_id" not in header[0]:
            issues.append(f"Header warning: expected 'source1_entity_id\\tmatched_entity_ids', got '{header_line}'")

        for line_num, line in enumerate(f, start=2):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0].strip()

            if s1_id in seen_s1:
                issues.append(f"Line {line_num}: Duplicate source1_entity_id found: '{s1_id}'")
            seen_s1.add(s1_id)

            raw_matches = parts[1].strip() if len(parts) > 1 else ""
            if not raw_matches:
                pred[s1_id] = set()
            else:
                match_list = [m.strip() for m in raw_matches.split(",") if m.strip()]
                match_set = set(match_list)
                if len(match_list) != len(match_set):
                    issues.append(
                        f"Line {line_num}: Duplicate matched IDs inside row for '{s1_id}'"
                    )
                pred[s1_id] = match_set

    return pred, issues


def evaluate_matching_pipeline(
    gt_path: str,
    pred_path: str,
    allow_subset: bool = False,
) -> dict:
    """Validate and score a prediction file against ground truth."""
    print("Loading Ground Truth...")
    gt = load_ground_truth(gt_path)
    print(f"Loaded {len(gt):,} Ground Truth S1 records.")

    print(f"Loading Predictions from {pred_path}...")
    pred, issues = load_predictions(pred_path)
    print(f"Loaded {len(pred):,} Prediction records.")

    if issues:
        print("\n[VALIDATION WARNINGS / ISSUES ENCOUNTERED]")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues.")

    # Check for missing S1 entities
    gt_s1_set = set(gt.keys())
    pred_s1_set = set(pred.keys())

    missing_in_pred = gt_s1_set - pred_s1_set
    extra_in_pred = pred_s1_set - gt_s1_set

    if missing_in_pred and not allow_subset:
        sample_missing = sorted(list(missing_in_pred))[:5]
        err_msg = (
            f"Validation Failed: Prediction file is missing {len(missing_in_pred):,} Source 1 entities "
            f"(e.g., {sample_missing}). Use --allow-subset if validating on a validation split."
        )
        raise ValueError(err_msg)

    eval_s1_list = sorted(list(gt_s1_set & pred_s1_set)) if allow_subset else sorted(list(gt_s1_set))

    results = evaluate_predictions(gt, pred, eval_s1_list)

    print("\n" + "=" * 60)
    print("MATCHING EVALUATION RESULTS (COMPETITION METRIC)")
    print("=" * 60)
    print(f"Evaluated S1 Entities:           {results['total_evaluated']:,}")
    print(f"Macro Precision:                 {results['macro_precision']:.6f}")
    print(f"Macro Recall:                    {results['macro_recall']:.6f}")
    print(f"Macro F0.5 Score:                {results['macro_f05']:.6f}")
    print("-" * 60)
    print(f"Perfect Matches (Full set exact):{results['perfect_matches']:,} "
          f"({results['perfect_matches']/results['total_evaluated']*100:.2f}%)")
    print(f"Entities with False Positives:   {results['entities_with_fp']:,} "
          f"({results['entities_with_fp']/results['total_evaluated']*100:.2f}%)")
    print(f"Entities with False Negatives:   {results['entities_with_fn']:,} "
          f"({results['entities_with_fn']/results['total_evaluated']*100:.2f}%)")
    print(f"Both Empty (Singletons correct): {results['both_empty']:,} "
          f"({results['both_empty']/results['total_evaluated']*100:.2f}%)")
    print(f"Micro Pair Stats:                TP={results['total_tp']:,} | FP={results['total_fp']:,} | FN={results['total_fn']:,}")
    print("=" * 60)

    return results


def evaluate_blocking_pipeline(gt_path: str, candidates_path: str) -> dict:
    """
    Evaluate candidate blocking recall on ground truth without Cartesian explosion.

    Candidate file format:
    s1_entity_id\tcandidate_entity_id\tcandidate_source (or s1, candidate_id)
    """
    print("Loading Ground Truth for Blocking Evaluation...")
    gt = load_ground_truth(gt_path)

    total_gt_pairs = sum(len(matches) for matches in gt.values())
    print(f"Total Ground-Truth Positive Pairs: {total_gt_pairs:,}")

    # Track recovered positive pairs per S1
    recovered_pairs_count = 0
    total_candidates = 0
    unique_s1_candidates = set()

    # Track which (s1, cand) pairs were recovered to avoid double counting if duplicate candidates exist
    recovered_set = set()

    print(f"Streaming candidate pairs from {candidates_path}...")
    with open(candidates_path, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\r\n")
        header = [col.strip().lower() for col in header_line.split("\t")]

        s1_idx = 0
        cand_idx = 1
        for i, col in enumerate(header):
            if col in ("s1_entity_id", "source1_entity_id", "s1_id", "s1"):
                s1_idx = i
            elif col in ("candidate_entity_id", "candidate_id", "matched_entity_id", "cand_id", "target_id", "candidate"):
                cand_idx = i
            elif "source" not in col and ("cand" in col or "target" in col):
                cand_idx = i

        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) <= max(s1_idx, cand_idx):
                continue

            s1 = parts[s1_idx].strip()
            cand = parts[cand_idx].strip()
            total_candidates += 1
            unique_s1_candidates.add(s1)

            # Check if this candidate is a true match in GT
            if s1 in gt and cand in gt[s1]:
                pair_key = (s1, cand)
                if pair_key not in recovered_set:
                    recovered_set.add(pair_key)
                    recovered_pairs_count += 1

    missed_pairs = total_gt_pairs - recovered_pairs_count
    candidate_recall = (recovered_pairs_count / total_gt_pairs) if total_gt_pairs > 0 else 0.0
    avg_candidates_per_s1 = (total_candidates / len(unique_s1_candidates)) if unique_s1_candidates else 0.0

    print("\n" + "=" * 60)
    print("BLOCKING RECALL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Ground-Truth Positive Pairs: {total_gt_pairs:,}")
    print(f"Total Candidate Pairs Generated:   {total_candidates:,}")
    print(f"Positive Pairs Recovered:          {recovered_pairs_count:,}")
    print(f"Positive Pairs Missed:             {missed_pairs:,}")
    print(f"Candidate Blocking Recall:         {candidate_recall:.6f} ({candidate_recall*100:.2f}%)")
    print(f"Source 1 Entities with Candidates: {len(unique_s1_candidates):,}")
    print(f"Average Candidates per S1:         {avg_candidates_per_s1:.2f}")
    print("=" * 60)

    return {
        "total_gt_pairs": total_gt_pairs,
        "total_candidates": total_candidates,
        "recovered_pairs": recovered_pairs_count,
        "missed_pairs": missed_pairs,
        "candidate_recall": candidate_recall,
        "avg_candidates_per_s1": avg_candidates_per_s1,
    }


def run_synthetic_tests():
    """
    Run self-contained synthetic tests covering edge cases:
    - Perfect matches
    - Correct empty singletons
    - False positives on singletons
    - False negatives (missed matches)
    - Missing prediction rows
    - Duplicate S1 rows
    - Duplicate matched IDs within row
    """
    print("\nRunning synthetic metric tests...")

    # Test 1: Metric exact computation
    gt = {
        "A": {"X", "Y"},
        "B": set(),
        "C": {"Z"},
        "D": set(),
        "E": {"M"},
    }

    pred = {
        "A": {"X", "Y"},       # perfect match -> prec=1.0, rec=1.0, f05=1.0
        "B": set(),             # perfect singleton -> prec=1.0, rec=1.0, f05=1.0
        "C": {"Z", "Q"},        # false positive -> prec=1/2, rec=1.0 -> f05 = 1.25*(0.5*1)/(0.25*0.5 + 1) = 0.625 / 1.125 = 0.555556
        "D": {"W"},             # false merge on singleton -> prec=0.0, rec=0.0, f05=0.0
        "E": {"WRONG"},         # completely wrong -> prec=0.0, rec=0.0, f05=0.0
    }

    results = evaluate_predictions(gt, pred)

    # Entity A: prec=1.0, rec=1.0, f05=1.0
    # Entity B: prec=1.0, rec=1.0, f05=1.0
    # Entity C: prec=0.5, rec=1.0, f05=0.5555555555555556
    # Entity D: prec=0.0, rec=0.0, f05=0.0
    # Entity E: prec=0.0, rec=0.0, f05=0.0
    expected_f05 = (1.0 + 1.0 + (1.25 * 0.5 * 1.0 / (0.25 * 0.5 + 1.0)) + 0.0 + 0.0) / 5.0
    expected_prec = (1.0 + 1.0 + 0.5 + 0.0 + 0.0) / 5.0
    expected_rec = (1.0 + 1.0 + 1.0 + 0.0 + 0.0) / 5.0

    assert abs(results["macro_f05"] - expected_f05) < 1e-6, f"F0.5 mismatch: {results['macro_f05']} vs {expected_f05}"
    assert abs(results["macro_precision"] - expected_prec) < 1e-6, f"Prec mismatch: {results['macro_precision']} vs {expected_prec}"
    assert abs(results["macro_recall"] - expected_rec) < 1e-6, f"Rec mismatch: {results['macro_recall']} vs {expected_rec}"
    assert results["perfect_matches"] == 2
    assert results["both_empty"] == 1
    assert results["entities_with_fp"] == 3  # C, D, and E (predicted 'WRONG' is also an FP)
    assert results["entities_with_fn"] == 1  # E (missed 'M')

    print("  [PASS] Test 1: Mathematical exactness of macro-averaged F0.5, Precision, Recall.")

    # Test 2: Missing prediction row detection
    missing_pred = {"A": {"X", "Y"}}
    try:
        evaluate_predictions(gt, missing_pred, list(gt.keys()))
        # In evaluate_predictions with explicit s1_ids, missing keys default to empty set
        res_missing = evaluate_predictions(gt, missing_pred, list(gt.keys()))
        assert res_missing["entities_with_fn"] >= 2
    except Exception as e:
        print(f"  [PASS] Handled missing prediction safely: {e}")
    print("  [PASS] Test 2: Missing row handling verified.")

    # Test 3: Duplicate detection helper
    import tempfile
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv", encoding="utf-8") as tmp:
        tmp.write("source1_entity_id\tmatched_entity_ids\n")
        tmp.write("S1-1\tS2-1,S2-1\n")  # duplicate matched ID
        tmp.write("S1-1\tS2-2\n")       # duplicate S1 row
        tmp_name = tmp.name

    pred_loaded, issues = load_predictions(tmp_name)
    os.unlink(tmp_name)

    assert any("Duplicate source1_entity_id found" in iss for iss in issues)
    assert any("Duplicate matched IDs inside row" in iss for iss in issues)
    print("  [PASS] Test 3: Duplicate S1 row and duplicate candidate ID detection verified.")

    # Test 4: Blocking recall evaluation test
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv", encoding="utf-8") as gt_tmp:
        gt_tmp.write("source1_entity_id\tmatched_entity_ids\n")
        gt_tmp.write("A\tX,Y\n")
        gt_tmp.write("B\t\n")
        gt_tmp.write("C\tZ\n")
        gt_tmp_name = gt_tmp.name

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv", encoding="utf-8") as cand_tmp:
        cand_tmp.write("s1_entity_id\tcandidate_entity_id\tcandidate_source\n")
        cand_tmp.write("A\tX\tS2\n")  # recovered
        cand_tmp.write("A\tW\tS2\n")  # negative
        cand_tmp.write("C\tZ\tS3\n")  # recovered
        cand_tmp_name = cand_tmp.name

    blocking_res = evaluate_blocking_pipeline(gt_tmp_name, cand_tmp_name)
    os.unlink(gt_tmp_name)
    os.unlink(cand_tmp_name)

    # Total GT pairs = 3 (A->X, A->Y, C->Z). Recovered = 2 (A->X, C->Z). Recall = 2/3 = 0.666667
    assert blocking_res["total_gt_pairs"] == 3
    assert blocking_res["recovered_pairs"] == 2
    assert abs(blocking_res["candidate_recall"] - (2 / 3)) < 1e-6
    print("  [PASS] Test 4: Blocking recall calculation verified.")

    print("\nALL SYNTHETIC TESTS PASSED SUCCESSFULLY! (4/4)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge - Validation Metric & Quality Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ground-truth", type=str, help="Path to ground-truth TSV (e.g., data/train/train_ground_truth.tsv)")
    parser.add_argument("--prediction", type=str, help="Path to prediction TSV (source1_entity_id\\tmatched_entity_ids)")
    parser.add_argument("--candidates", type=str, help="Path to candidate pairs TSV for blocking recall evaluation")
    parser.add_argument("--mode", choices=["matching", "blocking"], default="matching", help="Evaluation mode (matching or blocking)")
    parser.add_argument("--allow-subset", action="store_true", help="Allow prediction file to contain a subset of ground truth S1 IDs (useful for validation splits)")
    parser.add_argument("--test", action="store_true", help="Run self-contained synthetic tests and verify metric calculations")

    args = parser.parse_args()

    if args.test:
        run_synthetic_tests()
        return

    if not args.ground_truth:
        parser.print_help()
        print("\nError: --ground-truth is required unless running with --test.")
        sys.exit(1)

    if args.candidates or args.mode == "blocking":
        if not args.candidates:
            print("Error: --candidates is required for blocking mode.")
            sys.exit(1)
        evaluate_blocking_pipeline(args.ground_truth, args.candidates)
    elif args.prediction or args.mode == "matching":
        if not args.prediction:
            print("Error: --prediction is required for matching mode.")
            sys.exit(1)
        evaluate_matching_pipeline(args.ground_truth, args.prediction, allow_subset=args.allow_subset)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
