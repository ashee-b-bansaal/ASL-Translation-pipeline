import os
import re
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Set


EXPRESSIONS_BY_CATEGORY = {
    "emotion": {"happy", "sad", "angry"},
    "mouth_morpheme": {"mm", "cs", "th", "cha"},
    "grammar": {"raise", "shake", "furrow"},
}

EXPR_TO_CATEGORY: Dict[str, str] = {
    expr: cat for cat, exprs in EXPRESSIONS_BY_CATEGORY.items() for expr in exprs
}


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_SIGNS_PATH = os.path.join(
    SCRIPT_DIR,
    "..",
    "final_pipeline_txt_220110",
    "analysis",
    "ground_truth_signs.txt",
)


def load_sign_categories(path: str) -> Dict[str, str]:
    categories: Dict[str, str] = {}
    if not os.path.exists(path):
        return categories

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            word, rest = line.split(":", 1)
            word = word.strip().upper()
            category = rest.strip()
            # Handle stray leading ":" (e.g., "IF: :upper")
            category = category.lstrip(":").strip().lower()
            if not category:
                category = "none"
            categories[word] = category

    return categories


SIGN_CATEGORY_MAP = load_sign_categories(os.path.abspath(GROUND_TRUTH_SIGNS_PATH))
SIGN_CATEGORY_ORDER = ["upper", "lower", "none"]


def extract_gloss_expressions(gloss_text: str) -> Set[str]:
    """
    Extract expressions from ASL gloss text.
    Expressions appear in parentheses like (raise), (mm), (raise,mm), etc.
    
    Returns set of expression strings found in the gloss.
    """
    expressions = set()
    # Pattern to match expressions in parentheses: (expr) or (expr1,expr2)
    pattern = r'\(([^)]+)\)'
    matches = re.findall(pattern, gloss_text)
    for match in matches:
        # Split by comma to handle cases like (raise,mm)
        for expr in match.split(','):
            expr = expr.strip()
            if expr in EXPR_TO_CATEGORY:
                expressions.add(expr)
    return expressions


def has_overlap_in_gloss(gloss_text: str) -> bool:
    """
    Check if the ASL gloss has overlap - meaning at least one sign word has 
    multiple expressions assigned to it (e.g., WORD(raise,mm) or WORD(raise)(mm)).
    
    Returns True if overlap exists, False otherwise.
    """
    # Pattern to match expressions in parentheses: (expr) or (expr1,expr2)
    pattern = r'\(([^)]+)\)'
    matches = re.findall(pattern, gloss_text)
    
    for match in matches:
        # Check if this parentheses contains multiple expressions (comma-separated)
        exprs = [e.strip() for e in match.split(',')]
        # Filter to only known expressions
        known_exprs = [e for e in exprs if e in EXPR_TO_CATEGORY]
        if len(known_exprs) >= 2:
            return True
    
    # Also check for consecutive parentheses on the same word
    # Pattern to match word followed by multiple parentheses: WORD(expr1)(expr2)
    # This matches a word, then (expr), then another (expr) without space or word in between
    consecutive_pattern = r'\w+\([^)]+\)\s*\([^)]+\)'
    if re.search(consecutive_pattern, gloss_text):
        return True
    
    return False


def parse_gloss_assignments(gloss_text: str) -> List[Tuple[str, str, Optional[int]]]:
    """
    Parse the ASL gloss text and return a list of (sign_word, expression) tuples.
    Handles cases like WORD(expr1,expr2) and WORD(expr1)(expr2).
    """
    assignments: List[Tuple[str, str, Optional[int]]] = []
    # Match a word followed by one or more parenthesis groups
    pattern = re.compile(r"([A-Za-z0-9\-']+)((?:\([^)]*\))+)")
    group_counter = 0
    for match in pattern.finditer(gloss_text):
        word = match.group(1)
        paren_block = match.group(2)
        inner_exprs = re.findall(r"\(([^)]*)\)", paren_block)
        group_id = None
        if len(inner_exprs) > 1 or any("," in expr for expr in inner_exprs):
            group_id = group_counter
            group_counter += 1
        for expr_group in inner_exprs:
            for expr in expr_group.split(","):
                cleaned = expr.strip()
                if cleaned:
                    assignments.append((word, cleaned, group_id))
    return assignments


def map_gt_to_sign_categories(
    gt: List[str], gloss_assignments: List[Tuple[str, str, Optional[int]]]
) -> List[str]:
    """
    Map each ground-truth expression to the sign category of the word
    it is attached to in the ASL gloss.
    """
    categories: List[str] = []
    if not gt:
        return categories

    filtered_assignments: List[Tuple[str, str, Optional[int]]] = [
        (word, expr, group_id)
        for word, expr, group_id in gloss_assignments
        if expr in EXPR_TO_CATEGORY
    ]

    if not filtered_assignments:
        return ["none"] * len(gt)

    categories = ["none"] * len(gt)
    used = [False] * len(filtered_assignments)

    group_to_indices: Dict[Optional[int], List[int]] = defaultdict(list)
    for idx, (_, _, group_id) in enumerate(filtered_assignments):
        if group_id is not None:
            group_to_indices[group_id].append(idx)

    i = 0
    while i < len(gt):
        expr = gt[i]
        matched = False

        # Try to match multi-expression groups first
        for idx, (word, assigned_expr, group_id) in enumerate(filtered_assignments):
            if used[idx] or assigned_expr != expr or group_id is None:
                continue

            group_indices = [j for j in group_to_indices[group_id] if not used[j]]
            group_exprs = [filtered_assignments[j][1] for j in group_indices]

            if len(group_exprs) <= 1:
                continue

            if i + len(group_exprs) > len(gt):
                continue

            if gt[i : i + len(group_exprs)] == group_exprs:
                for offset, assignment_idx in enumerate(group_indices):
                    word_match, _, _ = filtered_assignments[assignment_idx]
                    cat = SIGN_CATEGORY_MAP.get(word_match.upper(), "none")
                    categories[i + offset] = cat if cat in SIGN_CATEGORY_ORDER else "none"
                    used[assignment_idx] = True
                i += len(group_exprs)
                matched = True
                break

        if matched:
            continue

        # Fallback to single assignment match
        for idx, (word, assigned_expr, _) in enumerate(filtered_assignments):
            if used[idx] or assigned_expr != expr:
                continue
            cat = SIGN_CATEGORY_MAP.get(word.upper(), "none")
            categories[i] = cat if cat in SIGN_CATEGORY_ORDER else "none"
            used[idx] = True
            matched = True
            break

        if not matched:
            categories[i] = "none"

        i += 1

    return categories


def parse_line(line: str):
    """
    Parse a single line of exp_best_outputs.txt.

    Expected format:
        ground_truth_exprs - recognized_exprs <<<<< ... >>>> ASL_GLOSS ---- ...

    Returns:
        (gt_list, rec_list, gloss_exprs_set, gloss_text, gloss_assignments)
        filtered to known expressions, or None if parse fails.
        gloss_exprs_set is None if no gloss section found.
        gloss_text is None if no gloss section found.
        gloss_assignments is the list of (sign_word, expression) tuples from the gloss.
    """
    line = line.strip()
    if not line:
        return None

    # Split ground truth and the rest
    if " - " not in line:
        return None

    gt_part, _, right = line.partition(" - ")

    if "<<<<<" in right:
        rec_part, _, after_rec = right.partition("<<<<<")
    else:
        rec_part = right
        after_rec = ""

    gt_tokens = [t.strip() for t in gt_part.split(",") if t.strip()]
    rec_tokens = [t.strip() for t in rec_part.split(",") if t.strip()]

    # Keep only expressions we care about
    gt = [t for t in gt_tokens if t in EXPR_TO_CATEGORY]
    rec = [t for t in rec_tokens if t in EXPR_TO_CATEGORY]

    # Extract ASL gloss section (between >>>> and ----)
    gloss_exprs = None
    gloss_text = None
    gloss_assignments: List[Tuple[str, str]] = []
    if ">>>>" in after_rec and "----" in after_rec:
        gloss_start = after_rec.find(">>>>")
        gloss_end = after_rec.find("----")
        if gloss_start < gloss_end:
            gloss_text = after_rec[gloss_start + 5:gloss_end].strip()  # +5 to skip ">>>> "
            gloss_exprs = extract_gloss_expressions(gloss_text)
            gloss_assignments = parse_gloss_assignments(gloss_text)

    return gt, rec, gloss_exprs, gloss_text, gloss_assignments


def align_sequences(
    gt: List[str], rec: List[str]
) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Align two expression sequences using edit distance with operations:
      - match (cost 0)
      - substitute (cost 1)
      - delete from gt (missing, cost 1)
      - insert into rec (extra, cost 1)

    Returns list of (gt_expr_or_None, rec_expr_or_None) pairs.
    """
    n, m = len(gt), len(rec)

    # dp[i][j] = minimal cost to align gt[:i] with rec[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = ("del", i - 1, None)  # delete gt[i-1]

    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = ("ins", None, j - 1)  # insert rec[j-1]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt[i - 1] == rec[j - 1]:
                # Exact match, always prefer this
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = ("match", i - 1, j - 1)
            else:
                # Costs for substitute, delete, insert
                subst_cost = dp[i - 1][j - 1] + 1
                del_cost = dp[i - 1][j] + 1
                ins_cost = dp[i][j - 1] + 1

                best_cost = subst_cost
                best_op = ("subst", i - 1, j - 1)

                # Tie-breaking order: delete > substitute > insert
                if del_cost < best_cost or (
                    del_cost == best_cost and best_op[0] != "del"
                ):
                    best_cost = del_cost
                    best_op = ("del", i - 1, None)

                if ins_cost < best_cost or (
                    ins_cost == best_cost and best_op[0] not in ("del", "ins")
                ):
                    best_cost = ins_cost
                    best_op = ("ins", None, j - 1)

                dp[i][j] = best_cost
                back[i][j] = best_op

    # Backtrace
    aligned: List[Tuple[Optional[str], Optional[str]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op, gi, gj = back[i][j]
        if op == "match":
            aligned.append((gt[gi], rec[gj]))
            i -= 1
            j -= 1
        elif op == "subst":
            aligned.append((gt[gi], rec[gj]))
            i -= 1
            j -= 1
        elif op == "del":
            aligned.append((gt[gi], None))
            i -= 1
        elif op == "ins":
            aligned.append((None, rec[gj]))
            j -= 1
        else:
            raise RuntimeError(f"Unknown op in backtrace: {op}")

    aligned.reverse()
    return aligned


def analyze_file(input_path: str, output_path: str) -> None:
    # Per-category stats
    category_total = {cat: 0 for cat in EXPRESSIONS_BY_CATEGORY}
    category_misclassified = {cat: 0 for cat in EXPRESSIONS_BY_CATEGORY}
    category_missing = {cat: 0 for cat in EXPRESSIONS_BY_CATEGORY}

    # Per-expression stats
    expr_total = {expr: 0 for expr in EXPR_TO_CATEGORY}
    expr_misclassified = {expr: 0 for expr in EXPR_TO_CATEGORY}
    expr_missing = {expr: 0 for expr in EXPR_TO_CATEGORY}
    expr_mis_as: Dict[str, Dict[str, int]] = {
        expr: defaultdict(int) for expr in EXPR_TO_CATEGORY
    }

    # Extra recognized (no corresponding GT): None -> expr
    none_to_expr: Dict[str, int] = defaultdict(int)

    # Analysis by number of expressions in ground truth
    # Key: number of expressions, Value: dict with stats
    def make_count_stats():
        return {
            "total_cases": 0,
            "total_exprs": 0,  # Total expressions across all cases
            "misclassified": 0,
            "missing": 0,
            "extra": 0,
            "missing_by_count": defaultdict(int),  # Key: how many missing, Value: count of cases
            "misclassified_by_count": defaultdict(int),  # Key: how many misclassified, Value: count of cases
            "misclassified_details_by_count": defaultdict(int)  # Key: (mis_count, gt_expr, rec_expr), Value: count
        }
    by_count_stats: Dict[int, Dict] = defaultdict(make_count_stats)

    # Stats by sign category (lower/upper/none)
    sign_category_stats: Dict[str, Dict[str, int]] = {
        cat: {"total": 0, "misclassified": 0, "missing": 0}
        for cat in SIGN_CATEGORY_ORDER
    }

    # Per-expression stats broken down by sign category
    expr_sign_stats: Dict[str, Dict[str, Dict[str, object]]] = {
        expr: {
            cat: {
                "total": 0,
                "misclassified": 0,
                "missing": 0,
                "misclassified_as": defaultdict(int),
            }
            for cat in SIGN_CATEGORY_ORDER
        }
        for expr in EXPR_TO_CATEGORY
    }

    # Per-expression-category stats broken down by sign category
    expr_category_sign_stats: Dict[str, Dict[str, Dict[str, int]]] = {
        expr_cat: {
            cat: {"total": 0, "misclassified": 0, "missing": 0}
            for cat in SIGN_CATEGORY_ORDER
        }
        for expr_cat in EXPRESSIONS_BY_CATEGORY
    }

    # Overlap/Non-overlap stats (counted at sentence level)
    overlap_stats = {
        "total_exprs": 0,
        "misclassified_sentences": 0,  # Number of sentences with at least one misclassification
        "missing_sentences": 0,  # Number of sentences with at least one missing expression
        "occurrences": 0  # Number of sentences with overlap
    }
    non_overlap_stats = {
        "total_exprs": 0,
        "misclassified_sentences": 0,  # Number of sentences with at least one misclassification
        "missing_sentences": 0,  # Number of sentences with at least one missing expression
        "occurrences": 0  # Number of sentences without overlap
    }

    total_sentences = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            parsed = parse_line(raw_line)
            if parsed is None:
                continue

            gt, rec, gloss_exprs, gloss_text, gloss_assignments = parsed
            if not gt and not rec:
                continue

            total_sentences += 1

            if gt:
                if gloss_assignments:
                    gt_sign_categories = map_gt_to_sign_categories(gt, gloss_assignments)
                else:
                    gt_sign_categories = ["none"] * len(gt)
            else:
                gt_sign_categories = []

            # Count ground-truth occurrences
            for idx, g in enumerate(gt):
                cat = EXPR_TO_CATEGORY[g]
                category_total[cat] += 1
                expr_total[g] += 1

                sign_cat = (
                    gt_sign_categories[idx]
                    if idx < len(gt_sign_categories)
                    else "none"
                )
                if sign_cat not in sign_category_stats:
                    sign_category_stats[sign_cat] = {
                        "total": 0,
                        "misclassified": 0,
                        "missing": 0,
                    }
                sign_category_stats[sign_cat]["total"] += 1

                if sign_cat not in expr_sign_stats[g]:
                    expr_sign_stats[g][sign_cat] = {
                        "total": 0,
                        "misclassified": 0,
                        "missing": 0,
                        "misclassified_as": defaultdict(int),
                    }
                expr_sign_stats[g][sign_cat]["total"] += 1

                expr_cat = EXPR_TO_CATEGORY[g]
                if sign_cat not in expr_category_sign_stats[expr_cat]:
                    expr_category_sign_stats[expr_cat][sign_cat] = {
                        "total": 0,
                        "misclassified": 0,
                        "missing": 0,
                    }
                expr_category_sign_stats[expr_cat][sign_cat]["total"] += 1

            # Analysis by number of expressions in ground truth
            gt_count = len(gt)
            if gt_count > 0:
                by_count_stats[gt_count]["total_cases"] += 1
                by_count_stats[gt_count]["total_exprs"] += gt_count

            # Determine overlap at sentence level (if gloss has multiple expressions on one sign word)
            has_overlap = False
            if gloss_text is not None:
                # Check if the gloss has overlap (multiple expressions on one sign word)
                has_overlap = has_overlap_in_gloss(gloss_text)
                
                if has_overlap:
                    overlap_stats["occurrences"] += 1  # Count as 1 occurrence (one sentence)
                    # Count all expressions in this sentence for overlap stats
                    for g in gt:
                        overlap_stats["total_exprs"] += 1
                else:
                    non_overlap_stats["occurrences"] += 1  # Count as 1 occurrence (one sentence)
                    # Count all expressions in this sentence for non-overlap stats
                    for g in gt:
                        non_overlap_stats["total_exprs"] += 1

            # Align and accumulate errors
            aligned = align_sequences(gt, rec)
            
            # Track missing count for this sentence (for by_count_stats)
            missing_count_in_sentence = 0
            extra_count_in_sentence = 0
            misclassified_count_in_sentence = 0
            misclassifications_in_sentence = []  # List of (gt_expr, rec_expr) tuples
            has_misclassification_in_sentence = False
            has_missing_in_sentence = False
            
            gt_category_idx = 0

            for g, r in aligned:
                current_sign_category = None
                if g is not None:
                    if gt_category_idx < len(gt_sign_categories):
                        current_sign_category = gt_sign_categories[gt_category_idx]
                    else:
                        current_sign_category = "none"
                    gt_category_idx += 1
                if current_sign_category and current_sign_category not in sign_category_stats:
                    sign_category_stats[current_sign_category] = {
                        "total": 0,
                        "misclassified": 0,
                        "missing": 0,
                    }
                if g is not None and current_sign_category:
                    if current_sign_category not in expr_sign_stats[g]:
                        expr_sign_stats[g][current_sign_category] = {
                            "total": 0,
                            "misclassified": 0,
                            "missing": 0,
                            "misclassified_as": defaultdict(int),
                        }
                    expr_cat = EXPR_TO_CATEGORY[g]
                    if current_sign_category not in expr_category_sign_stats[expr_cat]:
                        expr_category_sign_stats[expr_cat][current_sign_category] = {
                            "total": 0,
                            "misclassified": 0,
                            "missing": 0,
                        }
                if g is not None and r is not None:
                    if g == r:
                        # Correct
                        continue
                    # Misclassified: g -> r
                    expr_misclassified[g] += 1
                    expr_mis_as[g][r] += 1
                    category_misclassified[EXPR_TO_CATEGORY[g]] += 1
                    misclassified_count_in_sentence += 1
                    misclassifications_in_sentence.append((g, r))
                    has_misclassification_in_sentence = True

                    if current_sign_category:
                        sign_category_stats[current_sign_category]["misclassified"] += 1
                        expr_sign_stats[g][current_sign_category]["misclassified"] += 1
                        expr_sign_stats[g][current_sign_category]["misclassified_as"][r] += 1
                        expr_category = EXPR_TO_CATEGORY[g]
                        expr_category_sign_stats[expr_category][current_sign_category][
                            "misclassified"
                        ] += 1
                    
                    # Track for by_count_stats
                    if gt_count > 0:
                        by_count_stats[gt_count]["misclassified"] += 1
                            
                elif g is not None and r is None:
                    # Missing ground-truth expression
                    expr_missing[g] += 1
                    category_missing[EXPR_TO_CATEGORY[g]] += 1
                    missing_count_in_sentence += 1
                    has_missing_in_sentence = True

                    if current_sign_category:
                        sign_category_stats[current_sign_category]["missing"] += 1
                        expr_sign_stats[g][current_sign_category]["missing"] += 1
                        expr_category = EXPR_TO_CATEGORY[g]
                        expr_category_sign_stats[expr_category][current_sign_category][
                            "missing"
                        ] += 1
                    
                    # Track for by_count_stats
                    if gt_count > 0:
                        by_count_stats[gt_count]["missing"] += 1
                            
                elif g is None and r is not None:
                    # Extra recognized expression
                    none_to_expr[r] += 1
                    extra_count_in_sentence += 1
                    
                    # Track for by_count_stats
                    if gt_count > 0:
                        by_count_stats[gt_count]["extra"] += 1
                # (g is None and r is None) does not occur
            
            # Track overlap/non-overlap at sentence level (count sentence once if it has misclassification or missing)
            if gloss_text is not None:
                if has_overlap:
                    if has_misclassification_in_sentence:
                        overlap_stats["misclassified_sentences"] += 1
                    if has_missing_in_sentence:
                        overlap_stats["missing_sentences"] += 1
                else:
                    if has_misclassification_in_sentence:
                        non_overlap_stats["misclassified_sentences"] += 1
                    if has_missing_in_sentence:
                        non_overlap_stats["missing_sentences"] += 1
            
            # Record missing count for this sentence (for breakdown by number of missing)
            if gt_count > 0 and missing_count_in_sentence > 0:
                by_count_stats[gt_count]["missing_by_count"][missing_count_in_sentence] += 1
            
            # Record misclassification count for this sentence (for breakdown by number of misclassified)
            if gt_count > 0 and misclassified_count_in_sentence > 0:
                by_count_stats[gt_count]["misclassified_by_count"][misclassified_count_in_sentence] += 1
                # Track misclassification details for this specific misclassification count
                for gt_expr, rec_expr in misclassifications_in_sentence:
                    by_count_stats[gt_count]["misclassified_details_by_count"][(misclassified_count_in_sentence, gt_expr, rec_expr)] += 1

    # Write summary
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"Total number of sentences = {total_sentences}\n\n")

        out.write("Per category:\n\n")
        for cat in ["emotion", "mouth_morpheme", "grammar"]:
            total = category_total[cat]
            mis = category_misclassified[cat]
            missing = category_missing[cat]
            correct = max(total - mis - missing, 0)

            mis_pct = (mis / total * 100.0) if total > 0 else 0.0
            missing_pct = (missing / total * 100.0) if total > 0 else 0.0
            accuracy_pct = (correct / total * 100.0) if total > 0 else 0.0

            out.write(
                f"{cat.capitalize()}: misclassified = {mis} / {total} = {mis_pct:.2f}%\n"
            )
            out.write(
                f"{cat.capitalize()}: missing = {missing} / {total} = {missing_pct:.2f}%\n"
            )
            out.write(
                f"{cat.capitalize()}: accuracy = {correct} / {total} = {accuracy_pct:.2f}%\n\n"
            )

        out.write("Per expression:\n\n")

        def safe_pct(num: int, den: int) -> float:
            return (num / den * 100.0) if den > 0 else 0.0

        # Order expressions by category, then name
        for cat in ["emotion", "mouth_morpheme", "grammar"]:
            for expr in sorted(EXPRESSIONS_BY_CATEGORY[cat]):
                total = expr_total[expr]
                mis = expr_misclassified[expr]
                missing = expr_missing[expr]

                mis_pct = safe_pct(mis, total)
                missing_pct = safe_pct(missing, total)

                out.write(
                    f"{expr}: misclassified = {mis} / {total} = {mis_pct:.2f}%\n"
                )

                # Top-5 misclassifications for this expression
                mis_as_dict = expr_mis_as[expr]
                if mis_as_dict:
                    sorted_mis_as = sorted(
                        mis_as_dict.items(), key=lambda kv: kv[1], reverse=True
                    )[:5]
                    out.write("    top misclassifications:\n")
                    for target_expr, count in sorted_mis_as:
                        share_pct = safe_pct(count, mis)
                        out.write(
                            f"        as {target_expr}: {count} / {mis} = {share_pct:.2f}%\n"
                        )
                else:
                    out.write("    top misclassifications: none\n")

                out.write(
                    f"{expr}: missing = {missing} / {total} = {missing_pct:.2f}%\n\n"
                )

        # Extra recognized expressions (no GT counterpart)
        if none_to_expr:
            out.write("Extra recognized expressions (no ground truth, i.e., None -> expr):\n")
            for expr, count in sorted(none_to_expr.items(), key=lambda kv: kv[1], reverse=True):
                out.write(f"    None misclassified as {expr}: {count}\n")
        
        out.write("\n" + "="*80 + "\n\n")
        
        # Analysis by number of expressions in ground truth
        out.write("Accuracy of expressions by NO. of facial expressions (in grnd truth):\n\n")
        
        # Sort by count
        for count in sorted(by_count_stats.keys()):
            stats = by_count_stats[count]
            total_cases = stats["total_cases"]
            total_exprs = stats["total_exprs"]
            mis = stats["misclassified"]
            missing = stats["missing"]
            extra = stats["extra"]
            
            if total_cases == 0:
                continue
            
            # Percentages should be based on total cases, not total expressions
            mis_pct = safe_pct(mis, total_cases)
            missing_pct = safe_pct(missing, total_cases)
            extra_pct = safe_pct(extra, total_cases)
            accuracy = 100.0 - mis_pct - missing_pct - extra_pct
            
            out.write(f"{count} expression(s):\n")
            out.write(f"  Total cases = {total_cases}\n")
            out.write(f"  Total expressions = {total_exprs}\n")
            out.write(f"  Misclassified: {mis} / {total_cases} = {mis_pct:.2f}%\n")
            
            # Show misclassification breakdown (1, 2, 3, etc.)
            if stats["misclassified_by_count"]:
                for mis_num in sorted(stats["misclassified_by_count"].keys(), reverse=True):
                    cases_with_mis = stats["misclassified_by_count"][mis_num]
                    pct = safe_pct(cases_with_mis, total_cases)
                    out.write(f"    Misclassification {mis_num}: {cases_with_mis} / {total_cases} = {pct:.2f}%\n")
                    
                    # Show top misclassifications for this category
                    # Filter misclassification details to only those cases with mis_num misclassifications
                    mis_details_for_count = {
                        (gt_expr, rec_expr): count
                        for (misc_count, gt_expr, rec_expr), count in stats["misclassified_details_by_count"].items()
                        if misc_count == mis_num
                    }
                    if mis_details_for_count:
                        # Get top misclassifications
                        sorted_mis = sorted(mis_details_for_count.items(), key=lambda kv: kv[1], reverse=True)[:5]
                        out.write(f"      Top misclassifications:\n")
                        for (gt_expr, rec_expr), mis_count in sorted_mis:
                            mis_pct_detail = safe_pct(mis_count, cases_with_mis) if cases_with_mis > 0 else 0.0
                            out.write(f"        {gt_expr} -> {rec_expr}: {mis_count} / {cases_with_mis} = {mis_pct_detail:.2f}%\n")
            
            out.write(f"  Missing: {missing} / {total_cases} = {missing_pct:.2f}%\n")
            
            # Show missing breakdown for 2+ expressions
            if count >= 2 and stats["missing_by_count"]:
                for missing_num in sorted(stats["missing_by_count"].keys(), reverse=True):
                    cases_with_missing = stats["missing_by_count"][missing_num]
                    pct = safe_pct(cases_with_missing, total_cases)
                    out.write(f"    {missing_num} Missing: {cases_with_missing} / {total_cases} = {pct:.2f}%\n")
            
            out.write(f"  Extra: {extra} / {total_cases} = {extra_pct:.2f}%\n")
            out.write(f"  Accuracy: {accuracy:.2f}% (100 - Misclassified - Missing - Extra)\n\n")
        
        out.write("\n" + "="*80 + "\n\n")
        
        # Overlap/Non-overlap analysis
        out.write("Overlap/Non-overlap score:\n\n")
        
        # Overlap stats (percentages based on occurrences/sentences, not expressions)
        overlap_occurrences = overlap_stats["occurrences"]
        if overlap_occurrences > 0:
            overlap_mis_pct = safe_pct(overlap_stats["misclassified_sentences"], overlap_occurrences)
            overlap_missing_pct = safe_pct(overlap_stats["missing_sentences"], overlap_occurrences)
            overlap_accuracy = 100.0 - overlap_mis_pct - overlap_missing_pct
            
            out.write(f"Overlap: {overlap_occurrences} occurrences\n")
            out.write(f"  Total expressions = {overlap_stats['total_exprs']}\n")
            out.write(f"  Misclassified: {overlap_stats['misclassified_sentences']} / {overlap_occurrences} = {overlap_mis_pct:.2f}%\n")
            out.write(f"  Missing: {overlap_stats['missing_sentences']} / {overlap_occurrences} = {overlap_missing_pct:.2f}%\n")
            out.write(f"  Accuracy: {overlap_accuracy:.2f}% (100 - Misclassified - Missing)\n\n")
        else:
            out.write("Overlap: 0 occurrences\n\n")
        
        # Non-overlap stats (percentages based on occurrences/sentences, not expressions)
        non_overlap_occurrences = non_overlap_stats["occurrences"]
        if non_overlap_occurrences > 0:
            non_overlap_mis_pct = safe_pct(non_overlap_stats["misclassified_sentences"], non_overlap_occurrences)
            non_overlap_missing_pct = safe_pct(non_overlap_stats["missing_sentences"], non_overlap_occurrences)
            non_overlap_accuracy = 100.0 - non_overlap_mis_pct - non_overlap_missing_pct
            
            out.write(f"Non-overlap (all the rest): {non_overlap_occurrences} occurrences\n")
            out.write(f"  Total expressions = {non_overlap_stats['total_exprs']}\n")
            out.write(f"  Misclassified: {non_overlap_stats['misclassified_sentences']} / {non_overlap_occurrences} = {non_overlap_mis_pct:.2f}%\n")
            out.write(f"  Missing: {non_overlap_stats['missing_sentences']} / {non_overlap_occurrences} = {non_overlap_missing_pct:.2f}%\n")
            out.write(f"  Accuracy: {non_overlap_accuracy:.2f}% (100 - Misclassified - Missing)\n\n")
        else:
            out.write("Non-overlap (all the rest): 0 occurrences\n\n")

        out.write("\n" + "=" * 80 + "\n\n")
        out.write("Expression accuracy by sign category (based on ASL gloss words):\n\n")
        for cat in SIGN_CATEGORY_ORDER:
            stats = sign_category_stats.get(
                cat, {"total": 0, "misclassified": 0, "missing": 0}
            )
            total = stats["total"]
            mis = stats["misclassified"]
            missing = stats["missing"]
            accuracy = max(total - mis - missing, 0)

            mis_pct = safe_pct(mis, total)
            missing_pct = safe_pct(missing, total)
            accuracy_pct = safe_pct(accuracy, total)

            out.write(f"{cat.capitalize()}: total = {total}\n")
            out.write(
                f"  Misclassified: {mis} / {total} = {mis_pct:.2f}%\n"
                if total > 0
                else "  Misclassified: 0 / 0 = 0.00%\n"
            )
            out.write(
                f"  Missing: {missing} / {total} = {missing_pct:.2f}%\n"
                if total > 0
                else "  Missing: 0 / 0 = 0.00%\n"
            )
            out.write(
                f"  Accuracy: {accuracy} / {total} = {accuracy_pct:.2f}%\n\n"
                if total > 0
                else "  Accuracy: 0 / 0 = 0.00%\n\n"
            )

        out.write("\nExpression-level performance by sign category:\n\n")
        for expr in sorted(EXPR_TO_CATEGORY.keys()):
            expr_cat = EXPR_TO_CATEGORY[expr]
            out.write(f"{expr} ({expr_cat}):\n")
            expr_has_data = False
            for sign_cat in SIGN_CATEGORY_ORDER:
                stats = expr_sign_stats.get(expr, {}).get(sign_cat)
                if not stats:
                    continue
                total = stats["total"]
                if total == 0:
                    continue
                expr_has_data = True
                mis = stats["misclassified"]
                missing = stats["missing"]
                accuracy = max(total - mis - missing, 0)

                out.write(
                    f"  {sign_cat.capitalize()}: total = {total}, "
                    f"accuracy = {accuracy} / {total} = {safe_pct(accuracy, total):.2f}%, "
                    f"misclassified = {mis} / {total} = {safe_pct(mis, total):.2f}%, "
                    f"missing = {missing} / {total} = {safe_pct(missing, total):.2f}%\n"
                )

                mis_details = stats["misclassified_as"]
                if mis > 0 and mis_details:
                    sorted_mis = sorted(
                        mis_details.items(), key=lambda kv: kv[1], reverse=True
                    )[:5]
                    out.write("    top misclassifications:\n")
                    for target_expr, count in sorted_mis:
                        out.write(
                            f"        as {target_expr}: {count} / {mis} = "
                            f"{safe_pct(count, mis):.2f}%\n"
                        )
            if not expr_has_data:
                out.write("  No occurrences in dataset.\n")
            out.write("\n")

        out.write("Expression category performance by sign category:\n\n")
        for expr_cat in ["emotion", "mouth_morpheme", "grammar"]:
            out.write(f"{expr_cat.capitalize()}:\n")
            category_has_data = False
            for sign_cat in SIGN_CATEGORY_ORDER:
                stats = expr_category_sign_stats.get(expr_cat, {}).get(sign_cat)
                if not stats:
                    continue
                total = stats["total"]
                if total == 0:
                    continue
                category_has_data = True
                mis = stats["misclassified"]
                missing = stats["missing"]
                accuracy = max(total - mis - missing, 0)
                out.write(
                    f"  {sign_cat.capitalize()}: total = {total}, "
                    f"accuracy = {accuracy} / {total} = {safe_pct(accuracy, total):.2f}%, "
                    f"misclassified = {mis} / {total} = {safe_pct(mis, total):.2f}%, "
                    f"missing = {missing} / {total} = {safe_pct(missing, total):.2f}%\n"
                )
            if not category_has_data:
                out.write("  No occurrences in dataset.\n")
            out.write("\n")


def main():
    # Resolve paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # exp_best_outputs.txt is under ../final_pipeline_txt_220110/best_outputs/
    input_path = os.path.join(
        script_dir,
        "..",
        "final_pipeline_txt_220110",
        "best_outputs",
        "exp_best_outputs.txt",
    )
    input_path = os.path.abspath(input_path)

    output_path = os.path.join("/Users/asheebbansaal/Documents/scifilab/IMWUT26/Language_model/Penultimate/final_pipeline_txt_220110/analysis", "recognition_analysis_summary.txt")

    analyze_file(input_path, output_path)
    print(f"Analysis complete. Summary written to: {output_path}")


if __name__ == "__main__":
    main()


