
import re
import math
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest


CATEGORY_KEYWORDS = {
    "cash_flow": [
        "cash and cash equivalents",
        "cash from operations",
        "cash flow",
    ],
    "receivables": [
        "accounts receivable",
        "trade receivables",
        "receivables",
    ],
    "payables": [
        "accounts payable",
        "trade payables",
        "payables",
    ],
    "profit_loss": [
        "net profit",
        "operating profit",
        "gross profit",
        "net loss",
        "operating loss",
        "profit",
        "loss",
    ],
    "liabilities": [
        "total liabilities",
        "current liabilities",
        "long-term debt",
        "short-term debt",
        "liabilities",
    ],
    "expenses": [
        "operating expenses",
        "employee expenses",
        "marketing expenses",
        "total expenses",
        "expenses",
        "expenditure",
        "operating costs",
        "operating cost",
        "expense",
        "cost",
    ],
    "revenue": [
        "income from operations",
        "total revenue",
        "revenue",
        "sales",
        "turnover",
    ],
    "assets": [
        "total assets",
        "current assets",
        "fixed assets",
        "non-current assets",
        "assets",
    ],
    "equity": [
        "shareholders' equity",
        "shareholder equity",
        "total equity",
        "equity",
    ],
}

CATEGORY_PRIORITY = [
    "cash_flow",
    "receivables",
    "payables",
    "profit_loss",
    "liabilities",
    "expenses",
    "revenue",
    "assets",
    "equity",
]

AMOUNT_PATTERN = re.compile(
    r"(?P<currency>₹|Rs\.?|INR|USD|\$|€|£)\s*"
    r"(?P<amount>\(?-?\d[\d,]*(?:\.\d+)?\)?)",
    re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(
    r"(?<![\w/.-])"
    r"(?P<amount>\(?-?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|-?\d+(?:\.\d+)?\)?)"
    r"(?![\w/.-])"
)

PAGE_PATTERN = re.compile(
    r"\[?\s*page\s+\d+(?:\s+of\s+\d+)?\s*\]?",
    re.IGNORECASE,
)

CURRENCY_ONLY_NUMBER_PATTERN = re.compile(
    r"^[₹$€£]?\s*\(?-?\d[\d,]*(?:\.\d+)?\)?$"
)


def _to_number(value: str) -> float | None:
    """Convert formatted numeric text to a finite float."""
    value = value.strip().replace(",", "")
    negative = value.startswith("(") and value.endswith(")")
    value = value.strip("()")

    try:
        number = float(value)

        if negative:
            number = -abs(number)

        return number if math.isfinite(number) else None

    except ValueError:
        return None


def _detect_category(context: str) -> str:
    """Infer a category from financial labels in the text."""
    normalized = context.lower()
    matches = []

    for category in CATEGORY_PRIORITY:
        for keyword in CATEGORY_KEYWORDS[category]:
            pattern = (
                r"(?<![a-z])"
                + re.escape(keyword)
                + r"(?![a-z])"
            )

            for match in re.finditer(pattern, normalized):
                matches.append(
                    (match.start(), len(keyword), category)
                )

    if not matches:
        return "uncategorized"

    # Prefer the rightmost label and longer phrases at the
    # same position.
    matches.sort(key=lambda item: (item[0], item[1]))
    return matches[-1][2]


def _is_page_or_metadata_line(line: str) -> bool:
    """Reject explicit page markers and obvious metadata."""
    stripped = line.strip()

    if not stripped:
        return True

    # Examples: [Page 2], Page 2, Page 2 of 10
    if PAGE_PATTERN.fullmatch(stripped):
        return True

    # Example: - 2 -
    if re.fullmatch(r"-\s*\d+\s*-", stripped):
        return True

    if re.search(
        r"\b(?:printed on|generated on)\b",
        stripped,
        re.IGNORECASE,
    ):
        return True

    return False


def extract_financial_amounts(
    text: str,
) -> list[dict[str, Any]]:
    """
    Extract financial amounts and infer their categories.

    Currency-labelled amounts can be extracted from a line.
    Plain numbers require a recognized category on the same line.
    Nearby category labels are used only for numeric-only lines.
    """
    lines = text.splitlines()
    candidates = []
    seen = set()

    line_categories = [
        _detect_category(line)
        for line in lines
    ]

    for index, line in enumerate(lines):
        stripped = line.strip()

        if _is_page_or_metadata_line(stripped):
            continue

        category = line_categories[index]

        # For a numeric-only line, use an adjacent category
        # only if the adjacent labels agree on one category.
        if (
            category == "uncategorized"
            and CURRENCY_ONLY_NUMBER_PATTERN.fullmatch(stripped)
        ):
            nearby_categories = []

            for nearby_index in (index - 1, index + 1):
                if not 0 <= nearby_index < len(lines):
                    continue

                nearby_line = lines[nearby_index].strip()

                if _is_page_or_metadata_line(nearby_line):
                    continue

                nearby_category = line_categories[nearby_index]

                if nearby_category != "uncategorized":
                    nearby_categories.append(nearby_category)

            unique_categories = set(nearby_categories)

            if len(unique_categories) == 1:
                category = unique_categories.pop()

        # Currency amounts are eligible on non-metadata lines.
        matches = list(AMOUNT_PATTERN.finditer(stripped))

        # Plain numeric values are accepted only when the same
        # line has a recognizable financial category.
        if line_categories[index] != "uncategorized":
            matches += list(NUMBER_PATTERN.finditer(stripped))

        for match in matches:
            value = _to_number(match.group("amount"))

            if value is None or value == 0:
                continue

            key = (index + 1, value)

            if key in seen:
                continue

            seen.add(key)

            candidates.append({
                "line_number": index + 1,
                "amount": value,
                "category": category,
                "evidence": stripped[:500],
            })

    return candidates


def detect_financial_anomalies(
    text: str,
    filename: str = "document.pdf",
) -> dict[str, Any]:
    """
    Detect statistical outliers within inferred categories.

    A category needs at least 8 candidate amounts. Findings
    are review signals, not proof of errors or fraud.
    """
    candidates = extract_financial_amounts(text)

    result = {
        "model": "IsolationForest",
        "filename": filename,
        "candidate_count": len(candidates),
        "anomaly_count": 0,
        "findings": [],
        "category_counts": {},
        "categories_tested": 0,
        "status": "completed",
        "message": "",
    }

    groups: dict[str, list[dict[str, Any]]] = {}

    for item in candidates:
        category = item["category"]

        result["category_counts"][category] = (
            result["category_counts"].get(category, 0) + 1
        )

        if category != "uncategorized":
            groups.setdefault(category, []).append(item)

    for category, items in groups.items():
        if len(items) < 8:
            continue

        amounts = np.array(
            [item["amount"] for item in items],
            dtype=float,
        )

        features = np.column_stack([
            np.sign(amounts) * np.log1p(np.abs(amounts)),
            (amounts < 0).astype(float),
        ])

        model = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
        )

        predictions = model.fit_predict(features)
        scores = model.decision_function(features)

        result["categories_tested"] += 1

        absolute_amounts = np.abs(amounts)
        median_amount = float(np.median(absolute_amounts))

        for item, prediction, score in zip(
            items, predictions, scores
        ):
            if prediction != -1:
                continue

            if median_amount <= 0:
                continue

            relative_deviation = (
                abs(abs(item["amount"]) - median_amount)
                / median_amount
            )

            # Keep only outliers more than 30% away from
            # the category median.
            if relative_deviation <= 0.30:
                continue

            result["findings"].append({
                "type": "statistical_outlier",
                "severity": "review",
                "category": category,
                "amount": item["amount"],
                "line_number": item["line_number"],
                "evidence": item["evidence"],
                "anomaly_score": round(float(score), 4),
                "relative_deviation": round(
                    float(relative_deviation), 4
                ),
                "message": (
                    f"This amount was identified as statistically "
                    f"unusual within the inferred "
                    f"{category.replace('_', ' ')} category. "
                    f"It differs from the category median by "
                    f"{relative_deviation:.1%}. Verify the amount "
                    "and category against the original report."
                ),
            })

    result["anomaly_count"] = len(result["findings"])

    if result["categories_tested"] == 0:
        result["status"] = "insufficient_category_data"
        result["message"] = (
            "No category had at least 8 candidate amounts. "
            "Category-specific ML analysis was skipped."
        )
    else:
        result["message"] = (
            f"Analysis completed for "
            f"{result['categories_tested']} categories. "
            "Flagged values require manual review."
        )

    return result

import re
import math
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest


CATEGORY_KEYWORDS = {
    "cash_flow": [
        "cash and cash equivalents",
        "cash from operations",
        "cash flow",
    ],
    "receivables": [
        "accounts receivable",
        "trade receivables",
        "receivables",
    ],
    "payables": [
        "accounts payable",
        "trade payables",
        "payables",
    ],
    "profit_loss": [
        "net profit",
        "operating profit",
        "gross profit",
        "net loss",
        "operating loss",
        "profit",
        "loss",
    ],
    "liabilities": [
        "total liabilities",
        "current liabilities",
        "long-term debt",
        "short-term debt",
        "liabilities",
    ],
    "expenses": [
        "operating expenses",
        "employee expenses",
        "marketing expenses",
        "total expenses",
        "expenses",
        "expenditure",
        "operating costs",
        "operating cost",
        "expense",
        "cost",
    ],
    "revenue": [
        "income from operations",
        "total revenue",
        "revenue",
        "sales",
        "turnover",
    ],
    "assets": [
        "total assets",
        "current assets",
        "fixed assets",
        "non-current assets",
        "assets",
    ],
    "equity": [
        "shareholders' equity",
        "shareholder equity",
        "total equity",
        "equity",
    ],
}

CATEGORY_PRIORITY = [
    "cash_flow",
    "receivables",
    "payables",
    "profit_loss",
    "liabilities",
    "expenses",
    "revenue",
    "assets",
    "equity",
]

AMOUNT_PATTERN = re.compile(
    r"(?P<currency>₹|Rs\.?|INR|USD|\$|€|£)\s*"
    r"(?P<amount>\(?-?\d[\d,]*(?:\.\d+)?\)?)",
    re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(
    r"(?<![\w/.-])"
    r"(?P<amount>\(?-?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|-?\d+(?:\.\d+)?\)?)"
    r"(?![\w/.-])"
)

PAGE_PATTERN = re.compile(
    r"\[?\s*page\s+\d+(?:\s+of\s+\d+)?\s*\]?",
    re.IGNORECASE,
)

CURRENCY_ONLY_NUMBER_PATTERN = re.compile(
    r"^[₹$€£]?\s*\(?-?\d[\d,]*(?:\.\d+)?\)?$"
)


def _to_number(value: str) -> float | None:
    """Convert formatted numeric text to a finite float."""
    value = value.strip().replace(",", "")
    negative = value.startswith("(") and value.endswith(")")
    value = value.strip("()")

    try:
        number = float(value)

        if negative:
            number = -abs(number)

        return number if math.isfinite(number) else None

    except ValueError:
        return None


def _detect_category(context: str) -> str:
    """Infer a category from financial labels in the text."""
    normalized = context.lower()
    matches = []

    for category in CATEGORY_PRIORITY:
        for keyword in CATEGORY_KEYWORDS[category]:
            pattern = (
                r"(?<![a-z])"
                + re.escape(keyword)
                + r"(?![a-z])"
            )

            for match in re.finditer(pattern, normalized):
                matches.append(
                    (match.start(), len(keyword), category)
                )

    if not matches:
        return "uncategorized"

    # Prefer the rightmost label and longer phrases at the
    # same position.
    matches.sort(key=lambda item: (item[0], item[1]))
    return matches[-1][2]


def _is_page_or_metadata_line(line: str) -> bool:
    """Reject explicit page markers and obvious metadata."""
    stripped = line.strip()

    if not stripped:
        return True

    # Examples: [Page 2], Page 2, Page 2 of 10
    if PAGE_PATTERN.fullmatch(stripped):
        return True

    # Example: - 2 -
    if re.fullmatch(r"-\s*\d+\s*-", stripped):
        return True

    if re.search(
        r"\b(?:printed on|generated on)\b",
        stripped,
        re.IGNORECASE,
    ):
        return True

    return False


def extract_financial_amounts(
    text: str,
) -> list[dict[str, Any]]:
    """
    Extract financial amounts and infer their categories.

    Currency-labelled amounts can be extracted from a line.
    Plain numbers require a recognized category on the same line.
    Nearby category labels are used only for numeric-only lines.
    """
    lines = text.splitlines()
    candidates = []
    seen = set()

    line_categories = [
        _detect_category(line)
        for line in lines
    ]

    for index, line in enumerate(lines):
        stripped = line.strip()

        if _is_page_or_metadata_line(stripped):
            continue

        category = line_categories[index]

        # For a numeric-only line, use an adjacent category
        # only if the adjacent labels agree on one category.
        if (
            category == "uncategorized"
            and (
                CURRENCY_ONLY_NUMBER_PATTERN.fullmatch(stripped)
                or AMOUNT_PATTERN.fullmatch(stripped)
            )
        ):
            nearby_categories = []

            for nearby_index in (index - 1, index + 1):
                if not 0 <= nearby_index < len(lines):
                    continue

                nearby_line = lines[nearby_index].strip()

                if _is_page_or_metadata_line(nearby_line):
                    continue

                nearby_category = line_categories[nearby_index]

                if nearby_category != "uncategorized":
                    nearby_categories.append(nearby_category)

            unique_categories = set(nearby_categories)

            if len(unique_categories) == 1:
                category = unique_categories.pop()

        # Currency amounts are eligible on non-metadata lines.
        matches = list(AMOUNT_PATTERN.finditer(stripped))

        # Plain numeric values are accepted only when the same
        # line has a recognizable financial category.
        if category != "uncategorized":
            matches += list(NUMBER_PATTERN.finditer(stripped))

        for match in matches:
            value = _to_number(match.group("amount"))

            if value is None or value == 0:
                continue

            key = (index + 1, value)

            if key in seen:
                continue

            seen.add(key)

            candidates.append({
                "line_number": index + 1,
                "amount": value,
                "category": category,
                "evidence": stripped[:500],
            })

    return candidates


def detect_financial_anomalies(
    text: str,
    filename: str = "document.pdf",
) -> dict[str, Any]:
    """
    Detect statistical outliers within inferred categories.

    A category needs at least 8 candidate amounts. Findings
    are review signals, not proof of errors or fraud.
    """
    candidates = extract_financial_amounts(text)

    result = {
        "model": "IsolationForest",
        "filename": filename,
        "candidate_count": len(candidates),
        "anomaly_count": 0,
        "findings": [],
        "category_counts": {},
        "categories_tested": 0,
        "status": "completed",
        "message": "",
    }

    groups: dict[str, list[dict[str, Any]]] = {}

    for item in candidates:
        category = item["category"]

        result["category_counts"][category] = (
            result["category_counts"].get(category, 0) + 1
        )

        if category != "uncategorized":
            groups.setdefault(category, []).append(item)

    for category, items in groups.items():
        if len(items) < 8:
            continue

        amounts = np.array(
            [item["amount"] for item in items],
            dtype=float,
        )

        features = np.column_stack([
            np.sign(amounts) * np.log1p(np.abs(amounts)),
            (amounts < 0).astype(float),
        ])

        model = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
        )

        predictions = model.fit_predict(features)
        scores = model.decision_function(features)

        result["categories_tested"] += 1

        absolute_amounts = np.abs(amounts)
        median_amount = float(np.median(absolute_amounts))

        for item, prediction, score in zip(
            items, predictions, scores
        ):
            if median_amount <= 0:
                continue

            relative_deviation = (
                abs(abs(item["amount"]) - median_amount)
                / median_amount
            )

            # Use Isolation Forest as one signal, but also apply a
            # transparent large-deviation rule. This helps prevent
            # an obvious extreme value from being missed merely
            # because Isolation Forest did not label it as an outlier.
            is_extreme_by_median = relative_deviation >= 3.0

            if prediction != -1 and not is_extreme_by_median:
                continue

            # Ignore small deviations unless the value is an
            # extreme (at least 3x the category median deviation).
            if relative_deviation <= 0.30:
                continue

            result["findings"].append({
                "type": "statistical_outlier",
                "severity": "review",
                "category": category,
                "amount": item["amount"],
                "line_number": item["line_number"],
                "evidence": item["evidence"],
                "anomaly_score": round(float(score), 4),
                "relative_deviation": round(
                    float(relative_deviation), 4
                ),
                "message": (
                    f"This amount was identified as statistically "
                    f"unusual within the inferred "
                    f"{category.replace('_', ' ')} category. "
                    f"It differs from the category median by "
                    f"{relative_deviation:.1%}. Verify the amount "
                    "and category against the original report."
                ),
            })

    result["anomaly_count"] = len(result["findings"])

    if result["categories_tested"] == 0:
        result["status"] = "insufficient_category_data"
        result["message"] = (
            "No category had at least 8 candidate amounts. "
            "Category-specific ML analysis was skipped."
        )
    else:
        result["message"] = (
            f"Analysis completed for "
            f"{result['categories_tested']} categories. "
            "Flagged values require manual review."
        )

    return result
