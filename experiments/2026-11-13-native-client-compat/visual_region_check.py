"""Supplement the archived check with an explicit search-field defect in one reported region."""
import re


def search_field_clipped(answer):
    entries = answer.get("regressions", [])
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        region = str(entry.get("region", "")).lower()
        change = str(entry.get("change", "")).lower()
        clauses = re.split(r"[.;,]|\b(?:and|while|but)\s+(?:the\s+)?(?=eth|price|value|table)", change)
        for clause in clauses:
            if not re.search(r"\b(search|placeholder)\b", region + " " + clause):
                continue
            # The seeded mask removes the right part of the control. Vertical-only
            # truncation is a different defect; leave an unqualified "clipped" eligible.
            if re.search(r"\bvertical(?:ly)?\b", clause) and not re.search(
                r"\bhorizontal(?:ly)?\b|\bwidth\b|\bright\b", clause
            ):
                continue
            if re.search(r"\b(clip\w*|truncat\w*|narrow\w*|shorten\w*|cut[ -]off|"
                         r"occlud\w*|obscur\w*|overlap\w*)\b|"
                         r"\b(?:width.{0,24}reduc\w*|reduc\w*.{0,24}width)\b", clause):
                return True
    return False
