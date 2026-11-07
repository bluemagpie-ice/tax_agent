"""Conservative temporal interpretation; publication years are never validity."""
import re
from datetime import date


def period_status(quotes, tax_period):
    years = re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", tax_period or "")
    if not years:
        return "unknown" if tax_period else "not_requested"
    start, end = date(int(min(years)), 1, 1), date(int(max(years)), 12, 31)
    for raw in quotes:
        text = re.sub(r"\s+", "", raw)
        span = re.search(r"((?:19|20)\d{2})(?:年(?:(\d{1,2})月(?:(\d{1,2})日)?)?)?[至到—–-]((?:19|20)\d{2})(?:年(?:(\d{1,2})月(?:(\d{1,2})日)?)?)?", text)
        if span and re.search(r"施行|执行|适用|有效|期间|优惠|免征|减征", text):
            try:
                left = date(int(span[1]), int(span[2] or 1), int(span[3] or 1))
                right = date(int(span[4]), int(span[5] or 12), int(span[6] or 31))
            except ValueError:
                continue
            if left <= start and right >= end:
                return "supported"
        # An open-ended start cannot prove that no later repeal exists.
        began = re.search(r"(?:自|从)?((?:19|20)\d{2})年(?:(\d{1,2})月(?:(\d{1,2})日)?)?起?(?:施行|执行|适用|生效)", text)
        if began and int(began[1]) <= int(min(years)):
            return "pending_verification"
    return "unknown"
