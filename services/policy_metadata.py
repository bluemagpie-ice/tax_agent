"""Extract only explicit metadata; absence never implies current validity."""
import re
from datetime import date


def extract_policy_metadata(text):
    metadata = {"validity_check": "unknown", "is_current": None}
    compact = re.sub(r"\s+", "", text)
    pattern = r"((?:19|20)\d{2})年(\d{1,2})月(\d{1,2})日"
    for name, regex in (
        ("effective_date", r"(?:自|从)" + pattern + r"起?(?:施行|执行|生效|至)"),
        ("expiry_date", r"(?:至|执行至|有效期至)" + pattern),
    ):
        match = re.search(regex, compact)
        if match:
            try:
                metadata[name] = date(*map(int, match.groups())).isoformat()
            except ValueError:
                pass
    number = re.search(r"(?:财税|财政部(?:国家)?税务总局公告|国家税务总局公告|税务总局公告)[〔\[（(]?(?:19|20)\d{2}[〕\]）)年]?第?\d+号", compact)
    if number:
        metadata["document_number"] = number.group()
    for label, kind in (("增值税", "vat"), ("企业所得税", "corporate_income"), ("个人所得税", "individual_income"), ("印花税", "stamp_duty")):
        if label in text:
            metadata["tax_type"] = kind
            break
    return metadata
