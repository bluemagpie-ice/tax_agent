"""Draft a case from a finalized report, never publish it automatically."""
import json
import copy
import re

from services.research_cases import CaseContent


def candidate_from_task(task, *, generalize=False):
    if task.get("status") != "finalized":
        raise ValueError("只有已定稿报告可以提取候选案例")
    report = task["report"]
    candidate = CaseContent(
        title=task.get("original_query", task["query"])[:120], question=task.get("original_query", task["query"]),
        facts=task["facts"], tax_types=task.get("tax_types", []), region=task.get("region"), tax_period=task.get("tax_period"),
        business_type=task.get("business_type"), enterprise_type=task.get("enterprise_type"),
        conditions=["待确认：" + item for item in report.get("unconfirmed_facts", [])], evidence=task["evidence"],
        conclusion=report["answer_draft"], unresolved=report.get("uncertainties", []),
    ).model_dump()
    if generalize:
        # Only a preview convenience; users must inspect free text for remaining identifiers.
        # Generalize case facts/opinion, never silently alter public legal thresholds/source URLs.
        sources = copy.deepcopy(candidate.pop("evidence"))
        text = json.dumps(candidate, ensure_ascii=False)
        text = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[手机号]", text)
        text = re.sub(r"(?<![A-Za-z0-9])\d{17}[\dXx](?![A-Za-z0-9])", "[身份标识]", text)
        text = re.sub(r"[\d,]+(?:\.\d+)?\s*(?:亿元|万元|元)", "[金额]", text)
        text = re.sub(r"[\u4e00-\u9fffA-Za-z]{2,30}(?:有限责任公司|股份有限公司|有限公司)", "[企业]", text)
        text = re.sub(r"(?:合同编号|合同号)[：: ]*[A-Za-z0-9_-]+", "合同编号：[已删除]", text)
        candidate = {**json.loads(text), "evidence":sources}
    return {"content": candidate, "source_task_id":task["task_id"], "source_report_version":task["report_version"],
            "notice":"候选案例尚未保存或发布。请检查全文、敏感信息及政策依据；泛化不保证完整脱敏。"}
