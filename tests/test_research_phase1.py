import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval import research_phase1 as phase1


@pytest.mark.parametrize("mode", phase1.MODES)
def test_mock_matrix_switches(mode, tmp_path):
    case = phase1.read_json(phase1.DATASET)["cases"][0]
    job = {"case": case, "mode": mode, "live": False, "budget": {}, "timeout": 25}
    result = asyncio.run(phase1.run_case(job, tmp_path))
    online, populated = phase1.MODES[mode]
    calls = result["tools"]["retrieval"]
    assert calls["search_tax_policy"] == {"success" if populated else "empty": 1}
    assert bool(calls["search_web_policy"]) is online
    assert bool(calls["read_web_page"]) is online
    assert result["clarification_sent"]
    assert any(call["purpose"] == "ClarificationUpdate" for call in result["model"]["calls"])
    assert result["status"] == "insufficient_evidence"
    assert not result["report_generated"]


def test_reference_answers_are_separate_from_worker_inputs():
    raw_cases = phase1.DATASET.read_text(encoding="utf-8")
    refs = phase1.read_json(phase1.DATASET.with_name("phase1_references.json"))["references"]
    assert set(refs) <= {case["id"] for case in phase1.read_json(phase1.DATASET)["cases"]}
    for ref in refs.values():
        assert ref["source_url"] not in raw_cases
        assert ref["answer_summary"] not in raw_cases


def test_summary_preserves_retrieval_failure_and_omits_raw_inputs():
    private = "sensitive-test-fixture"
    data = {
        "status": "failed", "stage": "assess",
        "failure": {"code": "structured_output_invalid", "message": private,
                    "details": {"output_error": "output_truncated", "validation_issues": [private]}},
        "metrics": {
            "llm_calls": [{"purpose": "Assessment", "outcome": "output_truncated", "response": private}],
            "tool_calls": [
                {"tool": "search_tax_policy", "status": "error", "args": {"positional": [private]}},
                {"tool": "search_web_policy", "status": "empty"},
                {"tool": "read_web_page", "status": "error",
                 "args": {"positional": [f"https://www.chinatax.gov.cn/page?token={private}"]}},
            ],
        },
    }
    summary = phase1.summarize(data)
    assert set(summary["categories"]) == {"structured_output_failed", "local_policy_failed", "official_web_failed"}
    assert summary["model"]["outcomes"] == {"output_truncated": 1}
    assert summary["tools"]["success_rate"] == pytest.approx(1 / 3)
    assert private not in str(summary)


@pytest.mark.parametrize("source_check, sufficient, expected", [
    ("official_domain", True, "completed_sufficient_evidence"),
    ("official_domain", False, "completed_insufficient_evidence"),
    ("stored", True, "completed_sufficient_evidence"),
    ("unverified", True, "other_failure"),
])
def test_report_classification_distinguishes_coverage_and_provenance(source_check, sufficient, expected):
    summary = phase1.summarize({
        "status": "awaiting_review",
        "questions": ["core question"],
        "assessment": {"coverage": [{"question_id": "q1", "sufficient": sufficient}]},
        "report": {"policy_evidence": [{"evidence_id": "e1", "content": "可读原文", "kind": "policy_rag",
                    "file_id": "stored-file" if source_check == "stored" else None,
                    "source_url": "https://www.chinatax.gov.cn/test" if source_check == "official_domain" else None}],
                   "analysis": [{"question_id": "q1", "evidence_ids": ["e1"]}]},
    })
    assert summary["categories"] == [expected]


def test_reference_analysis_status_never_counts_as_report():
    summary = phase1.summarize({"status": "reference_analysis", "reference_analysis": {"notice": "待核验"}})
    assert summary["categories"] == ["completed_unverified_analysis"]
    assert not summary["report_generated"] and summary["reference_analysis_generated"]


def test_reference_queue_compares_source_discovery_after_execution(tmp_path):
    reference = phase1.read_json(phase1.REFERENCES)["references"]["sta-prepayment"]
    url = reference["source_url"]
    phase1.write_review_queue(tmp_path, [{
        "case_id": "sta-prepayment", "mode": "C", "live": True, "status": "failed", "report_generated": False,
        "tools": {"calls": [{"tool": "search_web_policy", "urls": [url]},
                            {"tool": "read_web_page", "url": url, "status": "success"}]},
    }])
    entry, = phase1.read_json(tmp_path / "reference_review.json")
    assert entry["status"] == "no_report"
    assert entry["target_found"] and entry["target_read"]
    assert entry["reference"] == reference


def test_matrix_keeps_running_after_a_worker_timeout(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.json"
    phase1.write_json(dataset, {"cases": [{"id": "first"}, {"id": "second"}]})
    monkeypatch.setattr(phase1, "DATASET", dataset)
    monkeypatch.setattr(phase1, "ROOT", tmp_path)
    monkeypatch.setattr(phase1, "SNAPSHOT", dataset)
    # The manifest also fingerprints the application modules.
    for path in ("agents/research_agent.py", "agents/research_output.py",
                 "services/tools/kb_tools.py", "knowledge/embedding.py"):
        source = tmp_path / path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()
    monkeypatch.setattr(phase1, "__file__", str(dataset))

    def run(command, **kwargs):
        assert kwargs["timeout"] <= 10
        workdir = Path(command[-1])
        if workdir.name == "A-first":
            phase1.write_json(workdir / "progress.json", {"stage": "retrieve"})
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        phase1.write_json(workdir / "result.json", {"status": "insufficient_evidence",
                          "categories": ["completed_insufficient_evidence"]})

    monkeypatch.setattr(phase1.subprocess, "run", run)
    options = SimpleNamespace(case=None, live=False, modes=["A"], case_timeout=10, total_timeout=30,
                              max_model_calls=5, max_tool_calls=4, max_tokens=8000, rounds=1, save_drafts=False)
    first, second = phase1.run_matrix(options)
    assert first["failure"] == {"code": "case_timeout", "stage": "retrieve"}
    assert first["metrics_partial"]
    assert second["status"] == "insufficient_evidence"
