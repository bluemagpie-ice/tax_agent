"""Decode Research model responses without treating reasoning or schema as data."""
import json
import re
from urllib.parse import urlsplit

from pydantic import ValidationError


class InvalidOutput(ValueError):
    def __init__(self, code, *, issues=(), candidate=""):
        self.code = code
        self.issues = issues
        self.candidate = candidate


def model_options(model, reserve, *, simple):
    name = getattr(model, "model_name", "")
    base_url = getattr(model, "openai_api_base", None)
    if (isinstance(base_url, str) and urlsplit(base_url).hostname == "api.deepseek.com"
            and name in ("deepseek-flash", "deepseek-v4-pro")):
        # ChatOpenAI maps top-level max_tokens to max_completion_tokens.
        return {"response_format": {"type": "json_object"},
                "extra_body": {"max_tokens": reserve, "thinking": {"type": "disabled" if simple else "enabled"}},
                **({} if simple else {"reasoning_effort": "low"})}
    return {"reasoning_effort": "low"} if name == "glm-5.3" else {}


def _answer_content(result):
    if isinstance(result, str):
        return result
    content = result.get("content") if isinstance(result, dict) else getattr(result, "content", None)
    if isinstance(content, list):
        return "".join(part if isinstance(part, str) else part["text"]
                       for part in content if isinstance(part, str) or (
                           isinstance(part, dict) and part.get("type") in ("text", "output_text")
                           and isinstance(part.get("text"), str)))
    return content


def _unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate JSON keys")
    return result


def _reject_constant(_value):
    raise ValueError("non-JSON numeric constant")


def parse_output(result, schema, *, finish_reason=None, secret="", array_fields=(), repairs=None):
    if finish_reason in ("length", "max_tokens", "max_output_tokens"):
        raise InvalidOutput("output_truncated")
    if finish_reason in ("content_filter", "refusal") or (getattr(result, "additional_kwargs", None) or {}).get("refusal"):
        raise InvalidOutput("output_refused")
    raw = _answer_content(result)
    if isinstance(raw, dict):
        raw = json.dumps(raw, ensure_ascii=False)
    if not isinstance(raw, str) or not raw.strip().lstrip("\ufeff").strip():
        raise InvalidOutput("output_empty")
    # Redact before including any candidate in a second request or storing values.
    if secret:
        for form in {secret, json.dumps(secret)[1:-1], json.dumps(secret, ensure_ascii=False)[1:-1]}:
            raw = raw.replace(form, "[密钥已脱敏]")
    raw = raw.strip().lstrip("\ufeff").strip()
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    try:
        # Accept one complete object with an optional fence or surrounding prose.
        # Never recover a nested object from a broken object or a top-level array.
        start = raw.find("{")
        if start < 0 or re.search(r"[\[\]{}]", raw[:start]):
            raise ValueError
        value, end = decoder.raw_decode(raw, start)
        if re.search(r"[\[\]{}]", raw[end:]):
            raise ValueError
    except (ValueError, RecursionError):
        raise InvalidOutput("output_json_invalid", candidate=raw) from None

    fields = schema.model_fields
    wrapped = value.get("properties")
    applied = []
    if (isinstance(wrapped, dict) and set(value) <= {"properties", "required", "title", "type", "$defs", "$schema"}
            and set(wrapped) & fields.keys()):
        # An envelope with real values is recoverable. Actual field schema
        # descriptors must still fail validation instead of becoming facts.
        value = wrapped
        applied.append({"kind": "properties_wrapper"})
    if not set(value) & fields.keys():
        raise InvalidOutput("output_schema_invalid", candidate=raw)
    for field in array_fields:
        wrapper = value.get(field)
        if (isinstance(wrapper, dict) and wrapper.get("type") == "array"
                and wrapper.keys() <= {"type", "items", "description", "title", "maxItems"}
                and isinstance(wrapper.get("description", ""), str) and isinstance(wrapper.get("title", ""), str)
                and isinstance(wrapper.get("items"), list)
                and ("maxItems" not in wrapper or (type(wrapper["maxItems"]) is int
                     and wrapper["maxItems"] >= len(wrapper["items"])))
                and all(isinstance(item, str) for item in wrapper["items"])):
            value[field] = wrapper["items"]
            applied.append({"kind": "array_items_wrapper", "field": field})
    try:
        parsed = schema.model_validate(value, strict=True).model_dump()
    except ValidationError as exc:
        issues = [{"field": error["loc"][0] if error["loc"] and error["loc"][0] in fields else "",
                   "type": error["type"],
                   **{key: value for key, value in (error.get("ctx") or {}).items()
                      if key in ("min_length", "max_length") and type(value) is int}}
                  for error in exc.errors(include_input=False)[:20]]
        raise InvalidOutput("output_schema_invalid", issues=issues, candidate=raw) from None
    if repairs is not None:
        repairs.extend(applied)
    return parsed
