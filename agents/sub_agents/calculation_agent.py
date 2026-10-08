"""
CalculationAgent —— 税务计算子 Agent
====================================

核心改造：从「纯 LLM 脑补计算」升级为「精确代码计算 + LLM 辅助解释」。

改造前：用户 query → RAG 检索 → 拼 Prompt → LLM 生成（可能算错）
改造后：用户 query → 参数提取 Prompt（LLM 只做解析）
                   → 调 services.calculators 纯函数（100% 精确）
                   → 把计算结果 + 分步过程 + RAG 政策依据拼进 Prompt
                   → LLM 生成最终解释性答复

继承 RAGSubAgent，所以：
  - RAG 检索（"税收计算" 前缀）保留
  - LangGraph 编译 + SqliteSaver checkpointer 保留
  - run() / invoke_messages() 接口保留
override：
  - _rag_node：加入「参数提取 → 精确计算 → 注入结果」的 pre-process
"""

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from services.config_manager import ConfigManager
from agents.sub_agents.rag_sub_agent import RAGSubAgent
from services.calculators import (
    calculate_vat,
    calculate_corporate_income_tax,
    calculate_individual_income_annual,
    calculate_monthly_wage_withholding,
    calculate_business_income_tax,
    calculate_surcharges,
    calculate_stamp_duty,
    fmt_money,
)

logger = logging.getLogger(__name__)

class CalculationAgent(RAGSubAgent):
    """
    税务计算子 Agent（精确计算版）。

    参数提取 + 计算引擎：services.calculators 纯函数
    自然语言解释：LLM（temperature=0.3 保守生成）
    政策依据：RAG 检索（继承 RAGSubAgent）
    """

    NODE_NAME = "calculation"

    # ---- 第一步：参数提取 Prompt（LLM 只解析，不计算） ----
    # 输出严格 JSON，用 json.loads 解析，任何计算错误都是 Prompt 设计的问题
    EXTRACT_PARAMS_PROMPT = """你是税务计算参数提取器。请从用户的问题中提取税务计算所需的参数，输出严格的 JSON（不要任何其他文字）。

可用的计算器类型：
  - vat                 增值税
  - corporate_income    企业所得税
  - individual_annual   个人所得税（综合所得年度汇算）
  - monthly_withholding 个人所得税（工资薪金月度预扣）
  - business_income     个人所得税（经营所得）
  - surcharges          附加税费（城建/教育费/地方教育）
  - stamp_duty          印花税

输出格式（严格 JSON）：
{{
  "calculator_type": "vat",
  "params": {{...该计算器对应的参数字段...}},
  "reasoning": "简短说明提取依据（供调试）"
}}

参数提取规则：
  1. 数字必须填，字符串和布尔值没有默认填 null
  2. 不要填入你猜测的数字，只填用户明确提到的
  3. 如果用户问题需要多个计算器（比如"算增值税和附加税"），选主要的那个，附加税费会自动关联

如果用户问题不是计算类的，返回：
{{"calculator_type": "none", "params": {{}}, "reasoning": "用户无计算需求"}}

用户问题：
{query}
"""

    # ---- 第二步：最终答复 Prompt（覆盖父类，增加精确计算结果注入） ----
    SYSTEM_PROMPT = """你是一位资深的税务计算专家，下面是你已经通过精确代码计算得出的结果，以及检索到的相关政策依据。
请基于这些信息，用清晰、准确、易懂的语言向用户解释计算结果。

## 精确计算结果（代码硬公式，100% 精确）
{calculation_result}

## 税率/公式依据（来自税务知识库）
{retrieved_context}

## 回答格式（严格按此结构）
### 假设前提
（简要说明计算基于的假设，如"假设你是增值税一般纳税人，适用税率 13%"）

### 计算过程
（引用精确计算结果中的 steps，用「公式 + 代入数值 = 结果」格式展示，让用户能核对）

### 最终结果
| 项目 | 金额/税率 |
|------|-----------|
| 应纳税额 | xxx 元 |
| 计税依据 | xxx 元 |
| 适用税率 | xx% |

### 注意事项
（引用精确计算结果中的 notes 和 assumptions，说明边界条件、优惠政策适用前提等）

### 政策依据
（引用 retrieved_context 中的法规条文，标注来源）

重要：**严格按照精确计算结果回答，不要自行修改数字，如果你对某政策有疑问，请说"根据你提供的数据和现行政策，精确计算结果为 xxx，但关于 xxx 方面请以当地税务机关解释为准"。**
"""

    RETRIEVE_PREFIX = "税收计算"
    REF_TAG = "税率依据"
    FALLBACK_TEXT = "（税务计算暂未返回有效内容，请稍后重试）"

    # ---- 元信息 ----
    name = "calculation_agent"
    description = "税务计算智能体（精确计算版）：基于硬编码公式 + 税务知识库 RAG，支持增值税、企业所得税、个人所得税、附加税费、印花税"
    capabilities = [
        "增值税精确计算",
        "企业所得税精确计算（含小微/高新优惠）",
        "个人所得税精确计算（年度汇算/月度预扣/经营所得）",
        "附加税费精确计算",
        "印花税精确计算",
        "应纳税额核算与税负分析",
    ]
    _log_prefix = "CalculationAgent"

    # ==============================================================
    # 覆盖：初始化
    # ==============================================================
    def __init__(self, workdir_root: str = ".", **kwargs):
        super().__init__(workdir_root, **kwargs)

        # 参数提取用 LLM（和主 LLM 共用配置，可用更便宜的模型）
        llm_cfg = ConfigManager().get_llm_config()
        self.extract_llm = ChatOpenAI(
            model=kwargs.get("extract_model_id") or llm_cfg["model"] or "glm-4-flash",
            api_key=kwargs.get("extract_api_key") or llm_cfg["api_key"],
            base_url=kwargs.get("extract_base_url") or llm_cfg["base_url"],
            temperature=0.0,  # 参数提取要绝对确定，temperature=0
            timeout=60.0,
            max_retries=3,
        )

    # ==============================================================
    # 核心：精确计算分发器
    # ==============================================================
    def _run_calculator(self, calculator_type: str, params: dict) -> dict:
        """根据 calculator_type 调对应的纯函数，返回 CalcResult.to_dict() 或 None"""
        dispatch = {
            "vat": lambda: calculate_vat(**_normalize_vat_params(params)),
            "corporate_income": lambda: calculate_corporate_income_tax(**_normalize_corp_params(params)),
            "individual_annual": lambda: calculate_individual_income_annual(**params),
            "monthly_withholding": lambda: calculate_monthly_wage_withholding(**params),
            "business_income": lambda: calculate_business_income_tax(**params),
            "surcharges": lambda: calculate_surcharges(**params),
            "stamp_duty": lambda: calculate_stamp_duty(**params),
        }

        handler = dispatch.get(calculator_type)
        if handler is None:
            return None

        try:
            return handler().to_dict()
        except TypeError as e:
            # 参数缺失 / 类型错误
            logger.warning(f"[CalculationAgent] 计算器参数错误 ({calculator_type}): {e}, params={params}")
            return None
        except Exception as e:
            logger.error(f"[CalculationAgent] 计算器执行异常 ({calculator_type}): {e}", exc_info=True)
            return None

    # ==============================================================
    # 覆盖：RAG 节点（加入精确计算 pre-process）
    # ==============================================================
    def _rag_node(self, state: dict) -> dict:
        """
        参数提取 → 精确计算 → 拼 SystemMessage → LLM 生成最终解释
        """
        # 1. 提取用户 query
        user_query = ""
        for msg in reversed(state["messages"]):
            if hasattr(msg, "content") and hasattr(msg, "__class__") and msg.__class__.__name__ == "HumanMessage":
                user_query = msg.content
                break

        if not user_query:
            return {
                "messages": [SystemMessage(content=self.FALLBACK_TEXT)],
                "step_count": state.get("step_count", 0) + 1,
            }

        # 2. 参数提取（调一个轻量 LLM，temperature=0）
        extraction_result = self._extract_calc_params(user_query)

        # 3. 执行精确计算
        calculation_result_dict: str
        if extraction_result and extraction_result.get("calculator_type") != "none":
            calc_type = extraction_result["calculator_type"]
            raw_params = extraction_result.get("params") or {}

            logger.info(
                f"[CalculationAgent] 参数提取 → calculator={calc_type}, "
                f"params={raw_params}"
            )

            calc_dict = self._run_calculator(calc_type, raw_params)
            if calc_dict:
                # 格式化给 LLM 看：把 CalcResult 关键字段转成人类可读文本
                calculation_result_dict = self._format_calc_result_for_prompt(calc_dict)
                logger.info(
                    f"[CalculationAgent] 精确计算完成 → "
                    f"tax_amount={calc_dict.get('tax_amount')}"
                )
            else:
                calculation_result_dict = "（精确计算执行失败，以下为基于检索到的政策条文的解释）"
        else:
            calculation_result_dict = "（用户问题不涉及具体计算，以下为基于检索到的政策条文的解释）"

        # 4. RAG 检索（继承父类方法）
        retrieved_context = self.retrieve_context(user_query)

        # 5. 拼 SystemMessage（注入精确计算结果 + RAG 上下文）
        system_msg = SystemMessage(
            content=self.SYSTEM_PROMPT.format(
                calculation_result=calculation_result_dict,
                retrieved_context=retrieved_context,
            )
        )

        full_messages = [system_msg] + state["messages"]

        # 6. LLM 生成最终解释
        response = self.llm.invoke(full_messages)

        return {
            "messages": [response],
            "step_count": state.get("step_count", 0) + 1,
            "retrieved_context": retrieved_context,
        }

    # ==============================================================
    # 工具方法：参数提取 + 结果格式化
    # ==============================================================
    def _extract_calc_params(self, query: str) -> dict | None:
        """用轻量 LLM 从 query 中提取计算参数"""
        prompt = self.EXTRACT_PARAMS_PROMPT.format(query=query)

        try:
            raw = self.extract_llm.invoke(prompt)
            text = raw.content if hasattr(raw, "content") else str(raw)

            # 清理可能的 markdown 标记
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            text = text.rstrip("```").strip()

            return json.loads(text)
        except Exception as e:
            logger.warning(f"[CalculationAgent] 参数提取失败: {e}, query={query}")
            return None

    @staticmethod
    def _format_calc_result_for_prompt(calc_dict: dict) -> str:
        """把 CalcResult.to_dict() 格式化为 LLM 友好的文本"""
        parts = []
        parts.append(f"- 税种：{calc_dict.get('title', calc_dict.get('tax_type'))}")
        parts.append(f"- 应纳税额：{calc_dict.get('tax_amount', 0):.2f} 元")
        parts.append(f"- 计税依据：{calc_dict.get('taxable_amount', 0):.2f} 元")
        if calc_dict.get("tax_rate"):
            parts.append(f"- 适用税率：{calc_dict['tax_rate'] * 100:.2f}%")

        # 分步过程
        steps = calc_dict.get("steps", [])
        if steps:
            parts.append("\n📋 分步计算过程：")
            for i, step in enumerate(steps, 1):
                parts.append(f"  {i}. {step.get('label', '步骤')}：")
                parts.append(f"     公式：{step.get('formula', '')}")
                parts.append(f"     结果：{step.get('result', 0):.2f}")

        # 假设和注意事项
        if calc_dict.get("assumptions"):
            parts.append("\n📝 计算假设：")
            for a in calc_dict["assumptions"]:
                parts.append(f"  - {a}")

        if calc_dict.get("notes"):
            parts.append("\n⚠️ 注意事项：")
            for n in calc_dict["notes"]:
                parts.append(f"  - {n}")

        if calc_dict.get("policy_reference"):
            parts.append(f"\n📜 政策依据：{calc_dict['policy_reference']}")

        return "\n".join(parts)

# ============================================================
# 私有参数归一化（LLM 提取的参数可能是 null，需要处理）
# ============================================================

def _safe_float(v, default=0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

def _safe_str(v, default=None) -> str | None:
    if v is None or v == "":
        return default
    return str(v)

def _normalize_vat_params(raw: dict) -> dict:
    """把 LLM 提取的 VAT 参数规范化为 calculate_vat() 的签名"""
    params = {
        "taxpayer_type": raw.get("taxpayer_type", "general") or "general",
        "sales_amount": _safe_float(raw.get("sales_amount")),
        "sales_is_inclusive": bool(raw.get("sales_is_inclusive", False)),
        "input_tax": _safe_float(raw.get("input_tax")),
        "vat_rate": _safe_str(raw.get("vat_rate")),
        "quarterly_sales": _safe_str(raw.get("quarterly_sales")),
    }
    if params["vat_rate"] is not None:
        params["vat_rate"] = float(params["vat_rate"])
    if params["quarterly_sales"] is not None:
        params["quarterly_sales"] = float(params["quarterly_sales"])
    return params

def _normalize_corp_params(raw: dict) -> dict:
    """把 LLM 提取的企业所得税参数规范化"""
    type_mapping = {
        "normal": "normal",
        "small": "small_low_profit",
        "small_low_profit": "small_low_profit",
        "high_tech": "high_tech",
    }
    return {
        "tax_type": type_mapping.get(
            raw.get("tax_type", "normal"), "normal"
        ),
        "taxable_income": _safe_float(raw.get("taxable_income")),
        "prior_loss": _safe_float(raw.get("prior_loss")),
        "rd_expense": _safe_float(raw.get("rd_expense")),
        "direct_deduction": _safe_float(raw.get("direct_deduction")),
    }