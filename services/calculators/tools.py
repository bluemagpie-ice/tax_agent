"""
把纯函数计算器封装成 LangChain StructuredTool
=============================================
作用：让 LLM 可以通过 function_call 自主调用计算器（ReAct 模式）。
但默认**不强制使用 Tool**——CalculationAgent 先走纯函数强制调用路径，
Tool 封装作为备选方案，后续需要时再接入。

封装原则：
  1. Tool 内部**不写计算逻辑**，只做「参数提取 → 调纯函数 → 返回 CalcResult.to_dict()」
  2. 统一异常兜底，不让 LLM 看到 traceback
"""

import json
import logging
from typing import Optional, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from services.calculators import (
    calculate_vat,
    calculate_corporate_income_tax,
    calculate_monthly_wage_withholding,
    calculate_individual_income_annual,
    calculate_business_income_tax,
    calculate_surcharges,
    calculate_stamp_duty,
    CORP_NORMAL,
    CORP_SMALL_LOW_PROFIT,
    CORP_HIGH_TECH,
)

logger = logging.getLogger(__name__)

# ============================================================
# 1. 增值税计算 Tool
# ============================================================

class VATInput(BaseModel):
    """增值税计算输入参数"""
    taxpayer_type: Literal["general", "small"] = Field(
        default="general",
        description="纳税人类型：general=一般纳税人，small=小规模纳税人",
    )
    sales_amount: float = Field(description="销售额（数字）")
    sales_is_inclusive: bool = Field(
        default=False, description="销售额是否含税（默认不含税）"
    )
    input_tax: float = Field(default=0.0, description="进项税额（仅一般纳税人抵扣用）")
    vat_rate: Optional[float] = Field(
        default=None,
        description="税率（0.13/0.09/0.06/0.03/0.01），为 None 时按纳税人类型自动取默认值",
    )
    quarterly_sales: Optional[float] = Field(
        default=None, description="小规模季度累计销售额（判断 30 万免征）"
    )

def _tool_calculate_vat(
    taxpayer_type: str = "general",
    sales_amount: float = 0.0,
    sales_is_inclusive: bool = False,
    input_tax: float = 0.0,
    vat_rate: float | None = None,
    quarterly_sales: float | None = None,
) -> str:
    try:
        result = calculate_vat(
            taxpayer_type=taxpayer_type,
            sales_amount=sales_amount,
            sales_is_inclusive=sales_is_inclusive,
            input_tax=input_tax,
            vat_rate=vat_rate,
            quarterly_sales=quarterly_sales,
        )
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[VATTool] 计算失败: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)

VAT_TOOL = StructuredTool.from_function(
    func=_tool_calculate_vat,
    name="calculate_vat",
    description=(
        "计算增值税应纳税额。"
        "支持一般纳税人（销项-进项抵扣，税率 13%/9%/6%）和小规模纳税人"
        "（不含进项抵扣，征收率 1% 或 3%，季度 ≤ 30 万免征）。"
        "含税销售额需要设置 sales_is_inclusive=True。"
    ),
    args_schema=VATInput,
    return_direct=False,
)

# ============================================================
# 2. 企业所得税计算 Tool
# ============================================================

class CorporateIncomeInput(BaseModel):
    tax_type: Literal["normal", "small_low_profit", "high_tech"] = Field(
        default="normal",
        description="企业类型：normal=普通企业(25%)，small_low_profit=小型微利(分段优惠)，high_tech=高新(15%)",
    )
    taxable_income: float = Field(description="应纳税所得额（已扣除成本费用后的余额）")
    prior_loss: float = Field(default=0.0, description="以前年度可弥补亏损")
    rd_expense: float = Field(default=0.0, description="研发费用（触发加计扣除，比例 100%）")
    direct_deduction: float = Field(default=0.0, description="直接减免税额")

def _tool_calculate_corporate_income(
    tax_type: str = "normal",
    taxable_income: float = 0.0,
    prior_loss: float = 0.0,
    rd_expense: float = 0.0,
    direct_deduction: float = 0.0,
) -> str:
    mapping = {"normal": CORP_NORMAL, "small_low_profit": CORP_SMALL_LOW_PROFIT, "high_tech": CORP_HIGH_TECH}
    try:
        result = calculate_corporate_income_tax(
            tax_type=mapping.get(tax_type, CORP_NORMAL),
            taxable_income=taxable_income,
            prior_loss=prior_loss,
            rd_expense=rd_expense,
            direct_deduction=direct_deduction,
        )
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[CorpIncomeTool] 计算失败: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)

CORPORATE_INCOME_TOOL = StructuredTool.from_function(
    func=_tool_calculate_corporate_income,
    name="calculate_corporate_income_tax",
    description=(
        "计算企业所得税。"
        "普通企业法定税率 25%；小型微利企业 ≤ 100 万实际税率 5%、"
        "100~300 万实际税率 10%；高新技术企业税率 15%。"
        "支持研发费用加计扣除（100%）和以前年度亏损弥补。"
    ),
    args_schema=CorporateIncomeInput,
    return_direct=False,
)

# ============================================================
# 3. 个人所得税（综合所得年度）计算 Tool
# ============================================================

class IndividualIncomeInput(BaseModel):
    annual_wage: float = Field(default=0.0, description="年度工资薪金收入")
    annual_labor: float = Field(default=0.0, description="年度劳务报酬收入（80% 计入综合所得）")
    annual_royalty: float = Field(default=0.0, description="年度稿酬收入（80%×70% 计入）")
    annual_author_fee: float = Field(default=0.0, description="年度特许权使用费收入（80% 计入）")
    special_deduction: float = Field(default=0.0, description="专项扣除（社保公积金个人缴纳部分）")
    special_additional_deduction: float = Field(
        default=0.0, description="专项附加扣除（子女教育/房贷/赡养老人等）"
    )
    paid_tax: float = Field(default=0.0, description="年度已预缴税额（用于汇算清缴补退）")

def _tool_calculate_individual_annual(**kwargs) -> str:
    try:
        result = calculate_individual_income_annual(**kwargs)
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[IndIncomeTool] 计算失败: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)

INDIVIDUAL_INCOME_ANNUAL_TOOL = StructuredTool.from_function(
    func=_tool_calculate_individual_annual,
    name="calculate_individual_income_annual",
    description=(
        "计算个人所得税综合所得年度汇算清缴。"
        "综合所得 = 工资薪金 + 劳务报酬×80% + 稿酬×80%×70% + 特许权使用费×80%。"
        "扣除项目：6 万基本减除 + 专项扣除（社保）+ 专项附加扣除（子女教育/房贷/赡养等）。"
        "税率：5~45% 超额累进。"
    ),
    args_schema=IndividualIncomeInput,
    return_direct=False,
)

# ============================================================
# 4. 个人所得税（月度预扣）计算 Tool
# ============================================================

class MonthlyWithholdingInput(BaseModel):
    monthly_wage: float = Field(description="本月工资薪金")
    month_index: int = Field(description="第几月（1-12，用于累计预扣法）")
    special_deduction_monthly: float = Field(default=0.0, description="社保公积金月度个人部分")
    special_additional_monthly: float = Field(default=0.0, description="专项附加扣除月度")
    prior_tax_withheld: float = Field(default=0.0, description="截至上月累计已预扣税额")
    prior_income: float = Field(default=0.0, description="截至上月累计工资收入")

def _tool_calculate_monthly_withholding(**kwargs) -> str:
    try:
        result = calculate_monthly_wage_withholding(**kwargs)
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[MonthlyWithholdTool] 计算失败: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)

MONTHLY_WITHHOLDING_TOOL = StructuredTool.from_function(
    func=_tool_calculate_monthly_withholding,
    name="calculate_monthly_wage_withholding",
    description=(
        "计算工资薪金月度个人所得税预扣额（累计预扣法）。"
        "累计预扣法：累计收入 - 累计扣除 → 年度税率表算累计应纳税额 → 减去已预扣 = 本月应预扣。"
        "month_index 必须是当年第几月（1-12）。"
    ),
    args_schema=MonthlyWithholdingInput,
    return_direct=False,
)

# ============================================================
# 5. 附加税费计算 Tool
# ============================================================

class SurchargesInput(BaseModel):
    vat_paid: float = Field(description="实际缴纳的增值税")
    consumption_tax_paid: float = Field(default=0.0, description="实际缴纳的消费税（可选）")
    city_type: Literal["urban", "county", "other"] = Field(
        default="urban", description="城建税税率档次：urban=市区7%，county=县城5%，other=其他地区1%"
    )
    is_small_scale: bool = Field(default=False, description="是否小规模（附加税费减半）")

def _tool_calculate_surcharges(**kwargs) -> str:
    try:
        result = calculate_surcharges(**kwargs)
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[SurchargesTool] 计算失败: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)

SURCHARGES_TOOL = StructuredTool.from_function(
    func=_tool_calculate_surcharges,
    name="calculate_surcharges",
    description=(
        "计算附加税费（城市维护建设税 + 教育费附加 + 地方教育附加）。"
        "计税依据 = 实际缴纳的增值税 + 消费税。"
        "城建税：市区7%/县城5%/其他1%；教育费附加3%；地方教育附加2%。"
        "小规模纳税人减半征收。"
    ),
    args_schema=SurchargesInput,
    return_direct=False,
)

# ============================================================
# 6. 印花税计算 Tool
# ============================================================

class StampDutyInput(BaseModel):
    category: str = Field(
        description=(
            "税目（中文），可选：买卖合同、借款合同、融资租赁合同、租赁合同、承揽合同、"
            "建设工程合同、运输合同、保管合同、仓储合同、财产保险合同、技术合同、"
            "物业服务合同、产权转移书据、营业账簿、证券交易"
        )
    )
    contract_amount: float = Field(description="合同金额/计税依据")
    is_small_scale: bool = Field(default=False, description="是否小规模（减半）")
    is_fund_increase: bool = Field(default=False, description="营业账簿是否为资本公积增加部分")

def _tool_calculate_stamp_duty(**kwargs) -> str:
    try:
        result = calculate_stamp_duty(**kwargs)
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[StampDutyTool] 计算失败: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)

STAMP_DUTY_TOOL = StructuredTool.from_function(
    func=_tool_calculate_stamp_duty,
    name="calculate_stamp_duty",
    description=(
        "计算印花税（《印花税法》2022-07-01 施行）。"
        "常见税率：买卖合同/承揽合同/建设工程/运输/技术合同万分之三；"
        "租赁合同千分之一（最低 1 元）；借款合同万分之零点五；"
        "产权转移书据万分之五；证券交易千分之零点五（卖出方单边）。"
        "小规模减半征收。"
    ),
    args_schema=StampDutyInput,
    return_direct=False,
)

# ============================================================
# 统一导出（给后续 ReAct 模式用）
# ============================================================

ALL_TAX_TOOLS = [
    VAT_TOOL,
    CORPORATE_INCOME_TOOL,
    INDIVIDUAL_INCOME_ANNUAL_TOOL,
    MONTHLY_WITHHOLDING_TOOL,
    SURCHARGES_TOOL,
    STAMP_DUTY_TOOL,
]