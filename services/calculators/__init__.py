"""
税务计算器包
============
两种调用方式：
  1. 纯函数直接调用（推荐起步）
     from services.calculators import calculate_vat, calculate_corporate_income_tax, ...

  2. LangChain Tool（给 LLM ReAct 模式用）
     from services.calculators.tools import ALL_TAX_TOOLS
"""

from services.calculators.base import (
    CalcStep,
    CalcResult,
    fmt_money,
    fmt_pct,
)

from services.calculators.vat import (
    calculate_vat,
    quick_vat_general,
    quick_vat_small,
    calc_exclusive_amount,
    calc_inclusive_amount,
    calc_tax_from_exclusive,
    calc_tax_from_inclusive,
)

from services.calculators.corporate_income import (
    calculate_corporate_income_tax,
    NORMAL as CORP_NORMAL,
    SMALL_LOW_PROFIT as CORP_SMALL_LOW_PROFIT,
    HIGH_TECH as CORP_HIGH_TECH,
)

from services.calculators.individual_income import (
    calculate_individual_income_annual,
    calculate_monthly_wage_withholding,
    calculate_business_income_tax,
)

from services.calculators.surcharges import (
    calculate_surcharges,
    URBAN as CITY_URBAN,
    COUNTY as CITY_COUNTY,
    OTHER as CITY_OTHER,
)

from services.calculators.stamp_duty import (
    calculate_stamp_duty,
    STAMP_DUTY_RATES,
)

# Tool 层（可选，需安装 langchain-core + pydantic）
try:
    from services.calculators.tools import ALL_TAX_TOOLS
    __all__ = [
        "CalcStep", "CalcResult", "fmt_money", "fmt_pct",
        "calculate_vat", "quick_vat_general", "quick_vat_small",
        "calc_exclusive_amount", "calc_inclusive_amount",
        "calc_tax_from_exclusive", "calc_tax_from_inclusive",
        "calculate_corporate_income_tax",
        "CORP_NORMAL", "CORP_SMALL_LOW_PROFIT", "CORP_HIGH_TECH",
        "calculate_individual_income_annual",
        "calculate_monthly_wage_withholding",
        "calculate_business_income_tax",
        "calculate_surcharges",
        "CITY_URBAN", "CITY_COUNTY", "CITY_OTHER",
        "calculate_stamp_duty", "STAMP_DUTY_RATES",
        "ALL_TAX_TOOLS",
    ]
except ImportError:
    __all__ = [
        "CalcStep", "CalcResult", "fmt_money", "fmt_pct",
        "calculate_vat", "quick_vat_general", "quick_vat_small",
        "calc_exclusive_amount", "calc_inclusive_amount",
        "calc_tax_from_exclusive", "calc_tax_from_inclusive",
        "calculate_corporate_income_tax",
        "CORP_NORMAL", "CORP_SMALL_LOW_PROFIT", "CORP_HIGH_TECH",
        "calculate_individual_income_annual",
        "calculate_monthly_wage_withholding",
        "calculate_business_income_tax",
        "calculate_surcharges",
        "CITY_URBAN", "CITY_COUNTY", "CITY_OTHER",
        "calculate_stamp_duty", "STAMP_DUTY_RATES",
    ]