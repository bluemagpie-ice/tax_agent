"""
增值税计算器
===========
支持：
  1. 一般纳税人：销项税额 - 进项税额（抵扣）
  2. 小规模纳税人：不含税销售额 × 征收率（不抵扣）
  3. 特殊税率：13% / 9% / 6% / 0% / 征收率 3% / 优惠期 1%
  4. 小规模季度 ≤ 30 万免征（2023-2027 政策）
  5. 含税 ↔ 不含税金额互转
"""

from services.calculators.base import CalcResult, CalcStep, fmt_money, fmt_pct

# ============================================================
# 税率表（2024 年现行税率）
# ============================================================

# 一般纳税人适用税率
VAT_RATES = {
    "standard": 0.13,        # 一般货物、应税劳务
    "transport": 0.09,       # 交通运输、建筑、不动产销售
    "services": 0.06,        # 现代服务业、生活服务、金融服务
    "zero": 0.00,            # 出口货物、跨境服务（退税）
}

# 小规模纳税人征收率
SMALL_SCALE_RATE = 0.03          # 法定征收率
SMALL_SCALE_PREFERRED = 0.01     # 2023.1-2027.12 减按 1% 征收

# 小规模季度免征额
SMALL_SCALE_FREE_LIMIT = 300_000  # 季度不含税 30 万（政策口径，财政部 税务总局公告 2023 年第 19 号）

# ============================================================
# 基础换算
# ============================================================

def calc_exclusive_amount(inclusive: float, rate: float) -> float:
    """含税金额 → 不含税金额：不含税 = 含税 / (1 + 税率)"""
    if rate < 0 or rate > 2:
        raise ValueError(f"税率 {rate} 超出合理范围 (0, 2)")
    return fmt_money(inclusive / (1 + rate))

def calc_inclusive_amount(exclusive: float, rate: float) -> float:
    """不含税金额 → 含税金额：含税 = 不含税 × (1 + 税率)"""
    return fmt_money(exclusive * (1 + rate))

def calc_tax_from_exclusive(exclusive: float, rate: float) -> float:
    """不含税金额计算税额：税额 = 不含税 × 税率"""
    return fmt_money(exclusive * rate)

def calc_tax_from_inclusive(inclusive: float, rate: float) -> float:
    """含税金额计算税额：税额 = 含税 / (1 + 税率) × 税率"""
    return fmt_money(inclusive / (1 + rate) * rate)

# ============================================================
# 主计算器
# ============================================================

def calculate_vat(
    *,
    taxpayer_type: str = "general",          # "general" 一般纳税人 | "small" 小规模纳税人
    sales_amount: float = 0.0,               # 销售额（默认视为不含税）
    sales_is_inclusive: bool = False,         # 销售额是否含税
    input_tax: float = 0.0,                  # 进项税额（一般纳税人抵扣用）
    vat_rate: float | None = None,           # 税率，为 None 时用默认
    use_preferred_rate: bool = True,         # 小规模是否使用优惠征收率（1%）
    quarterly_sales: float | None = None,    # 小规模季度销售额（用于判断免征）
) -> CalcResult:
    """
    计算增值税应纳税额。

    :param taxpayer_type: 纳税人类型 "general" | "small"
    :param sales_amount:  销售额
    :param sales_is_inclusive: 销售额是否含税（默认 False，即 sales_amount 视为不含税）
    :param input_tax:     进项税额（仅一般纳税人抵扣时用）
    :param vat_rate:      税率，为 None 时：一般纳税人默认 13%，小规模默认按 use_preferred_rate
    :param use_preferred_rate: 小规模是否用 1% 优惠（2023-2027 期）
    :param quarterly_sales: 小规模季度累计销售额（判断 30 万免征）
    :return: CalcResult
    """
    steps: list[CalcStep] = []
    assumptions: list[str] = []
    notes: list[str] = []

    # ---- 1. 确定税率 ----
    if vat_rate is None:
        if taxpayer_type == "small":
            vat_rate = SMALL_SCALE_PREFERRED if use_preferred_rate else SMALL_SCALE_RATE
            if use_preferred_rate:
                assumptions.append("适用 2023-2027 小规模纳税人减按 1% 征收优惠政策")
        else:
            vat_rate = VAT_RATES["standard"]
            assumptions.append("适用一般纳税人标准税率 13%（如为 9%/6% 请显式传入）")

    # ---- 2. 换算不含税销售额 ----
    if sales_is_inclusive:
        exclusive = calc_exclusive_amount(sales_amount, vat_rate)
        steps.append(CalcStep(
            label="不含税销售额换算",
            formula=f"{sales_amount} / (1 + {vat_rate})",
            input_values={"含税销售额": sales_amount, "税率": vat_rate},
            result=exclusive,
        ))
    else:
        exclusive = fmt_money(sales_amount)
        steps.append(CalcStep(
            label="确定不含税销售额",
            formula=f"{sales_amount}（已为不含税）",
            input_values={"不含税销售额": sales_amount},
            result=exclusive,
        ))

    # ---- 3. 计算销项税额（一般纳税人） / 应纳税额（小规模） ----
    if taxpayer_type == "general":
        output_tax = calc_tax_from_exclusive(exclusive, vat_rate)
        steps.append(CalcStep(
            label="销项税额",
            formula=f"{exclusive} × {vat_rate}",
            input_values={"不含税销售额": exclusive, "税率": vat_rate},
            result=output_tax,
        ))

        # 4. 抵扣进项
        net_tax = fmt_money(output_tax - input_tax)
        steps.append(CalcStep(
            label="进项税额抵扣",
            formula=f"{output_tax} - {input_tax}",
            input_values={"销项税额": output_tax, "进项税额": input_tax},
            result=net_tax,
        ))

        # 5. 应纳税额不能为负（留抵退税另算，此处简化为 0）
        if net_tax < 0:
            notes.append("进项 > 销项，产生留抵税额，本期应纳税额为 0（留抵部分可申请退税或结转下期）")
            net_tax = 0.0

        final_tax = net_tax
    else:
        # 小规模：应纳税额 = 不含税 × 征收率，不抵扣
        gross_tax = calc_tax_from_exclusive(exclusive, vat_rate)
        steps.append(CalcStep(
            label="小规模应纳税额（未考虑免征）",
            formula=f"{exclusive} × {vat_rate}",
            input_values={"不含税销售额": exclusive, "征收率": vat_rate},
            result=gross_tax,
        ))

        # 季度不含税 ≤ 30 万免征（quarterly_sales 口径跟随 sales_is_inclusive 自动处理）
        final_tax = gross_tax
        if quarterly_sales is not None:
            q_sales_exclusive = (
                calc_exclusive_amount(quarterly_sales, vat_rate)
                if sales_is_inclusive
                else quarterly_sales
            )
            if q_sales_exclusive <= SMALL_SCALE_FREE_LIMIT:
                final_tax = 0.0
                notes.append(
                    f"季度不含税销售额 {q_sales_exclusive:.2f} ≤ 免征额 {SMALL_SCALE_FREE_LIMIT}，"
                    f"免征增值税（2023-2027 政策）"
                )

    return CalcResult(
        tax_type="vat",
        title="增值税",
        tax_amount=final_tax,
        taxable_amount=exclusive,
        tax_rate=fmt_pct(vat_rate),
        steps=steps,
        assumptions=assumptions,
        notes=notes,
        policy_reference="《增值税暂行条例》；财政部 税务总局公告 2023 年第 19 号",
    )

# ============================================================
# 便捷函数
# ============================================================

def quick_vat_general(sales_inclusive: float, vat_rate: float = 0.13, input_tax: float = 0.0) -> CalcResult:
    """一般纳税人快速计算（销售额含税、税率 13% 默认）"""
    return calculate_vat(
        taxpayer_type="general",
        sales_amount=sales_inclusive,
        sales_is_inclusive=True,
        input_tax=input_tax,
        vat_rate=vat_rate,
    )

def quick_vat_small(quarterly_sales_inclusive: float) -> CalcResult:
    """小规模纳税人快速计算（季度含税销售额，自动判断免征）"""
    return calculate_vat(
        taxpayer_type="small",
        sales_amount=quarterly_sales_inclusive,
        sales_is_inclusive=True,
        quarterly_sales=quarterly_sales_inclusive,
        use_preferred_rate=True,
    )