"""
企业所得税计算器
===============
支持：
  1. 基本公式：应纳税所得额 × 税率 - 减免税额
  2. 小型微利企业优惠（资格门槛：应纳税所得额 ≤ 300 万）
     - 300 万以内统一实际税负 5%（减按 25% 计入，按 20% 税率，2023-2027 一档）
     - 超过 300 万 → 整体丧失小微资格，全额按 25% 计税
  3. 高新技术企业优惠税率 15%
  4. 研发费用加计扣除（100%，科技型中小企业 2023 起）
  5. 亏损结转（最长 5 年，高新/科技型中小企业 10 年）
"""

from services.calculators.base import CalcResult, CalcStep, fmt_money, fmt_pct

# 企业类型
NORMAL = "normal"                   # 普通企业
SMALL_LOW_PROFIT = "small_low_profit"   # 小型微利企业
HIGH_TECH = "high_tech"             # 高新技术企业

def calculate_corporate_income_tax(
    *,
    tax_type: str = NORMAL,               # 企业类型
    taxable_income: float = 0.0,          # 应纳税所得额（已扣除成本费用后的余额）
    prior_loss: float = 0.0,               # 以前年度可弥补亏损
    rd_expense: float = 0.0,              # 研发费用（用于加计扣除）
    rd_expense_deduct_ratio: float = 1.0, # 加计扣除比例（1.0 = 100%）
    direct_deduction: float = 0.0,        # 直接减免税额（如"三免三减半"等）
    income_threshold_high: float = 3_000_000,     # 小微资格门槛（应纳税所得额 ≤ 300 万）
    small_low_profit_tax_rate: float = 0.25 * 0.2,  # 25% × 20% = 5%（2023 起统一一档）
    # --- 以下为向后兼容保留的旧参数，不再参与计算 ---
    income_threshold_low: float = 1_000_000,
    small_low_profit_upper_tax_rate: float = 0.25 * 0.4,
    small_low_profit_lower_tax_rate: float = 0.25 * 0.2,
    normal_tax_rate: float = 0.25,
    high_tech_tax_rate: float = 0.15,
) -> CalcResult:
    """
    计算企业所得税。

    :param tax_type: 企业类型
    :param taxable_income: 应纳税所得额
    :param prior_loss: 以前年度可弥补亏损
    :param rd_expense: 研发费用（会触发加计扣除）
    :param rd_expense_deduct_ratio: 加计扣除比例
    :param direct_deduction: 直接减免税额
    """
    steps: list[CalcStep] = []
    assumptions: list[str] = []
    notes: list[str] = []

    # ---- 1. 研发费用加计扣除 ----
    actual_income = taxable_income
    if rd_expense > 0:
        rd_deduction = fmt_money(rd_expense * rd_expense_deduct_ratio)
        actual_income = fmt_money(taxable_income - rd_deduction)
        steps.append(CalcStep(
            label="研发费用加计扣除",
            formula=f"{rd_expense} × {rd_expense_deduct_ratio} = {rd_deduction}",
            input_values={"研发费用": rd_expense, "加计比例": rd_expense_deduct_ratio},
            result=actual_income,
        ))

    # ---- 2. 弥补以前年度亏损 ----
    if prior_loss > 0 and actual_income > 0:
        offset = min(prior_loss, actual_income)
        actual_income = fmt_money(actual_income - offset)
        assumptions.append(f"用以前年度亏损 {offset:.2f} 弥补本年利润")
        steps.append(CalcStep(
            label="弥补以前年度亏损",
            formula=f"{actual_income + offset} - {offset}",
            input_values={"弥补前利润": actual_income + offset, "可弥补亏损": offset},
            result=actual_income,
        ))

    if actual_income <= 0:
        notes.append("应纳税所得额 ≤ 0，本年度免征企业所得税（亏损可结转以后年度弥补）")
        return CalcResult(
            tax_type="corporate_income",
            title="企业所得税",
            tax_amount=0.0,
            taxable_amount=actual_income,
            steps=steps,
            assumptions=assumptions,
            notes=notes,
        )

    # ---- 3. 计算应纳税额 ----
    gross_tax = 0.0

    if tax_type == SMALL_LOW_PROFIT:
        # ---- 关键：小微优惠是「资格门槛」，不是分段累进税率 ----
        # 超过 300 万 → 整体丧失小微资格，全额按 25% 法定税率计税
        if actual_income > income_threshold_high:
            notes.append(
                f"应纳税所得额 {actual_income:.2f} > 资格门槛 {income_threshold_high:.0f} 元（300 万），"
                f"已丧失小型微利企业资格，全额按法定税率 {normal_tax_rate} 计税"
            )
            gross_tax = fmt_money(actual_income * normal_tax_rate)
            steps.append(CalcStep(
                label="丧失小微资格，全额计税",
                formula=f"{actual_income} × {normal_tax_rate}（法定税率 25%）",
                input_values={"应纳税所得额": actual_income, "税率": normal_tax_rate},
                result=gross_tax,
            ))
        else:
            assumptions.append("按小型微利企业优惠政策计算（2023-2027 统一一档 5%）")
            gross_tax = fmt_money(actual_income * small_low_profit_tax_rate)
            steps.append(CalcStep(
                label="小型微利企业优惠（一档 5%）",
                formula=f"{actual_income} × {small_low_profit_tax_rate}（减按 25% × 20%）",
                input_values={"应纳税所得额": actual_income, "实际税率": small_low_profit_tax_rate},
                result=gross_tax,
            ))

    elif tax_type == HIGH_TECH:
        assumptions.append(f"高新技术企业优惠税率 {high_tech_tax_rate}")
        gross_tax = fmt_money(actual_income * high_tech_tax_rate)
        steps.append(CalcStep(
            label="高新技术企业优惠税率",
            formula=f"{actual_income} × {high_tech_tax_rate}",
            input_values={"应纳税所得额": actual_income, "税率": high_tech_tax_rate},
            result=gross_tax,
        ))

    else:
        gross_tax = fmt_money(actual_income * normal_tax_rate)
        steps.append(CalcStep(
            label="法定税率计算",
            formula=f"{actual_income} × {normal_tax_rate}",
            input_values={"应纳税所得额": actual_income, "税率": normal_tax_rate},
            result=gross_tax,
        ))

    # ---- 4. 减免税额 ----
    final_tax = fmt_money(max(0, gross_tax - direct_deduction))
    if direct_deduction > 0:
        steps.append(CalcStep(
            label="减免税额",
            formula=f"{gross_tax} - {direct_deduction}",
            input_values={"减免前税额": gross_tax, "减免额": direct_deduction},
            result=final_tax,
        ))

    # 确定实际适用税率（用于展示；小微资格丧失时应反映 25%）
    if tax_type == NORMAL or (
        tax_type == SMALL_LOW_PROFIT and actual_income > income_threshold_high
    ):
        display_rate = normal_tax_rate
    elif tax_type == HIGH_TECH:
        display_rate = high_tech_tax_rate
    else:
        display_rate = small_low_profit_tax_rate

    return CalcResult(
        tax_type="corporate_income",
        title="企业所得税",
        tax_amount=final_tax,
        taxable_amount=actual_income,
        tax_rate=fmt_pct(display_rate),
        steps=steps,
        assumptions=assumptions,
        notes=notes,
        policy_reference="《企业所得税法》；财政部 税务总局公告 2023 年第 12 号（小微优惠）",
    )