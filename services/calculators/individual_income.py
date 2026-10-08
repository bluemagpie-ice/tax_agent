"""
个人所得税计算器（综合所得 + 经营所得）
========================================
支持：
  1. 工资薪金（综合所得，累计预扣法 / 年度汇算清缴）
  2. 劳务报酬（预扣预缴 + 年度汇算）
  3. 稿酬、特许权使用费
  4. 经营所得（5 级超额累进）
  5. 专项附加扣除（子女教育/房贷/房租/赡养老人/继续教育/大病医疗）

累计预扣法核心：
  累计应纳税额 = (累计收入 - 累计扣除) × 预扣税率 - 速算扣除数
  本月应预扣税额 = 累计应纳税额 - 累计已预扣税额
"""

from services.calculators.base import CalcResult, CalcStep, fmt_money, fmt_pct

# ============================================================
# 税率表（年度）
# ============================================================

# 综合所得年度税率表（工资薪金 + 劳务报酬 + 稿酬 + 特许权使用费 合并）
# 含税级距（年应纳税所得额）
INCOME_TAX_TABLE = [
    (36_000,        0.03, 0),      # 不超过 3.6 万：3%，速算扣除 0
    (144_000,       0.10, 2520),   # 3.6~14.4 万：10%，速算扣除 2520
    (300_000,       0.20, 16920),  # 14.4~30 万：20%，速算扣除 16920
    (420_000,       0.25, 31920),  # 30~42 万：25%，速算扣除 31920
    (660_000,       0.30, 52920),  # 42~66 万：30%
    (960_000,       0.35, 85920),  # 66~96 万：35%
    (float("inf"),  0.45, 181920), # 超 96 万：45%
]

# 经营所得年度税率表
BUSINESS_INCOME_TAX_TABLE = [
    (30_000,     0.05, 0),
    (90_000,     0.10, 1500),
    (300_000,    0.20, 10500),
    (500_000,    0.30, 40500),
    (float("inf"), 0.35, 65500),
]

# 基本减除费用（每年固定 6 万）
ANNUAL_BASIC_DEDUCTION = 60_000
MONTHLY_BASIC_DEDUCTION = 5_000

# ============================================================
# 工具函数
# ============================================================

def apply_progressive_tax(taxable: float, table: list[tuple[float, float, float]]) -> tuple[float, float]:
    """
    超额累进税率计算。

    :param taxable: 应纳税所得额
    :param table:   [(上限, 税率, 速算扣除), ...]
    :return: (应纳税额, 适用税率)
    """
    if taxable <= 0:
        return 0.0, 0.0

    for upper, rate, quick_deduction in table:
        if taxable <= upper:
            return fmt_money(taxable * rate - quick_deduction), rate

    # 兜底
    upper, rate, quick_deduction = table[-1]
    return fmt_money(taxable * rate - quick_deduction), rate

# ============================================================
# 综合所得年度汇算清缴
# ============================================================

def calculate_individual_income_annual(
    *,
    annual_wage: float = 0.0,            # 年度工资薪金
    annual_labor: float = 0.0,           # 年度劳务报酬
    annual_royalty: float = 0.0,         # 年度稿酬
    annual_author_fee: float = 0.0,      # 年度特许权使用费
    special_deduction: float = 0.0,      # 专项扣除（社保公积金个人缴纳部分）
    special_additional_deduction: float = 0.0,  # 专项附加扣除（子女教育/房贷等）
    other_deduction: float = 0.0,        # 其他扣除（商业健康险等）
    paid_tax: float = 0.0,               # 年度已预缴税额
    is_resident: bool = True,
) -> CalcResult:
    """
    综合所得年度汇算清缴（居民个人）。

    综合所得 = 工资薪金 + 劳务报酬 × (1-20%) + 稿酬 × (1-20%) × 70% + 特许权使用费 × (1-20%)
    """
    steps: list[CalcStep] = []
    assumptions: list[str] = []
    notes: list[str] = []

    # ---- 1. 计算综合所得收入额 ----
    # 劳务报酬、稿酬、特许权使用费均按 80% 计入（减除 20% 费用）
    labor_counted = fmt_money(annual_labor * 0.8)
    royalty_counted = fmt_money(annual_royalty * 0.8 * 0.7)   # 稿酬再打 7 折
    author_counted = fmt_money(annual_author_fee * 0.8)

    total_income = fmt_money(annual_wage + labor_counted + royalty_counted + author_counted)
    steps.append(CalcStep(
        label="综合所得收入额",
        formula=f"{annual_wage}(工资) + {labor_counted}(劳务×80%) + "
                f"{royalty_counted}(稿酬×80%×70%) + {author_counted}(特许权×80%)",
        result=total_income,
    ))

    # ---- 2. 减除费用 ----
    total_deduction = fmt_money(
        ANNUAL_BASIC_DEDUCTION + special_deduction + special_additional_deduction + other_deduction
    )
    taxable = fmt_money(total_income - total_deduction)

    steps.append(CalcStep(
        label="可减除费用合计",
        formula=f"{ANNUAL_BASIC_DEDUCTION}(基本6万) + {special_deduction}(专项) + "
                f"{special_additional_deduction}(附加) + {other_deduction}(其他)",
        input_values={"合计": total_deduction},
        result=taxable,
    ))

    if taxable <= 0:
        notes.append("综合所得减除全部扣除后 ≤ 0，年度应纳税额为 0")
        final_tax = 0.0
        diff = fmt_money(paid_tax - 0)  # 多交了可退
    else:
        steps.append(CalcStep(
            label="应纳税所得额",
            formula=f"{total_income} - {total_deduction}",
            input_values={"综合所得": total_income, "扣除合计": total_deduction},
            result=taxable,
        ))

        # ---- 3. 用超额累进税率算 ----
        gross_tax, applicable_rate = apply_progressive_tax(taxable, INCOME_TAX_TABLE)
        steps.append(CalcStep(
            label="年度应纳税额（综合所得累进税率）",
            formula=f"{taxable} × 税率 - 速算扣除数",
            input_values={"应纳税所得额": taxable},
            result=gross_tax,
        ))

        final_tax = gross_tax
        diff = fmt_money(final_tax - paid_tax)

    assumptions.append("按居民个人综合所得年度汇算清缴计算")

    return CalcResult(
        tax_type="individual_income",
        title="个人所得税（综合所得年度汇算）",
        tax_amount=final_tax,
        taxable_amount=max(0, taxable),
        steps=steps,
        assumptions=assumptions,
        notes=notes + [
            f"年度已预缴税额: {paid_tax:.2f}",
            f"汇算后应补退税额: {diff:.2f}（正数=应补，负数=应退）",
        ],
        policy_reference="《个人所得税法》；国务院令第 707 号（实施条例）",
    )

# ============================================================
# 工资薪金月度累计预扣法
# ============================================================

def calculate_monthly_wage_withholding(
    *,
    monthly_wage: float,
    month_index: int = 1,                   # 第几月（1-12）
    special_deduction_monthly: float = 0.0, # 社保公积金月度个人部分
    special_additional_monthly: float = 0.0,# 专项附加扣除月度
    prior_tax_withheld: float = 0.0,       # 截至上月累计已预扣税额
    prior_income: float = 0.0,             # 截至上月累计工资收入
) -> CalcResult:
    """
    工资薪金月度预扣（累计预扣法）。

    累计应纳税额 = (累计收入 - 累计扣除) × 月度预扣税率表 - 累计速算扣除数(月度表)
    本月预扣 = 累计应纳税额 - 累计已预扣
    """
    steps: list[CalcStep] = []

    # 累计数
    cumulative_income = prior_income + monthly_wage
    cumulative_basic = MONTHLY_BASIC_DEDUCTION * month_index
    cumulative_special = special_deduction_monthly * month_index
    cumulative_additional = special_additional_monthly * month_index
    cumulative_deduction = cumulative_basic + cumulative_special + cumulative_additional

    taxable_cumulative = fmt_money(cumulative_income - cumulative_deduction)

    steps.append(CalcStep(
        label=f"累计应纳税所得额（第 {month_index} 月）",
        formula=f"{cumulative_income}(累计收入) - {cumulative_deduction}(累计扣除)",
        input_values={
            "累计收入": cumulative_income,
            "累计基本减除": cumulative_basic,
            "累计专项": cumulative_special,
            "累计附加": cumulative_additional,
        },
        result=max(0, taxable_cumulative),
    ))

    # 用年度税率表算累计应纳税额（税法规定：月度预扣按年度税率表，不按月度表了，2019 后）
    if taxable_cumulative <= 0:
        cumulative_tax = 0.0
    else:
        cumulative_tax, rate = apply_progressive_tax(taxable_cumulative, INCOME_TAX_TABLE)

    monthly_withheld = fmt_money(cumulative_tax - prior_tax_withheld)
    if monthly_withheld < 0:
        monthly_withheld = 0.0  # 累计不足扣减时，本月预扣为 0（退税要等年度汇算）

    steps.append(CalcStep(
        label=f"本月应预扣税额（第 {month_index} 月）",
        formula=f"{cumulative_tax}(累计应纳税额) - {prior_tax_withheld}(累计已预扣)",
        input_values={"累计应纳税额": cumulative_tax, "累计已预扣": prior_tax_withheld},
        result=monthly_withheld,
    ))

    return CalcResult(
        tax_type="individual_income",
        title=f"个人所得税（工资薪金第 {month_index} 月预扣）",
        tax_amount=monthly_withheld,
        taxable_amount=max(0, taxable_cumulative),
        steps=steps,
        assumptions=["按累计预扣法计算"],
        policy_reference="国家税务总局公告 2018 年第 61 号",
    )

# ============================================================
# 经营所得
# ============================================================

def calculate_business_income_tax(
    *,
    annual_profit: float,                # 年度经营利润
    special_additional_deduction: float = 0.0,  # 专项附加扣除（可在经营所得里扣）
) -> CalcResult:
    """
    个体工商户 / 个人独资企业 / 合伙企业 经营所得个税。
    查账征收：应纳税所得额 × 5 级累进税率 - 速算扣除数。
    """
    steps: list[CalcStep] = []
    assumptions: list[str] = []

    taxable = fmt_money(annual_profit - special_additional_deduction)
    steps.append(CalcStep(
        label="应纳税所得额",
        formula=f"{annual_profit}(利润) - {special_additional_deduction}(专项附加扣除)",
        result=taxable,
    ))

    if taxable <= 0:
        return CalcResult(
            tax_type="individual_income_business",
            title="个人所得税（经营所得）",
            tax_amount=0.0,
            taxable_amount=0.0,
            steps=steps,
            assumptions=assumptions + ["按查账征收方式计算"],
        )

    tax, rate = apply_progressive_tax(taxable, BUSINESS_INCOME_TAX_TABLE)
    steps.append(CalcStep(
        label="经营所得应纳税额",
        formula=f"{taxable} × 税率 {rate} - 速算扣除数",
        result=tax,
    ))

    return CalcResult(
        tax_type="individual_income_business",
        title="个人所得税（经营所得）",
        tax_amount=tax,
        taxable_amount=taxable,
        tax_rate=fmt_pct(rate),
        steps=steps,
        assumptions=assumptions + ["按查账征收方式计算（5 级超额累进税率）"],
        policy_reference="《个人所得税法》经营所得税率表",
    )