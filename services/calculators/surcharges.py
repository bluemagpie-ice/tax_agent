"""
附加税费计算器：城市维护建设税 + 教育费附加 + 地方教育附加
=========================================================
公式：附加税费 = 实际缴纳的增值税（+ 消费税）× 各附加税率

城建税税率（按所在地）：
  - 市区（含县城、建制镇）：7%
  - 县城/建制镇：5%
  - 其他地区：1%

教育费附加：3%（固定）
地方教育附加：2%（固定，大部分省）

小规模纳税人减半征收（2023-2027 财政部 税务总局公告 2023 年第 12 号）
"""

from services.calculators.base import CalcResult, CalcStep, fmt_money, fmt_pct

# 城建税区域划分
URBAN = "urban"          # 市区/县城/建制镇
COUNTY = "county"        # 县城/建制镇（历史上区分，现在基本合并）
OTHER = "other"          # 不在市区/县城/建制镇的

CITY_TAX_RATES = {
    URBAN: 0.07,
    COUNTY: 0.05,
    OTHER: 0.01,
}

EDUCATION_SURCHARGE = 0.03
LOCAL_EDUCATION_SURCHARGE = 0.02


def calculate_surcharges(
    *,
    vat_paid: float = 0.0,             # 实际缴纳的增值税
    consumption_tax_paid: float = 0.0, # 实际缴纳的消费税（可选）
    city_type: str = URBAN,            # 城建税税率档次
    is_small_scale: bool = False,      # 是否小规模（减半征收）
    extra_local_rate: float = 0.0,     # 地方附加附加率（部分地区有特殊政策，如 0.5%）
) -> CalcResult:
    """
    计算三项附加税费。

    计税依据 = 实际缴纳的增值税 + 消费税（进口不征、出口不退）
    """
    steps: list[CalcStep] = []
    assumptions: list[str] = []
    notes: list[str] = []

    # ---- 1. 计税依据 ----
    base_tax = fmt_money(vat_paid + consumption_tax_paid)
    steps.append(CalcStep(
        label="附加税计税依据",
        formula=f"{vat_paid}(增值税) + {consumption_tax_paid}(消费税)",
        result=base_tax,
    ))

    if base_tax <= 0:
        return CalcResult(
            tax_type="surcharges",
            title="附加税费（城市维护建设税 + 教育费附加 + 地方教育附加）",
            tax_amount=0.0,
            taxable_amount=0.0,
            steps=steps,
            assumptions=assumptions + ["计税依据 ≤ 0，附加税费免征"],
        )

    # ---- 2. 各附加税率 ----
    city_rate = CITY_TAX_RATES.get(city_type, CITY_TAX_RATES[URBAN])

    # 小规模减半
    halve = 0.5 if is_small_scale else 1.0
    if is_small_scale:
        assumptions.append("小规模纳税人附加税费减半征收（2023-2027）")

    city_tax = fmt_money(base_tax * city_rate * halve)
    edu_surcharge = fmt_money(base_tax * EDUCATION_SURCHARGE * halve)
    local_edu = fmt_money(base_tax * LOCAL_EDUCATION_SURCHARGE * halve)
    extra_local = fmt_money(base_tax * extra_local_rate * halve)

    steps.append(CalcStep(
        label=f"城市维护建设税（税率 {city_rate}）",
        formula=f"{base_tax} × {city_rate} × {halve}",
        input_values={"税率": city_rate},
        result=city_tax,
    ))

    steps.append(CalcStep(
        label="教育费附加（3%）",
        formula=f"{base_tax} × {EDUCATION_SURCHARGE} × {halve}",
        result=edu_surcharge,
    ))

    steps.append(CalcStep(
        label="地方教育附加（2%）",
        formula=f"{base_tax} × {LOCAL_EDUCATION_SURCHARGE} × {halve}",
        result=local_edu,
    ))

    if extra_local > 0:
        steps.append(CalcStep(
            label=f"地方额外附加（{extra_local_rate}）",
            formula=f"{base_tax} × {extra_local_rate} × {halve}",
            result=extra_local,
        ))

    total = fmt_money(city_tax + edu_surcharge + local_edu + extra_local)

    return CalcResult(
        tax_type="surcharges",
        title="附加税费合计",
        tax_amount=total,
        taxable_amount=base_tax,
        steps=steps,
        assumptions=assumptions + [
            f"城建税适用税率 {city_rate}（{'市区' if city_type == URBAN else city_type}）",
        ],
        notes=notes + [
            "进口货物/服务不征附加税；出口退税不退已缴附加税",
            "两项附加费可申请退（增值税退税时一并退），城建税不退",
        ],
        policy_reference="《城市维护建设税暂行条例》；国发〔1986〕50 号；财综〔2010〕98 号",
    )