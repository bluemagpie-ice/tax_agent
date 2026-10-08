"""
印花税计算器
===========
《印花税法》2022 年 7 月 1 日施行。

常见税目税率（简化版）：
  - 合同类：
      买卖合同   0.03%（万分之三）
      借款合同   0.005%（万分之零点五，不含同业拆借）
      融资租赁合同 0.005%
      租赁合同   0.1%（千分之一，据实计税）
      承揽合同   0.03%
      建设工程合同 0.03%
      运输合同   0.03%
      保管合同   0.1%
      仓储合同   0.1%
      财产保险合同 0.1%（按保费，不含人身险）
      技术合同   0.03%
      物业服务合同 0.01%（千分之零点一，部分地区）
  - 产权转移书据：0.05%（万分之五）
  - 营业账簿：   0.25‰（原 0.5‰减半，2023.8.28 后取消？注意政策更新）
  - 证券交易：   0.05%（千分之零点五，卖出方单边）
  - 权利许可证照：免征（原按件，已取消）

小规模/小微企业减半征收（2023-2027）
"""

from services.calculators.base import CalcResult, CalcStep, fmt_money, fmt_pct

STAMP_DUTY_RATES = {
    # 合同类
    "买卖合同": 0.0003,
    "借款合同": 0.00005,
    "融资租赁合同": 0.00005,
    "租赁合同": 0.001,
    "承揽合同": 0.0003,
    "建设工程合同": 0.0003,
    "运输合同": 0.0003,
    "保管合同": 0.001,
    "仓储合同": 0.001,
    "财产保险合同": 0.001,
    "技术合同": 0.0003,
    "物业服务合同": 0.0001,
    # 书据
    "产权转移书据": 0.0005,
    # 账簿
    "营业账簿": 0.00025,
    # 证券
    "证券交易": 0.0005,
    # 证照（已取消）
    "权利许可证照": 0.0,
}

def calculate_stamp_duty(
    *,
    category: str,                     # 税目，必须是 STAMP_DUTY_RATES 的 key
    contract_amount: float,            # 合同金额/计税依据
    is_small_scale: bool = False,      # 小规模减半
    is_fund_increase: bool = False,    # 营业账簿是否为资本公积增加部分
) -> CalcResult:
    """
    计算单笔合同/事项的印花税。
    """
    steps: list[CalcStep] = []
    assumptions: list[str] = []
    notes: list[str] = []

    if category not in STAMP_DUTY_RATES:
        # 尝试模糊匹配
        notes.append(f" 未识别的税目「{category}」，请确认是否拼写正确。"
                      f"支持的税目：{list(STAMP_DUTY_RATES.keys())}")
        return CalcResult(
            tax_type="stamp_duty",
            title="印花税（计算失败）",
            tax_amount=0.0,
            notes=notes,
        )

    rate = STAMP_DUTY_RATES[category]
    halve = 0.5 if is_small_scale else 1.0
    if is_small_scale:
        assumptions.append("小规模纳税人印花税减半（2023-2027）")

    raw_tax = fmt_money(contract_amount * rate * halve)

    # 营业账簿：自 2023.8.28 起，记载资金的营业账簿按实收资本 + 资本公积的增加部分计税；
    # 已贴花的增加部分才征税（避免重复征税）
    if category == "营业账簿" and is_fund_increase:
        notes.append("营业账簿按增加的实收资本+资本公积部分计税")

    steps.append(CalcStep(
        label=f"印花税（{category}，税率 {rate}）",
        formula=f"{contract_amount} × {rate} × {halve}",
        input_values={"合同金额": contract_amount, "税率": rate},
        result=raw_tax,
    ))

    return CalcResult(
        tax_type="stamp_duty",
        title=f"印花税（{category}）",
        tax_amount=raw_tax,
        taxable_amount=contract_amount,
        tax_rate=fmt_pct(rate),
        steps=steps,
        assumptions=assumptions,
        notes=notes + [
            "2022-07-01 起《印花税法》实施，取消了营业账簿按件贴花、权利许可证照贴花等项目",
        ],
        policy_reference="《中华人民共和国印花税法》（2022-07-01 施行）",
    )