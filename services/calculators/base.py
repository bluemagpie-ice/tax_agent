"""
计算器通用工具
=============
- 金额格式化（保留 2 位小数，财务标准）
- 计算结果数据结构（统一返回格式）
- 税率/政策变更日志标记（方便追溯用了哪个时期的政策）
"""

from dataclasses import dataclass, field, asdict
from typing import Any

def fmt_money(value: float) -> float:
    """财务金额保留 2 位小数"""
    return round(float(value), 2)

def fmt_pct(value: float) -> float:
    """百分比保留 4 位小数（0.13 表示 13%）"""
    return round(float(value), 4)

@dataclass
class CalcStep:
    """计算过程中的单一步骤，便于最终展示给用户"""
    label: str                    # 步骤说明，如 "销项税额 = 不含税销售额 × 税率"
    formula: str = ""             # 公式，如 "1000000 × 0.13"
    input_values: dict = field(default_factory=dict)   # 用到的输入值
    result: float = 0.0           # 该步骤的结果
    unit: str = "元"

@dataclass
class CalcResult:
    """所有计算器统一的返回结构"""
    tax_type: str                                 # 税种标识，如 "vat" / "corporate_income"
    title: str                                    # 展示名称，如 "增值税"
    tax_amount: float = 0.0                       # 最终应纳税额
    taxable_amount: float = 0.0                   # 计税依据
    tax_rate: float = 0.0                         # 适用税率
    steps: list[CalcStep] = field(default_factory=list)   # 分步计算过程
    assumptions: list[str] = field(default_factory=list)  # 计算假设（如"假设为一般纳税人"）
    notes: list[str] = field(default_factory=list)       # 注意事项（如"小规模季度 30 万以下免征"）
    policy_reference: str = ""                    # 政策依据简述

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict，方便 Log / LangGraph 传递"""
        data = asdict(self)
        # steps 里的 result 也要格式化
        data["tax_amount"] = fmt_money(self.tax_amount)
        data["taxable_amount"] = fmt_money(self.taxable_amount)
        return data