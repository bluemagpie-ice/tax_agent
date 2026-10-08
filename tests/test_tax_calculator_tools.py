"""
tests/test_tax_calculator_tools.py
===================================
测试 services/calculators/tools.py 中的 LangChain StructuredTool 封装。
同时也测试底层纯函数（因为 Tool 只是薄包装）。

所有计算都是确定性纯函数 → 可以精确断言，无 mock 无网络。
"""
import json
import pytest

# ---- 纯函数（直接测计算逻辑）----
from services.calculators import (
    calculate_vat,
    calculate_corporate_income_tax,
    calculate_individual_income_annual,
    calculate_monthly_wage_withholding,
    calculate_business_income_tax,
    calculate_surcharges,
    calculate_stamp_duty,
    CORP_NORMAL,
    CORP_SMALL_LOW_PROFIT,
    CORP_HIGH_TECH,
    STAMP_DUTY_RATES,
)

from services.calculators.surcharges import URBAN

# ---- LangChain Tool 封装 ----
# 名称对齐 services/calculators/tools.py 实际导出：
#   INDIVIDUAL_INCOME_ANNUAL_TOOL（不是 INDIVIDUAL_INCOME_TOOL）
#   ALL_TAX_TOOLS（不是 ALL_CALC_TOOLS）
try:
    from services.calculators.tools import (
        VAT_TOOL,
        CORPORATE_INCOME_TOOL,
        INDIVIDUAL_INCOME_ANNUAL_TOOL,
        MONTHLY_WITHHOLDING_TOOL,
        SURCHARGES_TOOL,
        STAMP_DUTY_TOOL,
        ALL_TAX_TOOLS,
    )
    HAS_TOOLS = True
except ImportError:
    HAS_TOOLS = False

# ============================================================
# 1. 纯函数测试 — 增值税
# ============================================================

class TestVATCalculation:
    def test_general_taxpayer_basic(self):
        """一般纳税人：销售额 100 万不含税，进项 5 万，税率 13%"""
        result = calculate_vat(
            taxpayer_type="general",
            sales_amount=1_000_000,
            input_tax=50_000,
            vat_rate=0.13,
        )
        # 销项 = 100万 × 13% = 13万，应纳税 = 13万 - 5万 = 8万
        assert result.tax_amount == pytest.approx(80_000.0)
        assert result.taxable_amount == pytest.approx(1_000_000.0)

    def test_small_taxpayer_quarterly_exempt(self):
        """小规模纳税人季度 28 万 < 30 万免征额，应纳税为 0"""
        result = calculate_vat(
            taxpayer_type="small",
            sales_amount=280_000,          # 默认视为不含税
            vat_rate=0.01,
            quarterly_sales=280_000,
        )
        assert result.tax_amount == 0.0

    def test_small_taxpayer_quarterly_exceed(self):
        """
        小规模纳税人季度 40 万 > 30 万，全额征收。
        sales_amount 默认视为不含税（vat.py 第 61 行注释），所以直接 × 1% = 4000。
        如果要含税价税分离，必须显式传 sales_is_inclusive=True。
        """
        result = calculate_vat(
            taxpayer_type="small",
            sales_amount=400_000,           # 不含税
            vat_rate=0.01,
            quarterly_sales=400_000,
        )
        # 不含税 × 1% = 4000（不是价税分离的 3960）
        assert result.tax_amount == pytest.approx(4_000.0)

    def test_small_taxpayer_with_inclusive_amount(self):
        """
        小规模 + 含税销售额 → sales_is_inclusive=True 后，quarterly_sales 也自动按含税口径处理。
        含税 40 万 → 不含税 ≈ 396039.60 > 30 万，不免征。
        """
        result = calculate_vat(
            taxpayer_type="small",
            sales_amount=400_000,           # 含税
            sales_is_inclusive=True,
            vat_rate=0.01,
            quarterly_sales=400_000,        # 同样被视为含税（跟随 sales_is_inclusive）
        )
        # 不含税 = 40万 / 1.01 ≈ 396039.60，应纳税 = 396039.60 × 1% ≈ 3960.40
        assert result.tax_amount == pytest.approx(400_000 / 1.01 * 0.01, rel=1e-4)

    def test_small_taxpayer_inclusive_quarterly_just_under_30w(self):
        """
        关键修复验证：含税季度销售额 302,900 元，征收率 1%。
        不含税 = 302,900 / 1.01 ≈ 299,901 < 300,000 → 应该免征。
        ❌ 旧代码把 302,900 直接和 300,000 比较 → 302,900 > 300,000 → 不免征（错误）
        """
        result = calculate_vat(
            taxpayer_type="small",
            sales_amount=100_000,
            sales_is_inclusive=True,
            vat_rate=0.01,
            quarterly_sales=302_900,          # 含税季度销售额
        )
        assert result.tax_amount == 0.0
        assert any("免征" in n for n in result.notes)

    def test_small_taxpayer_inclusive_quarterly_just_over_30w(self):
        """
        含税季度销售额 303,100 元，征收率 1%。
        不含税 = 303,100 / 1.01 ≈ 300,099 > 300,000 → 不免征。
        """
        result = calculate_vat(
            taxpayer_type="small",
            sales_amount=100_000,
            sales_is_inclusive=True,
            vat_rate=0.01,
            quarterly_sales=303_100,
        )
        assert result.tax_amount > 0.0

    def test_inclusive_amount_conversion(self):
        """含税销售额需要先价税分离"""
        result = calculate_vat(
            taxpayer_type="general",
            sales_amount=113_000,
            sales_is_inclusive=True,
            vat_rate=0.13,
        )
        # 不含税 = 113000 / 1.13 = 100000，应纳税 = 100000 × 13% = 13000
        assert result.taxable_amount == pytest.approx(100_000.0)
        assert result.tax_amount == pytest.approx(13_000.0)

# ============================================================
# 2. 纯函数测试 — 企业所得税
# ============================================================

class TestCorporateIncomeTax:
    def test_normal_25_percent(self):
        """普通企业 25% 税率"""
        result = calculate_corporate_income_tax(
            tax_type=CORP_NORMAL,               # ← 正确参数名：tax_type
            taxable_income=1_000_000,            # ← 正确参数名：taxable_income
        )
        assert result.tax_amount == pytest.approx(250_000.0)

    def test_small_low_profit_200k(self):
        """
        小型微利企业 200 万应纳税所得额（2023 起一档统一 5%）：
          200 万 × 5% = 10 万
        ❌ 旧两档政策（2019-2022）会算出：100万×5% + 100万×10% = 15万
        """
        result = calculate_corporate_income_tax(
            tax_type=CORP_SMALL_LOW_PROFIT,
            taxable_income=2_000_000,
        )
        assert result.tax_amount == pytest.approx(2_000_000 * 0.05)

    def test_high_tech_15_percent(self):
        """高新技术企业 15% 税率"""
        result = calculate_corporate_income_tax(
            tax_type=CORP_HIGH_TECH,
            taxable_income=1_000_000,
        )
        assert result.tax_amount == pytest.approx(150_000.0)

    def test_small_low_profit_300w_boundary_in(self):
        """
        恰好 300 万（资格上限，仍在资格内）→ 一档统一 5%：
          300 万 × 5% = 15 万
        """
        result = calculate_corporate_income_tax(
            tax_type=CORP_SMALL_LOW_PROFIT,
            taxable_income=3_000_000,
        )
        assert result.tax_amount == pytest.approx(3_000_000 * 0.05)

    def test_small_low_profit_400w_lost_qualification(self):
        """
        关键测试：400 万 > 300 万门槛 → 丧失小微资格，全额按 25% 计税。
          400万 × 25% = 100万
        ❌ 错误旧逻辑会算出：25万(前300万分段) + 100万(超出100万×25%) = 125万
        ✅ 正确应该是：100万
        """
        result = calculate_corporate_income_tax(
            tax_type=CORP_SMALL_LOW_PROFIT,
            taxable_income=4_000_000,
        )
        # 全额 25%，不是分段叠加
        assert result.tax_amount == pytest.approx(4_000_000 * 0.25)
        # notes 中应提示资格丧失
        assert any("丧失" in n or "25%" in n for n in result.notes)

    def test_small_low_profit_just_over_300w(self):
        """
        301 万（刚越过门槛）→ 丧失资格，全额 25%。
          301万 × 25% = 75.25万
        用来验证边界不是「渐进」而是「突变」。
        """
        result = calculate_corporate_income_tax(
            tax_type=CORP_SMALL_LOW_PROFIT,
            taxable_income=3_010_000,
        )
        assert result.tax_amount == pytest.approx(3_010_000 * 0.25)

    def test_loss_no_tax(self):
        """应纳税所得额 ≤ 0 → 免征"""
        result = calculate_corporate_income_tax(
            tax_type=CORP_NORMAL,
            taxable_income=-500_000,
        )
        assert result.tax_amount == 0.0

# ============================================================
# 3. 纯函数测试 — 印花税 / 附加税
# ============================================================

class TestOtherTaxes:
    def test_stamp_duty_sales_contract(self):
        """买卖合同 0.03% → 100 万合同 = 300 元"""
        result = calculate_stamp_duty(
            category="买卖合同",              # ← 正确参数名：category（不是 duty_type）
            contract_amount=1_000_000,
        )
        assert result.tax_amount == pytest.approx(300.0)

    def test_stamp_duty_rates_table(self):
        """验证税目表中买卖合同税率确实是 0.03%"""
        assert STAMP_DUTY_RATES["买卖合同"] == 0.0003

    def test_surcharges_urban(self):
        """
        市区：城建税 7% + 教育费附加 3% + 地方教育附加 2% = 12%
        增值税 10 万 → 附加税 1.2 万
        """
        result = calculate_surcharges(
            vat_paid=100_000,                 # ← 正确参数名：vat_paid（不是 vat_payable）
            city_type=URBAN,                  # ← 正确参数名：city_type（不是 city_level）
        )
        assert result.tax_amount == pytest.approx(12_000.0)

    def test_surcharges_small_scale_half(self):
        """小规模纳税人附加税减半"""
        result = calculate_surcharges(
            vat_paid=100_000,
            city_type=URBAN,
            is_small_scale=True,
        )
        # 10万 × 12% × 50% = 6000
        assert result.tax_amount == pytest.approx(6_000.0)

    def test_surcharges_zero_base(self):
        """增值税为 0 → 附加税也为 0"""
        result = calculate_surcharges(vat_paid=0, city_type=URBAN)
        assert result.tax_amount == 0.0

# ============================================================
# 4. LangChain Tool 封装测试（如果可用）
# ============================================================

@pytest.mark.skipif(not HAS_TOOLS, reason="calculators.tools 未加载")
class TestCalculatorTools:
    def test_tools_exist(self):
        """6 个 Tool + 统一导出列表都必须存在"""
        assert VAT_TOOL is not None
        assert CORPORATE_INCOME_TOOL is not None
        assert INDIVIDUAL_INCOME_ANNUAL_TOOL is not None
        assert MONTHLY_WITHHOLDING_TOOL is not None
        assert SURCHARGES_TOOL is not None
        assert STAMP_DUTY_TOOL is not None
        assert isinstance(ALL_TAX_TOOLS, list)
        assert len(ALL_TAX_TOOLS) == 6

    def test_vat_tool_structure(self):
        """Tool 必须有 name / description / args_schema"""
        assert hasattr(VAT_TOOL, "name")
        assert VAT_TOOL.args_schema is not None

    def test_vat_tool_invoke(self):
        """Tool.invoke 返回 JSON 字符串，且能正确算出 8 万"""
        out = VAT_TOOL.invoke({
            "taxpayer_type": "general",
            "sales_amount": 1_000_000,
            "input_tax": 50_000,
            "vat_rate": 0.13,
        })
        data = json.loads(out)
        assert "tax_amount" in data
        assert data["tax_amount"] == pytest.approx(80_000.0)

    def test_corporate_income_tool_invoke(self):
        """企业所得税 Tool：普通企业 100 万应纳税所得额 → 25 万"""
        out = CORPORATE_INCOME_TOOL.invoke({
            "tax_type": "normal",
            "taxable_income": 1_000_000,
        })
        data = json.loads(out)
        assert "tax_amount" in data
        assert data["tax_amount"] == pytest.approx(250_000.0)

# ============================================================
# 5. 纯函数测试 — 个人所得税
# ============================================================

class TestIndividualIncomeAnnual:
    """综合所得年度汇算清缴"""

    def test_wage_only_basic(self):
        """仅工资 20 万，无扣除 → 应纳税 ≈ 11480"""
        result = calculate_individual_income_annual(annual_wage=200_000)
        # taxable = 200000 - 60000 = 140000, rate 10%, quick 2520
        expected = 140_000 * 0.10 - 2_520
        assert result.tax_amount == pytest.approx(expected)
        assert result.taxable_amount == pytest.approx(140_000)

    def test_with_labor_income(self):
        """工资 15w + 劳务报酬 5w，劳务按 80% 计入"""
        result = calculate_individual_income_annual(
            annual_wage=150_000,
            annual_labor=50_000,
        )
        # labor_counted = 50000 * 0.8 = 40000
        # total = 150000 + 40000 = 190000
        # taxable = 190000 - 60000 = 130000, rate 10%, quick 2520
        expected = 130_000 * 0.10 - 2_520
        assert result.tax_amount == pytest.approx(expected)

    def test_with_royalties(self):
        """稿酬按 80%×70%=56% 计入"""
        result = calculate_individual_income_annual(
            annual_wage=100_000,
            annual_royalty=100_000,
        )
        # royalty_counted = 100000 * 0.8 * 0.7 = 56000
        # total = 100000 + 56000 = 156000
        # taxable = 156000 - 60000 = 96000, rate 10%, quick 2520
        expected = 96_000 * 0.10 - 2_520
        assert result.tax_amount == pytest.approx(expected)

    def test_with_special_deductions(self):
        """工资 30w + 专项扣除 2w + 附加扣除 2.4w"""
        result = calculate_individual_income_annual(
            annual_wage=300_000,
            special_deduction=20_000,
            special_additional_deduction=24_000,
        )
        # taxable = 300000 - 60000 - 20000 - 24000 = 196000, rate 20%, quick 16920
        expected = 196_000 * 0.20 - 16_920
        assert result.tax_amount == pytest.approx(expected)

    def test_paid_tax_exceeds_owed(self):
        """已预缴 > 应纳税 → notes 中应提示应退"""
        result = calculate_individual_income_annual(
            annual_wage=100_000,
            paid_tax=10_000,           # 预缴 1 万
        )
        # taxable = 100000 - 60000 = 40000, rate 10%, quick 2520
        expected_tax = 40_000 * 0.10 - 2_520
        assert result.tax_amount == pytest.approx(expected_tax)
        # expected_tax ≈ 1480, paid 10000 → 应退 8520
        assert any("应退" in n or "-" in n for n in result.notes)

    def test_zero_income(self):
        """零收入 → 应纳税 0"""
        result = calculate_individual_income_annual(annual_wage=0)
        assert result.tax_amount == 0.0


class TestMonthlyWageWithholding:
    """工资薪金月度累计预扣法"""

    def test_first_month_basic(self):
        """第 1 月，月薪 1.5 万 → 预扣 300"""
        result = calculate_monthly_wage_withholding(
            monthly_wage=15_000,
            month_index=1,
        )
        # cumulative_income = 15000, cumulative_deduct = 5000
        # taxable = 10000, rate 3%, tax = 300
        assert result.tax_amount == pytest.approx(300.0)

    def test_third_month_cumulative(self):
        """
        第 3 月，月薪 1.5 万，前两月累计已预扣 600。
        cumulative_income = 45000, cumulative_deduct = 15000
        taxable = 30000, rate 3%, cumulative_tax = 900
        withheld = 900 - 600 = 300
        """
        result = calculate_monthly_wage_withholding(
            monthly_wage=15_000,
            month_index=3,
            prior_income=30_000,
            prior_tax_withheld=600,
        )
        assert result.tax_amount == pytest.approx(300.0)

    def test_cross_bracket(self):
        """
        第 4 月，月薪 3 万，前三月累计收入 4.5 万、已预扣 900。
        cumulative_tax = 55000×10% - 2520 = 2980
        monthly_withheld = 2980 - 900 = 2080
        注意：tax_amount 是本月预扣额，不是累计应纳税额。
        """
        result = calculate_monthly_wage_withholding(
            monthly_wage=30_000,
            month_index=4,
            prior_income=45_000,
            prior_tax_withheld=900,
        )
        cumulative_tax = 55_000 * 0.10 - 2_520
        expected = cumulative_tax - 900          # monthly_withheld, not cumulative_tax
        assert result.tax_amount == pytest.approx(expected)

    def test_zero_wage(self):
        """零工资 → 预扣 0"""
        result = calculate_monthly_wage_withholding(
            monthly_wage=0,
            month_index=1,
        )
        assert result.tax_amount == 0.0

    def test_prior_withheld_exceeds_cumulative(self):
        """累计已预扣 > 累计应纳税（如年中降薪）→ 本月不扣"""
        result = calculate_monthly_wage_withholding(
            monthly_wage=3_000,
            month_index=6,
            prior_income=200_000,
            prior_tax_withheld=20_000,
        )
        # 累计应纳税很小，已预扣远超 → 本月预扣 0
        assert result.tax_amount == 0.0


class TestBusinessIncomeTax:
    """经营所得个税（5 级超额累进）"""

    def test_first_bracket(self):
        """利润 2.5 万，5% 档"""
        result = calculate_business_income_tax(annual_profit=25_000)
        assert result.tax_amount == pytest.approx(1_250.0)
        assert result.tax_rate is not None

    def test_second_bracket(self):
        """利润 6 万（减扣除后 4.8 万），10% 档"""
        result = calculate_business_income_tax(
            annual_profit=60_000,
            special_additional_deduction=12_000,
        )
        # taxable = 60000 - 12000 = 48000, rate 10%, quick 1500
        expected = 48_000 * 0.10 - 1_500
        assert result.tax_amount == pytest.approx(expected)

    def test_top_bracket(self):
        """利润 60 万，最高 35% 档"""
        result = calculate_business_income_tax(annual_profit=600_000)
        # rate 35%, quick 65500
        expected = 600_000 * 0.35 - 65_500
        assert result.tax_amount == pytest.approx(expected)

    def test_zero_profit(self):
        """零利润 → 应纳税 0"""
        result = calculate_business_income_tax(annual_profit=0)
        assert result.tax_amount == 0.0


# ============================================================
# 6. 边界值 / 异常输入
# ============================================================

class TestEdgeCases:
    def test_zero_sales(self):
        result = calculate_vat(
            taxpayer_type="general",
            sales_amount=0,
            input_tax=0,
            vat_rate=0.13,
        )
        assert result.tax_amount == 0.0

    def test_negative_sales_handled(self):
        """负销售额：当前实现不抛异常，但销项为负 → 抵扣后为负 → 留抵归零逻辑兜住，税额应为 0 且不为负"""
        result = calculate_vat(
            taxpayer_type="general",
            sales_amount=-1000,
            vat_rate=0.13,
        )
        # 销项 = -1000 × 13% = -130 → net_tax = -130 < 0 → 置 0
        assert result.tax_amount == 0.0