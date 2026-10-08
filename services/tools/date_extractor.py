"""
从用户自然语言查询中自动提取日期。

支持的表达：
  - "2023年"            → 2023-12-31
  - "2023年3月"         → 2023-03-31
  - "2023-03-15"        → 2023-03-15
  - "去年" / "上一年"   → 当前年-1 的 12-31
  - "今年" / "本年"     → 当前年 12-31
  - "上个月"            → 上个月最后一天
  - "去年3月"           → 去年3月最后一天
  - "没有提及日期"      → None（使用最新条例）
"""

import re
import calendar
from datetime import date
from typing import Optional

MONTH_MAP = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
    "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12,
}

def _last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]

def extract_query_date(text: str, today: Optional[date] = None) -> Optional[str]:
    if not text:
        return None
    today = today or date.today()

    # 1. 完整日期：YYYY-MM-DD
    full_date = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if full_date:
        y, m, d = map(int, full_date.groups())
        try:
            return date(y, m, d).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # 2. "2023年3月" / "2023年03月"
    ym = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", text)
    if ym:
        y = int(ym.group(1))
        m = int(ym.group(2))
        if 1 <= m <= 12:
            d = _last_day_of_month(y, m)
            return f"{y:04d}-{m:02d}-{d:02d}"

    # 3. "2023年" 单独出现
    y_only = re.search(r"(\d{4})\s*年", text)
    if y_only:
        y = int(y_only.group(1))
        return f"{y:04d}-12-31"

    # 4. 相对时间：去年
    if re.search(r"去年|上一年|前一年", text):
        y = today.year - 1
        m_match = re.search(r"(去年|上一年)\s*(\d{1,2})\s*月", text)
        if m_match:
            m = int(m_match.group(2))
            if 1 <= m <= 12:
                d = _last_day_of_month(y, m)
                return f"{y:04d}-{m:02d}-{d:02d}"
        return f"{y:04d}-12-31"

    # 5. 相对时间：今年 / 本年
    if re.search(r"今年|本年", text):
        y = today.year
        m_match = re.search(r"(今年|本年)\s*(\d{1,2})\s*月", text)
        if m_match:
            m = int(m_match.group(2))
            if 1 <= m <= 12:
                d = _last_day_of_month(y, m)
                return f"{y:04d}-{m:02d}-{d:02d}"
        return f"{y:04d}-12-31"

    # 6. 相对时间：上个月
    if re.search(r"上个?月", text):
        if today.month == 1:
            y, m = today.year - 1, 12
        else:
            y, m = today.year, today.month - 1
        d = _last_day_of_month(y, m)
        return f"{y:04d}-{m:02d}-{d:02d}"

    # 7. 今天 / 最新
    if re.search(r"今天|最新|当前最新", text):
        return today.strftime("%Y-%m-%d")

    # 8. 没找到 → None（不限制日期，默认用最新条例）
    return None