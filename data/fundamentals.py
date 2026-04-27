# -*- coding: utf-8 -*-
"""A-share fundamentals loader with multi-source fallback and coverage tracking."""

import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from config import get_settings

logger = logging.getLogger(__name__)
_LAST_FETCH_DETAILS: Dict[str, Any] = {}
_FUNDAMENTAL_CACHE_VERSION = 2
_HISTORICAL_CACHE_SUFFIX = "_history.json"

try:
    import akshare as ak

    _AK_AVAILABLE = True
except ImportError:
    ak = None
    _AK_AVAILABLE = False


METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "roe": ("净资产收益率", "ROE", "净资产收益率摊薄"),
    "net_margin": ("销售净利率", "净利率", "净利润率"),
    "gross_margin": ("销售毛利率", "毛利率"),
    "asset_liability_ratio": ("资产负债率",),
    "current_ratio": ("流动比率",),
    "cash_ratio": ("现金比率", "速动比率"),
    "net_profit_yoy": ("净利润同比增长率", "归母净利润同比增长率", "净利润同比"),
    "eps": ("基本每股收益", "每股收益", "EPS"),
    "operating_cashflow": ("经营活动产生的现金流量净额", "经营现金流量净额", "经营现金流"),
    "free_cashflow": ("自由现金流",),
    "profit_cash_ratio": ("净利润现金含量", "经营现金流/净利润"),
    "goodwill_ratio": ("商誉占净资产", "商誉占总资产", "商誉"),
    "inventory_turnover": ("存货周转率",),
    "receivables_turnover": ("应收账款周转率",),
    "interest_bearing_debt_ratio": ("有息负债率",),
    "rd_ratio": ("研发费用率", "研发费用占营业收入比重", "研发费用/营业收入"),
}

STATEMENT_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "report_date": ("报告期", "报告日期", "截止日期", "日期"),
    "announcement_date": ("公告日期", "披露日期", "首次公告日期", "实际披露日期"),
    "total_assets": ("资产总计", "总资产"),
    "total_liabilities": ("负债合计", "总负债"),
    "current_assets": ("流动资产合计", "流动资产"),
    "current_liabilities": ("流动负债合计", "流动负债"),
    "monetary_funds": ("货币资金", "现金及现金等价物", "现金及等价物"),
    "goodwill": ("商誉",),
    "inventory_turnover": ("存货周转率",),
    "receivables_turnover": ("应收账款周转率",),
    "interest_bearing_debt_ratio": ("有息负债率",),
    "operating_revenue": ("营业总收入", "营业收入"),
    "net_profit": ("净利润", "归属于母公司股东的净利润", "归母净利润"),
    "basic_eps": ("基本每股收益", "每股收益"),
    "gross_margin": ("销售毛利率", "毛利率"),
    "net_margin": ("销售净利率", "净利率", "净利润率"),
    "roe": ("净资产收益率", "ROE"),
    "net_profit_yoy": ("净利润同比增长率", "归母净利润同比增长率", "净利润同比"),
    "rd_expense": ("研发费用", "研发投入"),
    "operating_cashflow": ("经营活动产生的现金流量净额", "经营现金流量净额", "经营现金流"),
    "capex": ("购建固定资产、无形资产和其他长期资产支付的现金", "资本开支", "购建固定资产支付的现金"),
    "free_cashflow": ("自由现金流",),
}

ABSTRACT_FUNCTIONS = [
    "stock_financial_abstract_ths",
    "stock_financial_analysis_indicator",
]
INCOME_FUNCTIONS = [
    "stock_lrb_em",
    "stock_lrb_ths",
]
BALANCE_FUNCTIONS = [
    "stock_zcfz_em",
    "stock_zcfz_ths",
]
CASHFLOW_FUNCTIONS = [
    "stock_xjll_em",
    "stock_xjll_ths",
]


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "--", "暂无", "不适用"}:
        return None

    negative = text.startswith("(") and text.endswith(")")
    text = text.replace(",", "").replace("%", "").replace("(", "").replace(")", "")
    text = text.replace("亿元", "").replace("万元", "").replace("元", "").replace("倍", "")
    text = text.replace(" ", "")
    if not text:
        return None

    try:
        number = float(text)
        return -number if negative else number
    except ValueError:
        return None


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    text = text.replace("\n", "").replace("\r", "").replace(" ", "")
    return text.lower()


def _is_period_like(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(re.search(r"(19|20)\d{2}", text))


def _utc_now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()


def _parse_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def is_cache_payload_fresh(payload: Dict[str, Any], ttl_days: int) -> bool:
    """Return whether a JSON cache payload is still usable under ttl_days."""
    if ttl_days <= 0:
        return True
    fetched_at = _parse_timestamp(payload.get("fetched_at"))
    if fetched_at is None:
        return False
    age = pd.Timestamp.utcnow() - fetched_at
    return age <= pd.Timedelta(days=ttl_days)


def _period_sort_key(value: Any) -> Tuple[int, int, str]:
    text = str(value or "")
    year_match = re.search(r"((19|20)\d{2})", text)
    year = int(year_match.group(1)) if year_match else 0
    quarter = 12
    quarter_match = re.search(r"([1-4])[季qQ]", text)
    if quarter_match:
        quarter = int(quarter_match.group(1))
    elif "中报" in text or "半年" in text or "-06-" in text:
        quarter = 2
    elif "三季" in text or "-09-" in text:
        quarter = 3
    elif "年报" in text or "-12-" in text:
        quarter = 4
    elif "-03-" in text:
        quarter = 1
    return (year, quarter, text)


def _match_alias(name: Any, aliases: Iterable[str]) -> bool:
    normalized = _normalize_text(name)
    return any(_normalize_text(alias) in normalized for alias in aliases)


def _find_matching_column(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    for col in df.columns:
        if _match_alias(col, aliases):
            return col
    return None


def _stock_code_variants(code: str) -> List[str]:
    plain = str(code).strip()
    if not plain:
        return []
    market = "sh" if plain.startswith("6") else "sz"
    variants = [
        plain,
        f"{market}{plain}",
        f"{market.upper()}{plain}",
        f"{plain}.{market.upper()}",
    ]
    seen = set()
    return [item for item in variants if item and not (item in seen or seen.add(item))]


def _ak_argument_sets(code: str) -> List[Dict[str, Any]]:
    variants = _stock_code_variants(code)
    argument_sets: List[Dict[str, Any]] = []
    for symbol in variants:
        argument_sets.extend(
            [
                {"symbol": symbol},
                {"stock": symbol},
                {"code": symbol},
                {"symbol": symbol, "indicator": "按报告期"},
                {"symbol": symbol, "indicator": "按年度"},
                {"symbol": symbol, "indicator": "按单季度"},
            ]
        )
    return argument_sets


def _call_ak(function_names: Iterable[str], code: str) -> Optional[pd.DataFrame]:
    global _LAST_FETCH_DETAILS
    details = {
        "akshare_available": _AK_AVAILABLE,
        "attempted_functions": [],
        "errors": [],
    }
    _LAST_FETCH_DETAILS = details

    if not _AK_AVAILABLE:
        details["errors"].append("akshare_not_installed")
        return None

    argument_sets = _ak_argument_sets(code)
    for func_name in function_names:
        func = getattr(ak, func_name, None)
        details["attempted_functions"].append(func_name)
        if func is None:
            details["errors"].append(f"{func_name}: missing")
            continue
        for kwargs in argument_sets:
            try:
                result = func(**kwargs)
            except TypeError:
                continue
            except Exception as exc:
                if len(details["errors"]) < 12:
                    details["errors"].append(f"{func_name}{kwargs}: {type(exc).__name__}: {exc}")
                result = None
            if isinstance(result, pd.DataFrame) and not result.empty:
                details["selected_function"] = func_name
                details["selected_kwargs"] = kwargs
                details["rows"] = int(len(result))
                return result
    return None


def _find_label_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ("指标", "项目", "科目", "报表日期", "报告期")
    for col in df.columns:
        if any(token in str(col) for token in candidates):
            return col
    object_cols = [col for col in df.columns if df[col].dtype == "object"]
    return object_cols[0] if object_cols else None


def _extract_from_abstract(df: pd.DataFrame, aliases: Iterable[str]) -> Dict[str, Any]:
    label_col = _find_label_column(df)
    if label_col is None:
        return {"value": None, "period": None, "source": None}

    value_cols = [col for col in df.columns if col != label_col]
    period_cols = sorted([col for col in value_cols if _is_period_like(col)], key=_period_sort_key)
    ordered_cols = period_cols if period_cols else value_cols

    for _, row in df.iterrows():
        label = row.get(label_col, "")
        if not _match_alias(label, aliases):
            continue
        for col in reversed(ordered_cols):
            value = _coerce_float(row.get(col))
            if value is not None:
                return {"value": value, "period": str(col), "source": f"abstract:{label}"}
    return {"value": None, "period": None, "source": None}


def _normalize_statement_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    data = df.copy()
    renamed: Dict[str, str] = {}
    for canonical, aliases in STATEMENT_FIELD_ALIASES.items():
        column = _find_matching_column(data, aliases)
        if column:
            renamed[column] = canonical
    data = data.rename(columns=renamed)

    if "report_date" not in data.columns:
        period_candidates = [col for col in data.columns if _is_period_like(col)]
        if len(period_candidates) == 1:
            data = data.rename(columns={period_candidates[0]: "report_date"})

    if "report_date" in data.columns:
        data["report_date"] = data["report_date"].astype(str)
        data = data.dropna(subset=["report_date"])
        data = data[data["report_date"].str.strip() != ""]
        data = data.sort_values("report_date", key=lambda s: s.map(_period_sort_key)).reset_index(drop=True)

    for col in list(data.columns):
        if col in {"report_date", "announcement_date"}:
            continue
        coerced = data[col].map(_coerce_float)
        if coerced.notna().sum() > 0:
            data[col] = coerced

    return data


def _latest_statement_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty:
        return None
    if "report_date" in df.columns:
        return df.sort_values("report_date", key=lambda s: s.map(_period_sort_key)).iloc[-1]
    return df.iloc[-1]


def _row_report_period(row: Optional[pd.Series], fallback: Optional[str] = None) -> Optional[str]:
    if row is None:
        return fallback
    if "report_date" in row.index:
        period = row.get("report_date")
        if period:
            return str(period)
    return fallback


def _find_same_period_row(df: pd.DataFrame, period: Optional[str]) -> Optional[pd.Series]:
    if df.empty or "report_date" not in df.columns or not period:
        return None
    matched = df[df["report_date"].astype(str) == str(period)]
    if matched.empty:
        return None
    return matched.iloc[-1]


def _find_previous_same_period(df: pd.DataFrame, latest_period: str) -> Optional[pd.Series]:
    if df.empty or "report_date" not in df.columns or not latest_period:
        return None
    latest_key = _period_sort_key(latest_period)
    latest_quarter = latest_key[1]
    latest_year = latest_key[0]
    previous = df[df["report_date"].map(lambda x: _period_sort_key(x)[0] == latest_year - 1 and _period_sort_key(x)[1] == latest_quarter)]
    if previous.empty:
        return None
    return previous.iloc[-1]


def _safe_div(numerator: Optional[float], denominator: Optional[float], multiplier: float = 1.0) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator) * multiplier


def _build_metric(value: Optional[float], period: Optional[str], source: Optional[str]) -> Dict[str, Any]:
    return {"value": value, "period": period, "source": source}


def _fill_direct_metrics(
    snapshot_metrics: Dict[str, Dict[str, Any]],
    sources_used: Dict[str, str],
    abstract_df: pd.DataFrame,
) -> None:
    if abstract_df.empty:
        return

    for metric_name, aliases in METRIC_ALIASES.items():
        extracted = _extract_from_abstract(abstract_df, aliases)
        snapshot_metrics[metric_name] = extracted
        if extracted["value"] is not None:
            sources_used[metric_name] = extracted["source"] or "abstract"


def _fill_derived_metrics(
    snapshot_metrics: Dict[str, Dict[str, Any]],
    sources_used: Dict[str, str],
    balance_df: pd.DataFrame,
    income_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
) -> None:
    latest_balance = _latest_statement_row(balance_df)
    latest_income = _latest_statement_row(income_df)
    latest_cashflow = _latest_statement_row(cashflow_df)

    def set_if_missing(metric_name: str, value: Optional[float], period: Optional[str], source: str):
        current = snapshot_metrics.get(metric_name, {})
        if current.get("value") is not None or value is None:
            return
        snapshot_metrics[metric_name] = _build_metric(value, period, source)
        sources_used[metric_name] = source

    if latest_balance is not None:
        balance_period = _row_report_period(latest_balance)
        total_assets = _coerce_float(latest_balance.get("total_assets"))
        total_liabilities = _coerce_float(latest_balance.get("total_liabilities"))
        current_assets = _coerce_float(latest_balance.get("current_assets"))
        current_liabilities = _coerce_float(latest_balance.get("current_liabilities"))
        monetary_funds = _coerce_float(latest_balance.get("monetary_funds"))
        goodwill = _coerce_float(latest_balance.get("goodwill"))

        set_if_missing(
            "asset_liability_ratio",
            _safe_div(total_liabilities, total_assets, 100.0),
            balance_period,
            "balance_sheet:负债合计/资产总计",
        )
        set_if_missing(
            "current_ratio",
            _safe_div(current_assets, current_liabilities, 1.0),
            balance_period,
            "balance_sheet:流动资产/流动负债",
        )
        set_if_missing(
            "cash_ratio",
            _safe_div(monetary_funds, current_liabilities, 1.0),
            balance_period,
            "balance_sheet:货币资金/流动负债",
        )
        set_if_missing(
            "goodwill_ratio",
            _safe_div(goodwill, total_assets, 100.0),
            balance_period,
            "balance_sheet:商誉/资产总计",
        )

    if latest_income is not None:
        report_date = _row_report_period(latest_income)
        revenue = _coerce_float(latest_income.get("operating_revenue"))
        net_profit = _coerce_float(latest_income.get("net_profit"))
        basic_eps = _coerce_float(latest_income.get("basic_eps"))
        rd_expense = _coerce_float(latest_income.get("rd_expense"))
        prev_income = _find_previous_same_period(income_df, report_date)
        prev_net_profit = _coerce_float(prev_income.get("net_profit")) if prev_income is not None else None

        if snapshot_metrics.get("eps", {}).get("value") is None and basic_eps is not None:
            snapshot_metrics["eps"] = _build_metric(basic_eps, report_date, "income_statement:基本每股收益")
            sources_used["eps"] = "income_statement:基本每股收益"

        set_if_missing(
            "net_margin",
            _safe_div(net_profit, revenue, 100.0),
            report_date,
            "income_statement:净利润/营业收入",
        )
        set_if_missing(
            "rd_ratio",
            _safe_div(rd_expense, revenue, 100.0),
            report_date,
            "income_statement:研发费用/营业收入",
        )
        set_if_missing(
            "net_profit_yoy",
            _safe_div((net_profit - prev_net_profit) if net_profit is not None and prev_net_profit is not None else None, abs(prev_net_profit) if prev_net_profit is not None else None, 100.0),
            report_date,
            "income_statement:同比净利润增速",
        )

    if latest_cashflow is not None:
        report_date = _row_report_period(latest_cashflow)
        operating_cashflow = _coerce_float(latest_cashflow.get("operating_cashflow"))
        capex = _coerce_float(latest_cashflow.get("capex"))
        free_cashflow = _coerce_float(latest_cashflow.get("free_cashflow"))

        if snapshot_metrics.get("operating_cashflow", {}).get("value") is None and operating_cashflow is not None:
            snapshot_metrics["operating_cashflow"] = _build_metric(
                operating_cashflow,
                report_date,
                "cashflow_statement:经营活动现金流净额",
            )
            sources_used["operating_cashflow"] = "cashflow_statement:经营活动现金流净额"

        if snapshot_metrics.get("free_cashflow", {}).get("value") is None:
            derived_fcf = free_cashflow
            if derived_fcf is None and operating_cashflow is not None and capex is not None:
                derived_fcf = operating_cashflow - abs(capex)
            if derived_fcf is not None:
                snapshot_metrics["free_cashflow"] = _build_metric(
                    derived_fcf,
                    report_date,
                    "cashflow_statement:经营现金流-资本开支",
                )
                sources_used["free_cashflow"] = "cashflow_statement:经营现金流-资本开支"

        same_period_income = _find_same_period_row(income_df, report_date)
        if same_period_income is not None:
            net_profit = _coerce_float(same_period_income.get("net_profit"))
            ratio = _safe_div(operating_cashflow, net_profit, 1.0)
            set_if_missing(
                "profit_cash_ratio",
                ratio,
                report_date,
                "cashflow_statement:经营现金流/净利润",
            )


def fetch_financial_abstract(code: str) -> pd.DataFrame:
    result = _call_ak(ABSTRACT_FUNCTIONS, code=code)
    return result if isinstance(result, pd.DataFrame) else pd.DataFrame()


def fetch_income_statement(code: str) -> pd.DataFrame:
    result = _call_ak(INCOME_FUNCTIONS, code=code)
    return result if isinstance(result, pd.DataFrame) else pd.DataFrame()


def fetch_balance_sheet(code: str) -> pd.DataFrame:
    result = _call_ak(BALANCE_FUNCTIONS, code=code)
    return result if isinstance(result, pd.DataFrame) else pd.DataFrame()


def fetch_cashflow_statement(code: str) -> pd.DataFrame:
    result = _call_ak(CASHFLOW_FUNCTIONS, code=code)
    return result if isinstance(result, pd.DataFrame) else pd.DataFrame()


def _row_announcement_date(row: Optional[pd.Series]) -> Optional[pd.Timestamp]:
    if row is None or "announcement_date" not in row.index:
        return None
    parsed = pd.to_datetime(row.get("announcement_date"), errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _estimated_announcement_date(period: Optional[str]) -> Optional[pd.Timestamp]:
    key = _period_sort_key(period)
    year, quarter = key[0], key[1]
    if year <= 0:
        return None
    month_day = {
        1: (4, 30),
        2: (8, 31),
        3: (10, 31),
        4: (4, 30),
    }.get(quarter, (4, 30))
    ann_year = year + 1 if quarter == 4 else year
    return pd.Timestamp(year=ann_year, month=month_day[0], day=month_day[1])


def _latest_by_period(df: pd.DataFrame) -> Dict[str, pd.Series]:
    if df.empty or "report_date" not in df.columns:
        return {}
    out: Dict[str, pd.Series] = {}
    ordered = df.sort_values("report_date", key=lambda s: s.map(_period_sort_key))
    for _, row in ordered.iterrows():
        period = _row_report_period(row)
        if period:
            out[period] = row
    return out


def _metric_record(value: Optional[float], period: Optional[str], source: str) -> Dict[str, Any]:
    return {"value": value, "period": period, "source": source}


def _build_history_record(
    code: str,
    name: str,
    period: str,
    balance_row: Optional[pd.Series],
    income_row: Optional[pd.Series],
    cashflow_row: Optional[pd.Series],
    strict_announcement_date: bool,
) -> Optional[Dict[str, Any]]:
    announcement_dates = [
        d for d in (
            _row_announcement_date(balance_row),
            _row_announcement_date(income_row),
            _row_announcement_date(cashflow_row),
        )
        if d is not None
    ]
    announcement_source = "reported"
    if announcement_dates:
        announcement_date = max(announcement_dates)
    elif strict_announcement_date:
        return None
    else:
        announcement_date = _estimated_announcement_date(period)
        announcement_source = "estimated_conservative"
        if announcement_date is None:
            return None

    metrics = {metric_name: {"value": None, "period": None, "source": None} for metric_name in METRIC_ALIASES}

    if balance_row is not None:
        total_assets = _coerce_float(balance_row.get("total_assets"))
        total_liabilities = _coerce_float(balance_row.get("total_liabilities"))
        current_assets = _coerce_float(balance_row.get("current_assets"))
        current_liabilities = _coerce_float(balance_row.get("current_liabilities"))
        monetary_funds = _coerce_float(balance_row.get("monetary_funds"))
        goodwill = _coerce_float(balance_row.get("goodwill"))
        metrics["asset_liability_ratio"] = _metric_record(
            _safe_div(total_liabilities, total_assets, 100.0), period, "history:balance_sheet"
        )
        metrics["current_ratio"] = _metric_record(
            _safe_div(current_assets, current_liabilities, 1.0), period, "history:balance_sheet"
        )
        metrics["cash_ratio"] = _metric_record(
            _safe_div(monetary_funds, current_liabilities, 1.0), period, "history:balance_sheet"
        )
        metrics["goodwill_ratio"] = _metric_record(
            _safe_div(goodwill, total_assets, 100.0), period, "history:balance_sheet"
        )

    if income_row is not None:
        revenue = _coerce_float(income_row.get("operating_revenue"))
        net_profit = _coerce_float(income_row.get("net_profit"))
        rd_expense = _coerce_float(income_row.get("rd_expense"))
        basic_eps = _coerce_float(income_row.get("basic_eps"))
        metrics["eps"] = _metric_record(basic_eps, period, "history:income_statement")
        metrics["net_margin"] = _metric_record(
            _safe_div(net_profit, revenue, 100.0), period, "history:income_statement"
        )
        metrics["rd_ratio"] = _metric_record(
            _safe_div(rd_expense, revenue, 100.0), period, "history:income_statement"
        )

    if cashflow_row is not None:
        operating_cashflow = _coerce_float(cashflow_row.get("operating_cashflow"))
        capex = _coerce_float(cashflow_row.get("capex"))
        free_cashflow = _coerce_float(cashflow_row.get("free_cashflow"))
        if free_cashflow is None and operating_cashflow is not None and capex is not None:
            free_cashflow = operating_cashflow - abs(capex)
        metrics["operating_cashflow"] = _metric_record(
            operating_cashflow, period, "history:cashflow_statement"
        )
        metrics["free_cashflow"] = _metric_record(free_cashflow, period, "history:cashflow_statement")
        if income_row is not None:
            net_profit = _coerce_float(income_row.get("net_profit"))
            metrics["profit_cash_ratio"] = _metric_record(
                _safe_div(operating_cashflow, net_profit, 1.0), period, "history:cashflow_statement"
            )

    filled_metrics = [metric_name for metric_name, entry in metrics.items() if entry.get("value") is not None]
    missing_metrics = [metric_name for metric_name, entry in metrics.items() if entry.get("value") is None]
    return {
        "code": code,
        "name": name,
        "report_date": period,
        "announcement_date": announcement_date.strftime("%Y-%m-%d"),
        "announcement_date_source": announcement_source,
        "metrics": metrics,
        "filled_metrics": filled_metrics,
        "missing_metrics": missing_metrics,
        "coverage_ratio": round(len(filled_metrics) / max(len(metrics), 1), 4),
    }


def load_fundamental_history(
    code: str,
    name: str = "",
    refresh: bool = False,
    strict_announcement_date: bool = True,
) -> pd.DataFrame:
    """Load point-in-time fundamental rows keyed by announcement_date.

    Backtests must call get_fundamental_snapshot_asof() on this output instead of
    using the latest snapshot, otherwise current filings can leak into history.
    """
    settings = get_settings()
    ttl_days = int(getattr(settings.cache, "fundamental_ttl_days", 30))
    force_refresh = bool(getattr(settings.cache, "force_refresh_fundamentals", False))
    cache_path = os.path.join(settings.paths.fundamental_cache_dir, f"{code}{_HISTORICAL_CACHE_SUFFIX}")
    should_refresh = refresh or force_refresh

    if os.path.exists(cache_path) and not should_refresh:
        try:
            with open(cache_path, "r", encoding="utf-8") as file:
                cached = json.load(file)
            if (
                cached.get("cache_version") == _FUNDAMENTAL_CACHE_VERSION
                and is_cache_payload_fresh(cached, ttl_days)
                and cached.get("strict_announcement_date") == strict_announcement_date
            ):
                rows = cached.get("records", [])
                history_df = pd.DataFrame(rows)
                if not history_df.empty:
                    history_df["announcement_date"] = pd.to_datetime(history_df["announcement_date"], errors="coerce")
                    history_df = history_df.dropna(subset=["announcement_date"]).sort_values("announcement_date")
                return history_df
        except Exception:
            logger.warning("Failed to load cached fundamental history for %s", code)

    income_df = _normalize_statement_df(fetch_income_statement(code))
    balance_df = _normalize_statement_df(fetch_balance_sheet(code))
    cashflow_df = _normalize_statement_df(fetch_cashflow_statement(code))

    income_by_period = _latest_by_period(income_df)
    balance_by_period = _latest_by_period(balance_df)
    cashflow_by_period = _latest_by_period(cashflow_df)
    periods = sorted(
        set(income_by_period) | set(balance_by_period) | set(cashflow_by_period),
        key=_period_sort_key,
    )

    records = []
    for period in periods:
        record = _build_history_record(
            code=code,
            name=name,
            period=period,
            balance_row=balance_by_period.get(period),
            income_row=income_by_period.get(period),
            cashflow_row=cashflow_by_period.get(period),
            strict_announcement_date=strict_announcement_date,
        )
        if record is not None:
            records.append(record)

    payload = {
        "code": code,
        "name": name,
        "fetched_at": _utc_now_iso(),
        "cache_ttl_days": ttl_days,
        "cache_version": _FUNDAMENTAL_CACHE_VERSION,
        "strict_announcement_date": strict_announcement_date,
        "records": records,
    }
    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    history_df = pd.DataFrame(records)
    if not history_df.empty:
        history_df["announcement_date"] = pd.to_datetime(history_df["announcement_date"], errors="coerce")
        history_df = history_df.dropna(subset=["announcement_date"]).sort_values("announcement_date")
    return history_df


def get_fundamental_snapshot_asof(history_df: pd.DataFrame, trade_date: Any) -> Optional[Dict[str, Any]]:
    """Return the newest fundamental snapshot announced on or before trade_date."""
    if history_df is None or history_df.empty or "announcement_date" not in history_df.columns:
        return None
    date = pd.to_datetime(trade_date, errors="coerce")
    if pd.isna(date):
        return None
    data = history_df.copy()
    data["announcement_date"] = pd.to_datetime(data["announcement_date"], errors="coerce")
    data = data.dropna(subset=["announcement_date"])
    available = data[data["announcement_date"] <= pd.Timestamp(date)]
    if available.empty:
        return None
    row = available.sort_values("announcement_date").iloc[-1]
    return row.to_dict()


def load_fundamental_snapshot(code: str, name: str = "", refresh: bool = False) -> Dict[str, Any]:
    settings = get_settings()
    cache_path = os.path.join(settings.paths.fundamental_cache_dir, f"{code}_snapshot.json")
    ttl_days = int(getattr(settings.cache, "fundamental_ttl_days", 30))
    force_refresh = bool(getattr(settings.cache, "force_refresh_fundamentals", False))
    should_refresh = refresh or force_refresh
    if os.path.exists(cache_path) and not should_refresh:
        try:
            with open(cache_path, "r", encoding="utf-8") as file:
                cached = json.load(file)
            cache_fresh = is_cache_payload_fresh(cached, ttl_days)
            cache_version_ok = cached.get("cache_version") == _FUNDAMENTAL_CACHE_VERSION
            if cached.get("raw_available") and float(cached.get("coverage_ratio", 0.0)) > 0 and cache_fresh and cache_version_ok:
                return cached
            if not cache_fresh:
                logger.info("Ignore stale fundamentals snapshot for %s", code)
            elif not cache_version_ok:
                logger.info("Ignore old-version fundamentals snapshot for %s", code)
            else:
                logger.info("Ignore stale empty fundamentals snapshot for %s", code)
        except Exception:
            logger.warning("Failed to load cached fundamentals for %s", code)

    metrics = {metric_name: {"value": None, "period": None, "source": None} for metric_name in METRIC_ALIASES}
    sources_used: Dict[str, str] = {}

    fetch_details: Dict[str, Dict[str, Any]] = {}

    abstract_df = fetch_financial_abstract(code)
    fetch_details["abstract"] = dict(_LAST_FETCH_DETAILS)

    income_raw_df = fetch_income_statement(code)
    fetch_details["income"] = dict(_LAST_FETCH_DETAILS)
    income_df = _normalize_statement_df(income_raw_df)

    balance_raw_df = fetch_balance_sheet(code)
    fetch_details["balance"] = dict(_LAST_FETCH_DETAILS)
    balance_df = _normalize_statement_df(balance_raw_df)

    cashflow_raw_df = fetch_cashflow_statement(code)
    fetch_details["cashflow"] = dict(_LAST_FETCH_DETAILS)
    cashflow_df = _normalize_statement_df(cashflow_raw_df)

    _fill_direct_metrics(metrics, sources_used, abstract_df)
    _fill_derived_metrics(metrics, sources_used, balance_df, income_df, cashflow_df)

    periods: List[str] = []
    for entry in metrics.values():
        period = entry.get("period")
        if period:
            periods.append(str(period))
    latest_period = max(periods, key=_period_sort_key) if periods else None

    filled_metrics = [name for name, entry in metrics.items() if entry.get("value") is not None]
    missing_metrics = [name for name, entry in metrics.items() if entry.get("value") is None]
    coverage_ratio = round(len(filled_metrics) / max(len(metrics), 1), 4)

    raw_available = bool(not abstract_df.empty or not income_df.empty or not balance_df.empty or not cashflow_df.empty)
    source_details = {
        "akshare_available": _AK_AVAILABLE,
        "abstract_rows": int(len(abstract_df)) if not abstract_df.empty else 0,
        "income_rows": int(len(income_df)) if not income_df.empty else 0,
        "balance_rows": int(len(balance_df)) if not balance_df.empty else 0,
        "cashflow_rows": int(len(cashflow_df)) if not cashflow_df.empty else 0,
        "metric_sources": sources_used,
        "fetch_details": fetch_details,
    }

    snapshot = {
        "code": code,
        "name": name,
        "data_source": "akshare" if raw_available else "unavailable",
        "latest_period": latest_period,
        "metrics": metrics,
        "raw_available": raw_available,
        "filled_metrics": filled_metrics,
        "missing_metrics": missing_metrics,
        "coverage_ratio": coverage_ratio,
        "source_details": source_details,
        "fetched_at": _utc_now_iso(),
        "cache_ttl_days": ttl_days,
        "cache_version": _FUNDAMENTAL_CACHE_VERSION,
    }

    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump(snapshot, file, ensure_ascii=False, indent=2)
    return snapshot
