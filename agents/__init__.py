# -*- coding: utf-8 -*-
"""Local analyst agents inspired by ai-hedge-fund."""

from agents.fundamentals_agent import analyze_fundamentals
from agents.technicals_agent import analyze_technicals

__all__ = ["analyze_fundamentals", "analyze_technicals"]
