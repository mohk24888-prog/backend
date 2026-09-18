from __future__ import annotations

from services.analysis.analyzers.base import BaseTestAnalyzer
from services.analysis.analyzers.sprint import SprintAnalyzer
from services.analysis.analyzers.agility import AgilityAnalyzer
from services.analysis.analyzers.dribbling import DribblingAnalyzer
from services.analysis.analyzers.ball_control import BallControlAnalyzer
from services.analysis.analyzers.passing import PassingAnalyzer
from services.analysis.analyzers.shooting import ShootingAnalyzer
from services.analysis.analyzers.endurance import EnduranceAnalyzer


class TestAnalyzerFactory:
    _analyzers: dict[str, BaseTestAnalyzer] = {}

    @classmethod
    def register(cls, analyzer: BaseTestAnalyzer) -> None:
        cls._analyzers[analyzer.test_type] = analyzer

    @classmethod
    def get(cls, test_type: str) -> BaseTestAnalyzer:
        analyzer = cls._analyzers.get(test_type)
        if analyzer is None:
            raise ValueError(f"Unknown test type: {test_type}")
        return analyzer

    @classmethod
    def available_types(cls) -> list[str]:
        return list(cls._analyzers.keys())


TestAnalyzerFactory.register(SprintAnalyzer())
TestAnalyzerFactory.register(AgilityAnalyzer())
TestAnalyzerFactory.register(DribblingAnalyzer())
TestAnalyzerFactory.register(BallControlAnalyzer())
TestAnalyzerFactory.register(PassingAnalyzer())
TestAnalyzerFactory.register(ShootingAnalyzer())
TestAnalyzerFactory.register(EnduranceAnalyzer())
