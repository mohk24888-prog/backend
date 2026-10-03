"""Analysis services.

`cv_integration` imports torch/ultralytics, which are not installed on the
Render free tier. Import it lazily so lightweight modules in this package -
such as `synthetic_tracking` - stay importable without the CV stack.
"""

from typing import Any

__all__ = ["CVIntegrationService"]


def __getattr__(name: str) -> Any:
    if name == "CVIntegrationService":
        from services.analysis.cv_integration import CVIntegrationService

        return CVIntegrationService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")