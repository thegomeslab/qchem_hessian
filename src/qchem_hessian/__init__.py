"""ASE Q-Chem calculator with Hessian support."""

import warnings
from typing import Any

from qchem_hessian.calculator import QChemHessian
from qchem_hessian.parser import QChemHessianData, parse_qchem_hessian_output

__all__ = ["QChemHessian", "QChemHessianData", "parse_qchem_hessian_output"]


def __getattr__(name: str) -> Any:
    if name == "QChemAnalyticHessian":
        warnings.warn(
            "QChemAnalyticHessian is deprecated; use QChemHessian.",
            DeprecationWarning,
            stacklevel=2,
        )
        return QChemHessian
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
