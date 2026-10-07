"""ASE Q-Chem calculator subclass with analytic Hessian support."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from ase.calculators.calculator import FileIOCalculator, all_changes
from ase.calculators.qchem import QChem

from qchem_analytic_hessian.parser import QChemHessianData, parse_qchem_hessian_output


_REM_INT_KEYS = {"charge", "multiplicity"}


def _rem_value(value: Any) -> Any:
    """Return a value compatible with ASE's QChem.write_input implementation."""

    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _normalize_rem_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key.lower() in _REM_INT_KEYS:
            normalized[key] = value
        else:
            normalized[key] = _rem_value(value)
    return normalized


def _with_default_rem(kwargs: dict[str, Any], key: str, value: Any) -> None:
    if not any(existing.lower() == key.lower() for existing in kwargs):
        kwargs[key] = value


class QChemAnalyticHessian(QChem):
    """Q-Chem ASE calculator that exposes Q-Chem analytic Hessians.

    The class subclasses ASE's :class:`ase.calculators.qchem.QChem` and preserves
    its normal energy/force behavior.  Hessians are calculated by running a
    separate Q-Chem ``JOBTYPE FREQ`` calculation with ``VIBMAN_PRINT`` set high
    enough to print the mass-weighted Hessian matrices.
    """

    name = "QChemAnalyticHessian"
    implemented_properties = tuple(dict.fromkeys([*QChem.implemented_properties, "hessian"]))

    def __init__(
        self,
        restart: str | None = None,
        ignore_bad_restart_file: Any = FileIOCalculator._deprecated,
        label: str = "qchem",
        scratch: str | None = None,
        np: int = 1,
        nt: int = 1,
        pbs: bool = False,
        basisfile: str | None = None,
        ecpfile: str | None = None,
        atoms: Any = None,
        *,
        vibman_print: int | str = 6,
        hessian_label_suffix: str = "_freq",
        hessian_scratch: str | None = None,
        no_reorient: bool = True,
        **kwargs: Any,
    ) -> None:
        self.vibman_print = int(vibman_print)
        self.hessian_label_suffix = hessian_label_suffix
        self.hessian_scratch = hessian_scratch
        self.qchem_scratch = scratch
        self.qchem_np = int(np)
        self.qchem_nt = int(nt)
        self.qchem_pbs = bool(pbs)
        self.last_hessian_data: QChemHessianData | None = None

        rem_kwargs = dict(kwargs)
        if no_reorient:
            _with_default_rem(rem_kwargs, "no_reorient", True)
        super().__init__(
            restart=restart,
            ignore_bad_restart_file=ignore_bad_restart_file,
            label=label,
            scratch=scratch,
            np=np,
            nt=nt,
            pbs=pbs,
            basisfile=basisfile,
            ecpfile=ecpfile,
            atoms=atoms,
            **_normalize_rem_kwargs(rem_kwargs),
        )
        jobtype = str(self.parameters.get("jobtype", "") or "").lower()
        if jobtype == "freq":
            self.parameters["vibman_print"] = str(max(self.vibman_print, 4))

    def read_results(self) -> None:
        """Read standard Q-Chem results and parse a Hessian when present."""

        super().read_results()
        atoms = getattr(self, "atoms", None)
        if atoms is None:
            return
        output_path = Path(f"{self.label}.out")
        if not output_path.exists():
            return
        try:
            data = parse_qchem_hessian_output(
                output_path,
                reference_positions=atoms.positions,
                reference_symbols=atoms.get_chemical_symbols(),
            )
        except ValueError:
            # Energy/force Q-Chem jobs do not print Hessians.  Leave normal ASE
            # behavior unchanged for those calculations.
            return
        self.last_hessian_data = data
        self.results["hessian"] = data.cartesian_hessian_ev_a2

    def get_hessian(self, atoms: Any | None = None) -> np.ndarray:
        """Return an analytic Cartesian Hessian in ``eV/A^2``.

        A separate Q-Chem frequency calculation is always run so that Hessian
        evaluations in optimizers and workflows do not accidentally reuse stale
        energy/force results.
        """

        if atoms is None:
            atoms = getattr(self, "atoms", None)
        if atoms is None:
            raise ValueError("get_hessian() requires an ASE Atoms object")

        calc = self._frequency_calculator()
        calc.calculate(atoms, properties=["hessian"], system_changes=all_changes)
        hessian = calc.results.get("hessian")
        if hessian is None:
            raise RuntimeError(
                f"Q-Chem frequency calculation did not produce a Hessian; see {calc.label}.out"
            )
        self.last_hessian_data = calc.last_hessian_data
        return np.asarray(hessian, dtype=float)

    def _frequency_calculator(self) -> "QChemAnalyticHessian":
        rem = dict(self.parameters)
        rem["jobtype"] = "freq"
        rem.pop("vibman_print", None)
        _with_default_rem(rem, "no_reorient", True)

        return type(self)(
            label=f"{self.label}{self.hessian_label_suffix}",
            scratch=self.hessian_scratch if self.hessian_scratch is not None else self.qchem_scratch,
            np=self.qchem_np,
            nt=self.qchem_nt,
            pbs=self.qchem_pbs,
            basisfile=self.basisfile,
            ecpfile=self.ecpfile,
            vibman_print=self.vibman_print,
            hessian_label_suffix=self.hessian_label_suffix,
            hessian_scratch=self.hessian_scratch,
            **dict(rem),
        )
