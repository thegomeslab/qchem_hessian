"""ASE Q-Chem calculator subclass with Hessian support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import CalculationFailed, FileIOCalculator, all_changes
from ase.calculators.qchem import QChem
from ase.vibrations import VibrationsData

from qchem_hessian.parser import (
    QChemHessianData,
    output_terminated_normally,
    parse_qchem_hessian_output,
)

# Parameters that ASE's QChem.write_input() writes into $molecule as integers.
_MOLECULE_KEYS = {"charge", "multiplicity"}


def _rem_value(value: Any) -> Any:
    """Return a value compatible with ASE's QChem.write_input (which calls ``.upper()``)."""

    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class QChemHessian(QChem):
    """ASE Q-Chem calculator that also provides the ``hessian`` property.

    Energies and forces behave exactly as in :class:`ase.calculators.qchem.QChem`.
    Requesting ``hessian`` (via :meth:`get_hessian`, ``get_property("hessian")``
    or ``calculate(properties=[..., "hessian"])``) runs a Q-Chem
    ``JOBTYPE FREQ`` job written to ``<label><hessian_label_suffix>.inp/.out``
    and stores a ``(3N, 3N)`` Cartesian Hessian in eV/Å² in the ASE frame of
    the ``Atoms`` object.  The result is cached with ASE's normal state
    checking, so it is reused until the atoms change.

    A calculator constructed with ``jobtype="freq"`` runs every job this way.
    The FREQ job also yields the energy; it does not print a gradient, so a
    later ``get_forces()`` call at the same geometry runs a FORCE job.

    REM keywords are case-insensitive (``JOBTYPE`` and ``jobtype`` are the same
    key).  Non-string values are converted to strings for ASE.
    """

    name = "QChemHessian"
    implemented_properties = [*QChem.implemented_properties, "hessian"]

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
        atoms: Atoms | None = None,
        *,
        vibman_print: int = 6,
        hessian_label_suffix: str = "_freq",
        hessian_scratch: str | None = None,
        no_reorient: bool = True,
        geometry_tolerance: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        """
        Parameters (in addition to :class:`ase.calculators.qchem.QChem`)
        ----------------------------------------------------------------
        vibman_print
            ``VIBMAN_PRINT`` for FREQ jobs; raised to at least 4, which Q-Chem
            needs to print the mass-weighted Hessian.
        hessian_label_suffix
            Suffix added to ``label`` for the FREQ job files.  Use ``""`` to
            write them to ``<label>.inp/.out``.
        hessian_scratch
            Q-Chem save name for the FREQ job.  Defaults to ``scratch``.
        no_reorient
            Add ``NO_REORIENT TRUE`` unless the REM already sets it.  The
            parser also undoes any reorientation it detects.
        geometry_tolerance
            Maximum aligned RMSD (Å) between the FREQ output geometry and the
            ``Atoms`` object; larger differences raise ``CalculationFailed``.
        """

        self.vibman_print = int(vibman_print)
        self.hessian_label_suffix = hessian_label_suffix
        self.hessian_scratch = hessian_scratch
        self.qchem_scratch = scratch
        self.geometry_tolerance = float(geometry_tolerance)
        self.hessian_data: QChemHessianData | None = None

        rem = dict(kwargs)
        if no_reorient and not any(key.lower() == "no_reorient" for key in rem):
            rem["no_reorient"] = True
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
            **rem,
        )

    # ------------------------------------------------------------------ parameters
    def set(self, **kwargs: Any) -> dict[str, Any]:
        """Lower-case REM keys, reject case-duplicates, and stringify values for ASE."""

        normalized: dict[str, Any] = {}
        for key, value in kwargs.items():
            low = key.lower()
            if low in normalized:
                raise ValueError(f"REM keyword {low!r} given twice (keys are case-insensitive)")
            normalized[low] = value
        if "vibman_print" in normalized:
            value = normalized.pop("vibman_print")
            if value is not None:
                self.vibman_print = int(value)
        for key, value in normalized.items():
            if key not in _MOLECULE_KEYS:
                normalized[key] = _rem_value(value)
        return super().set(**normalized)

    # ------------------------------------------------------------------ running jobs
    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | tuple[str, ...] = ("energy",),
        system_changes: list[str] | tuple[str, ...] = tuple(all_changes),
    ) -> None:
        user_freq = str(self.parameters.get("jobtype") or "").lower() == "freq"
        if "hessian" not in properties and not user_freq:
            super().calculate(atoms, properties, system_changes)
            return

        saved_prefix = self.prefix
        saved_parameters = dict(self.parameters)
        saved_command = self.command
        try:
            self.parameters["jobtype"] = "freq"
            self.parameters["vibman_print"] = str(max(self.vibman_print, 4))
            self.prefix = f"{saved_prefix}{self.hessian_label_suffix}"
            if self.hessian_scratch is not None and self.command is not None:
                base = self.command
                if self.qchem_scratch is not None:
                    base = base.removesuffix(f" {self.qchem_scratch}")
                self.command = f"{base} {self.hessian_scratch}"
            super().calculate(atoms, properties, system_changes)
        finally:
            self.prefix = saved_prefix
            self.parameters.clear()
            self.parameters.update(saved_parameters)
            self.command = saved_command

    def read_results(self) -> None:
        """Read energy/forces as ASE does; for FREQ jobs also read the Hessian."""

        super().read_results()
        if str(self.parameters.get("jobtype") or "").lower() != "freq":
            return
        output = Path(f"{self.label}.out")
        if not output_terminated_normally(output):
            raise CalculationFailed(f"Q-Chem FREQ job did not terminate normally; see {output}")
        try:
            data = parse_qchem_hessian_output(
                output,
                reference_positions=self.atoms.positions,
                reference_symbols=self.atoms.get_chemical_symbols(),
                geometry_tolerance=self.geometry_tolerance,
            )
        except ValueError as err:
            raise CalculationFailed(str(err)) from err
        self.hessian_data = data
        self.results["hessian"] = data.cartesian_hessian_ev_a2

    # ------------------------------------------------------------------ public API
    def get_hessian(self, atoms: Atoms | None = None) -> np.ndarray:
        """Cartesian Hessian, shape ``(3N, 3N)``, eV/Å², atom-major (x1, y1, z1, x2, ...).

        Same convention as ``ase.calculators.ff.ForceField.get_hessian`` and
        Sella's ``hessian_function``.  Equivalent to
        ``self.get_property("hessian", atoms)``.
        """

        return self.get_property("hessian", atoms)

    def get_vibrations(self, atoms: Atoms | None = None) -> VibrationsData:
        """Return an :class:`ase.vibrations.VibrationsData` built from the Q-Chem Hessian.

        The atoms carry the masses Q-Chem used (most abundant isotopes), so
        frequencies are comparable to Q-Chem's.  ``VibrationsData`` does not
        project out translations/rotations; at non-stationary geometries
        those six modes are not exactly zero.
        """

        hessian = self.get_hessian(atoms)
        vib_atoms = (atoms if atoms is not None else self.atoms).copy()
        vib_atoms.calc = None
        assert self.hessian_data is not None
        vib_atoms.set_masses(self.hessian_data.qchem_masses_amu)
        return VibrationsData.from_2d(vib_atoms, hessian)

    @property
    def last_hessian_data(self) -> QChemHessianData | None:
        """Backward-compatible alias for :attr:`hessian_data`."""

        return self.hessian_data
