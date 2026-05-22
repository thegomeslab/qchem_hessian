"""Parsers for Q-Chem frequency-output Hessian sections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

# CODATA values used by ASE closely enough for Hessian conversion.  Keeping the
# constants here makes the parser testable without importing ASE.
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903
HESSIAN_HA_BOHR2_TO_EV_A2 = HARTREE_TO_EV / BOHR_TO_ANG**2

_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?")
_MASS_RE = re.compile(
    r"Atom\s+(?P<index>\d+)\s+Element\s+(?P<element>[A-Za-z]+)\s+Has\s+Mass\s+"
    r"(?P<mass>[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?)"
)


@dataclass(frozen=True)
class QChemHessianData:
    """Structured Hessian data parsed from a Q-Chem frequency output."""

    cartesian_hessian_ev_a2: np.ndarray
    mass_weighted_hessian_au: np.ndarray
    projected_mass_weighted_hessian_au: np.ndarray | None
    projected_eigenvectors: np.ndarray | None
    frequencies_cm1: np.ndarray
    qchem_masses_amu: np.ndarray
    source_output: Path


def _floats(line: str) -> list[float]:
    return [float(token.replace("D", "E").replace("d", "e")) for token in _FLOAT_RE.findall(line)]


def _parse_block_matrix(lines: list[str], marker: str, size: int) -> np.ndarray | None:
    """Parse a Q-Chem dense matrix printed in row blocks of up to six columns."""

    try:
        start = next(i for i, line in enumerate(lines) if marker in line) + 1
    except StopIteration:
        return None

    matrix = np.empty((size, size), dtype=float)
    col = 0
    i = start
    while col < size:
        while i < len(lines) and not _floats(lines[i]):
            i += 1
        block_width = min(6, size - col)
        row = 0
        while row < size and i < len(lines):
            values = _floats(lines[i])
            if values:
                # Some Q-Chem matrices include a leading row label.  The Hessian
                # blocks in Q-Chem 6 usually do not, but keeping the last values
                # makes the parser robust to either form.
                if len(values) > block_width:
                    values = values[-block_width:]
                if len(values) != block_width:
                    raise ValueError(
                        f"Could not parse {marker!r} block at output line {i + 1}: "
                        f"expected {block_width} values, got {len(values)}"
                    )
                matrix[row, col : col + block_width] = values
                row += 1
            i += 1
        if row != size:
            raise ValueError(
                f"Could not parse {marker!r}: expected {size} rows in block starting "
                f"at column {col + 1}, got {row}"
            )
        col += block_width
    return 0.5 * (matrix + matrix.T)


def _parse_qchem_masses(lines: list[str], natoms: int) -> np.ndarray | None:
    masses = np.full(natoms, np.nan, dtype=float)
    for line in lines:
        match = _MASS_RE.search(line)
        if match is None:
            continue
        index = int(match.group("index")) - 1
        if 0 <= index < natoms:
            masses[index] = float(match.group("mass").replace("D", "E"))
    if np.all(np.isfinite(masses)):
        return masses
    return None


def _parse_frequencies(lines: list[str]) -> np.ndarray:
    values: list[float] = []
    for line in lines:
        if line.strip().startswith("Frequency:"):
            values.extend(_floats(line))
    return np.asarray(values, dtype=float)


def _mass_weighted_to_cartesian_ev_a2(
    mw_hessian_au: np.ndarray,
    masses_amu: np.ndarray,
) -> np.ndarray:
    masses_cart = np.repeat(np.asarray(masses_amu, dtype=float), 3)
    cart_ha_bohr2 = mw_hessian_au * np.sqrt(np.outer(masses_cart, masses_cart))
    return 0.5 * (cart_ha_bohr2 + cart_ha_bohr2.T) * HESSIAN_HA_BOHR2_TO_EV_A2


def parse_qchem_hessian_output(
    path: str | Path,
    *,
    natoms: int,
    fallback_masses_amu: np.ndarray | None = None,
) -> QChemHessianData:
    """Parse analytic Hessian data from a Q-Chem frequency output.

    Parameters
    ----------
    path
        Q-Chem ``.out`` file from a frequency calculation with ``VIBMAN_PRINT``
        high enough to print Hessian matrices.
    natoms
        Number of atoms in the ASE structure.  The Hessian size is ``3*natoms``.
    fallback_masses_amu
        Masses to use if Q-Chem masses are not printed.  The unprojected
        mass-weighted Hessian is converted to a Cartesian Hessian with these
        masses only when the output does not contain ``Atom ... Has Mass`` lines.
    """

    output_path = Path(path)
    lines = output_path.read_text(errors="replace").splitlines()
    size = 3 * int(natoms)

    mw = _parse_block_matrix(lines, "Mass-Weighted Hessian Matrix:", size)
    if mw is None:
        raise ValueError(
            "Q-Chem output does not contain 'Mass-Weighted Hessian Matrix:'. "
            "Set VIBMAN_PRINT = 4 or higher for frequency calculations."
        )

    projected = _parse_block_matrix(lines, "Projected Mass-Weighted Hessian Matrix:", size)
    evecs = _parse_block_matrix(lines, "Eigenvectors of Proj. Mass-Weighted Hessian Matrix:", size)

    qchem_masses = _parse_qchem_masses(lines, int(natoms))
    if qchem_masses is None:
        if fallback_masses_amu is None:
            raise ValueError(
                "Q-Chem masses were not found in the output and no fallback masses were supplied."
            )
        qchem_masses = np.asarray(fallback_masses_amu, dtype=float)
        if qchem_masses.shape != (int(natoms),):
            raise ValueError(
                f"fallback_masses_amu has shape {qchem_masses.shape}; expected {(int(natoms),)}"
            )

    return QChemHessianData(
        cartesian_hessian_ev_a2=_mass_weighted_to_cartesian_ev_a2(mw, qchem_masses),
        mass_weighted_hessian_au=mw,
        projected_mass_weighted_hessian_au=projected,
        projected_eigenvectors=evecs,
        frequencies_cm1=_parse_frequencies(lines),
        qchem_masses_amu=qchem_masses,
        source_output=output_path,
    )
