"""Parsers for Q-Chem frequency-output Hessian sections."""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import units
from ase.data import atomic_masses_common, atomic_numbers

HESSIAN_HA_BOHR2_TO_EV_A2 = units.Hartree / units.Bohr**2

_MW_MARKER = "Mass-Weighted Hessian Matrix:"
_PROJ_MARKER = "Projected Mass-Weighted Hessian Matrix:"
_EVEC_MARKER = "Eigenvectors of Proj. Mass-Weighted Hessian Matrix:"
_GEOM_MARKER = "Standard Nuclear Orientation"
_ANALYTIC_MARKER = "Calculating analytic Hessian"
_NORMAL_END = "Thank you very much for using Q-Chem"

_NUM = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"
_FLOAT_RE = re.compile(_NUM)
_NUMERIC_LINE_RE = re.compile(rf"^\s*{_NUM}(?:\s+{_NUM})*\s*$")
_MASS_RE = re.compile(
    rf"Atom\s+(?P<index>\d+)\s+Element\s+(?P<element>[A-Za-z]+)\s+Has\s+Mass\s+(?P<mass>{_NUM})"
)
_GEOM_ROW_RE = re.compile(
    rf"^\s*(?P<index>\d+)\s+(?P<symbol>[A-Za-z]+)\s+({_NUM})\s+({_NUM})\s+({_NUM})\s*$"
)


@dataclass(frozen=True)
class QChemHessianData:
    """Structured Hessian data parsed from a Q-Chem frequency output.

    ``cartesian_hessian_ev_a2`` is in the frame of the reference structure
    passed to the parser (``rotation`` maps output-frame coordinates onto it,
    ``x_ref = x_out @ rotation``).  All other matrices are in the frame of the
    Q-Chem output (``output_positions``).
    """

    cartesian_hessian_ev_a2: np.ndarray
    mass_weighted_hessian_au: np.ndarray
    projected_mass_weighted_hessian_au: np.ndarray | None
    projected_eigenvectors: np.ndarray | None
    frequencies_cm1: np.ndarray
    qchem_masses_amu: np.ndarray
    symbols: tuple[str, ...]
    output_positions: np.ndarray
    rotation: np.ndarray
    analytic: bool
    source_output: Path


def _floats(line: str) -> list[float]:
    return [float(token.replace("D", "E").replace("d", "e")) for token in _FLOAT_RE.findall(line)]


def _last_index(
    lines: list[str], match: Callable[[str], bool], before: int | None = None
) -> int | None:
    stop = len(lines) if before is None else before
    for i in range(stop - 1, -1, -1):
        if match(lines[i]):
            return i
    return None


def _is_unprojected_mw(line: str) -> bool:
    # The projected-Hessian and eigenvector headers contain the same substring.
    return _MW_MARKER in line and _PROJ_MARKER not in line and _EVEC_MARKER not in line


def _parse_block_matrix(lines: list[str], start: int, size: int, name: str) -> np.ndarray:
    """Parse a Q-Chem dense matrix printed in column blocks (six columns per block).

    ``start`` is the index of the header line.  The matrix is the sequence of
    runs of purely numeric lines that follows; it ends at the first non-blank,
    non-numeric line.  Every run must have exactly ``size`` rows of equal
    width and the widths must add up to ``size``, so a wrong atom count or a
    truncated output raises instead of silently reading into the next section.
    """

    runs: list[list[list[float]]] = []
    current: list[list[float]] = []
    for line in lines[start + 1 :]:
        if not line.strip():
            if current:
                runs.append(current)
                current = []
            continue
        if not _NUMERIC_LINE_RE.match(line):
            break
        current.append(_floats(line))
    if current:
        runs.append(current)

    matrix = np.empty((size, size), dtype=float)
    col = 0
    for run in runs:
        width = len(run[0])
        if len(run) != size or any(len(row) != width for row in run):
            raise ValueError(
                f"{name}: column block starting at column {col + 1} has {len(run)} rows "
                f"(expected {size}) or ragged rows; wrong atom count or truncated output"
            )
        if col + width > size:
            raise ValueError(f"{name}: more than {size} columns; wrong atom count")
        matrix[:, col : col + width] = np.asarray(run)
        col += width
    if col != size:
        raise ValueError(f"{name}: found {col} columns, expected {size}")
    return matrix


def _parse_geometry(lines: list[str], start: int) -> tuple[tuple[str, ...], np.ndarray]:
    symbols: list[str] = []
    positions: list[list[float]] = []
    for line in lines[start + 1 :]:
        match = _GEOM_ROW_RE.match(line)
        if match:
            symbols.append(match.group("symbol").capitalize())
            positions.append([float(x) for x in match.groups()[2:5]])
        elif positions and line.strip().startswith("-"):
            break
    return tuple(symbols), np.asarray(positions, dtype=float)


def _parse_qchem_masses(lines: list[str], start: int, natoms: int) -> np.ndarray | None:
    masses = np.full(natoms, np.nan, dtype=float)
    for line in lines[start:]:
        match = _MASS_RE.search(line)
        if match is None:
            continue
        index = int(match.group("index")) - 1
        if 0 <= index < natoms:
            masses[index] = float(match.group("mass").replace("D", "E"))
    if np.all(np.isfinite(masses)):
        return masses
    return None


def _parse_frequencies(lines: list[str], start: int) -> np.ndarray:
    values: list[float] = []
    for line in lines[start:]:
        if line.strip().startswith("Frequency:"):
            values.extend(_floats(line))
    return np.asarray(values, dtype=float)


def _mass_weighted_to_cartesian_ev_a2(
    mw_hessian_au: np.ndarray, masses_amu: np.ndarray
) -> np.ndarray:
    root_m = np.sqrt(np.repeat(np.asarray(masses_amu, dtype=float), 3))
    cart_ha_bohr2 = mw_hessian_au * np.outer(root_m, root_m)
    return 0.5 * (cart_ha_bohr2 + cart_ha_bohr2.T) * HESSIAN_HA_BOHR2_TO_EV_A2


def _kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Proper rotation R minimising |(mobile - c_m) @ R - (target - c_t)|; returns (R, RMSD)."""

    a = mobile - mobile.mean(axis=0)
    b = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    rmsd = float(np.sqrt(np.mean(np.sum((a @ rot - b) ** 2, axis=1))))
    return rot, rmsd


def rotate_cartesian_hessian(hessian: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Re-express a (3N, 3N) Hessian from frame A in frame B, where ``x_B = x_A @ rotation``."""

    big = np.kron(np.eye(hessian.shape[0] // 3), rotation)
    return big.T @ hessian @ big


def parse_qchem_hessian_output(
    path: str | Path,
    *,
    natoms: int | None = None,
    reference_positions: np.ndarray | None = None,
    reference_symbols: list[str] | tuple[str, ...] | None = None,
    geometry_tolerance: float = 1e-4,
    fallback_masses_amu: np.ndarray | None = None,
) -> QChemHessianData:
    """Parse Hessian data from the last frequency job in a Q-Chem output.

    Parameters
    ----------
    path
        Q-Chem ``.out`` file from a frequency calculation with ``VIBMAN_PRINT``
        of at least 4.
    natoms
        Optional consistency check.  The atom count is always taken from the
        ``Standard Nuclear Orientation`` table that precedes the Hessian.
    reference_positions, reference_symbols
        Geometry (Angstrom) and element order of the structure the Hessian is
        for, usually ``atoms.positions`` and ``atoms.get_chemical_symbols()``.
        The output geometry is aligned to the reference.  An aligned RMSD above
        ``geometry_tolerance`` (Angstrom) raises, which catches stale or foreign
        output files.  If Q-Chem reoriented the molecule, the Cartesian Hessian
        is rotated back into the reference frame.  Without a reference the
        Cartesian Hessian is in Q-Chem's output frame.
    fallback_masses_amu
        Masses used only if the output has no ``Atom ... Has Mass`` lines.  If
        omitted, most-abundant-isotope masses (Q-Chem's default) are used.
    """

    output_path = Path(path)
    lines = output_path.read_text(errors="replace").splitlines()

    mw_at = _last_index(lines, _is_unprojected_mw)
    if mw_at is None:
        raise ValueError(
            f"{output_path} does not contain {_MW_MARKER!r}. "
            "Set VIBMAN_PRINT = 4 or higher for frequency calculations."
        )

    geom_at = _last_index(lines, lambda line: _GEOM_MARKER in line, before=mw_at)
    if geom_at is None:
        raise ValueError(f"{output_path}: no {_GEOM_MARKER!r} table before the Hessian")
    symbols, out_pos = _parse_geometry(lines, geom_at)
    if natoms is not None and int(natoms) != len(symbols):
        raise ValueError(f"natoms={natoms} but the Q-Chem output has {len(symbols)} atoms")
    n = len(symbols)
    size = 3 * n

    mw = _parse_block_matrix(lines, mw_at, size, _MW_MARKER)
    if not np.allclose(mw, mw.T, rtol=0.0, atol=2e-6):
        raise ValueError(f"{output_path}: {_MW_MARKER} is not symmetric; parse error")
    mw = 0.5 * (mw + mw.T)

    projected = evecs = None
    proj_at = _last_index(lines, lambda line: _PROJ_MARKER in line)
    if proj_at is not None and proj_at > mw_at:
        projected = _parse_block_matrix(lines, proj_at, size, _PROJ_MARKER)
        projected = 0.5 * (projected + projected.T)
    evec_at = _last_index(lines, lambda line: _EVEC_MARKER in line)
    if evec_at is not None and evec_at > mw_at:
        evecs = _parse_block_matrix(lines, evec_at, size, _EVEC_MARKER)

    masses = _parse_qchem_masses(lines, mw_at, n)
    if masses is None:
        if fallback_masses_amu is None:
            masses = np.array([atomic_masses_common[atomic_numbers[s]] for s in symbols])
        else:
            masses = np.asarray(fallback_masses_amu, dtype=float)
            if masses.shape != (n,):
                raise ValueError(f"fallback_masses_amu has shape {masses.shape}; expected {(n,)}")
        warnings.warn(
            f"{output_path}: Q-Chem masses not printed; un-mass-weighting with fallback "
            "masses. The Cartesian Hessian is only correct if these are the masses Q-Chem used.",
            stacklevel=2,
        )

    hessian = _mass_weighted_to_cartesian_ev_a2(mw, masses)

    rotation = np.eye(3)
    if reference_symbols is not None and tuple(reference_symbols) != symbols:
        raise ValueError(f"{output_path}: element order differs from the reference structure")
    if reference_positions is not None:
        ref = np.asarray(reference_positions, dtype=float)
        if ref.shape != out_pos.shape:
            raise ValueError(
                f"{output_path}: output has {n} atoms, reference structure has {len(ref)}"
            )
        rotation, rmsd = _kabsch(out_pos, ref)
        if rmsd > geometry_tolerance:
            raise ValueError(
                f"{output_path}: output geometry differs from the reference structure "
                f"(aligned RMSD {rmsd:.2e} A > {geometry_tolerance:.1e} A); stale output?"
            )
        if np.allclose(rotation, np.eye(3), rtol=0.0, atol=1e-10):
            rotation = np.eye(3)
        else:
            hessian = rotate_cartesian_hessian(hessian, rotation)

    analytic = any(_ANALYTIC_MARKER in line for line in lines[:mw_at])
    if not analytic:
        warnings.warn(
            f"{output_path}: no {_ANALYTIC_MARKER!r} line; Q-Chem may have used finite "
            "differences of gradients.",
            stacklevel=2,
        )

    return QChemHessianData(
        cartesian_hessian_ev_a2=hessian,
        mass_weighted_hessian_au=mw,
        projected_mass_weighted_hessian_au=projected,
        projected_eigenvectors=evecs,
        frequencies_cm1=_parse_frequencies(lines, mw_at),
        qchem_masses_amu=masses,
        symbols=symbols,
        output_positions=out_pos,
        rotation=rotation,
        analytic=analytic,
        source_output=output_path,
    )


def output_terminated_normally(path: str | Path) -> bool:
    """True if a Q-Chem output ends with Q-Chem's normal-termination banner."""

    return _NORMAL_END in Path(path).read_text(errors="replace")
