from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from qchem_analytic_hessian.parser import (
    HESSIAN_HA_BOHR2_TO_EV_A2,
    _floats,
    parse_qchem_hessian_output,
    rotate_cartesian_hessian,
)

HERE = Path(__file__).parent
SAMPLE = HERE / "da.qcin.out"
ATOMS = read(HERE / "data" / "da_input.xyz")
# Independent PySCF HF/STO-3G Hessian at the input geometry (tests/data/make_pyscf_reference.py)
REF_AU = np.loadtxt(HERE / "data" / "da_pyscf_hf_sto3g_hessian_au.txt")
# Q-Chem prints the mass-weighted Hessian with 6 decimals; the largest Cartesian rounding
# error is 5e-7 * max(mass) = 6e-6 Eh/bohr^2 for this C/H system.
PRINT_TOL_AU = 1e-5


def parse(**kwargs):
    return parse_qchem_hessian_output(SAMPLE, **kwargs)


def parse_in_ase_frame(path=SAMPLE, **kwargs):
    return parse_qchem_hessian_output(
        path,
        reference_positions=ATOMS.positions,
        reference_symbols=ATOMS.get_chemical_symbols(),
        **kwargs,
    )


def test_shapes_and_frequencies():
    data = parse(natoms=16)
    assert data.symbols == tuple(ATOMS.get_chemical_symbols())
    assert data.cartesian_hessian_ev_a2.shape == (48, 48)
    assert data.mass_weighted_hessian_au.shape == (48, 48)
    assert data.projected_mass_weighted_hessian_au is not None
    assert data.projected_eigenvectors is not None
    assert data.frequencies_cm1.size == 42
    assert np.isclose(data.frequencies_cm1[0], -1055.42, atol=0.05)
    assert data.analytic
    np.testing.assert_allclose(data.qchem_masses_amu, [12.0] * 6 + [1.00783] * 10)


def test_mass_weighted_units_match_qchem_eigenvalues():
    """Eigenvalues of the parsed projected MW Hessian equal Q-Chem's printed a.u. values."""
    lines = SAMPLE.read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if "Vibrational Frequencies in atomic units" in line)
    printed = np.sort(np.concatenate([_floats(line) for line in lines[i + 1 : i + 9]]))
    parsed = np.linalg.eigvalsh(parse().projected_mass_weighted_hessian_au)
    np.testing.assert_allclose(parsed, printed, atol=2e-6)


def test_cartesian_hessian_matches_pyscf_in_ase_frame():
    data = parse_in_ase_frame()
    assert not np.allclose(data.rotation, np.eye(3))  # this sample was reoriented by Q-Chem
    np.testing.assert_allclose(
        data.cartesian_hessian_ev_a2 / HESSIAN_HA_BOHR2_TO_EV_A2, REF_AU, rtol=0, atol=PRINT_TOL_AU
    )


def test_without_reference_hessian_is_in_qchem_frame():
    rotation = parse_in_ase_frame().rotation
    ref_in_qchem_frame = rotate_cartesian_hessian(REF_AU, rotation.T)
    np.testing.assert_allclose(
        parse().cartesian_hessian_ev_a2 / HESSIAN_HA_BOHR2_TO_EV_A2,
        ref_in_qchem_frame,
        rtol=0,
        atol=PRINT_TOL_AU,
    )


def test_geometry_mismatch_raises():
    moved = ATOMS.positions.copy()
    moved[0, 0] += 0.01
    with pytest.raises(ValueError, match="stale output"):
        parse(reference_positions=moved)


def test_symbol_mismatch_raises():
    symbols = ATOMS.get_chemical_symbols()
    symbols[0], symbols[-1] = symbols[-1], symbols[0]
    with pytest.raises(ValueError, match="element order"):
        parse(reference_symbols=symbols)


def test_wrong_natoms_raises():
    with pytest.raises(ValueError, match="natoms"):
        parse(natoms=15)


def test_truncated_output_raises(tmp_path):
    lines = SAMPLE.read_text().splitlines()
    cut = next(k for k, line in enumerate(lines) if "Mass-Weighted Hessian Matrix:" in line) + 200
    bad = tmp_path / "truncated.out"
    bad.write_text("\n".join(lines[:cut]))
    with pytest.raises(ValueError, match="column"):
        parse_qchem_hessian_output(bad)


def test_missing_masses_falls_back_to_isotopic_masses(tmp_path):
    text = "\n".join(line for line in SAMPLE.read_text().splitlines() if "Has Mass" not in line)
    nomass = tmp_path / "nomass.out"
    nomass.write_text(text)
    with pytest.warns(UserWarning, match="masses not printed"):
        data = parse_in_ase_frame(nomass)
    # Most-abundant-isotope masses, not ASE's IUPAC averages (12.011 for C).
    assert data.qchem_masses_amu[0] == 12.0
    np.testing.assert_allclose(
        data.cartesian_hessian_ev_a2 / HESSIAN_HA_BOHR2_TO_EV_A2, REF_AU, rtol=0, atol=PRINT_TOL_AU
    )
