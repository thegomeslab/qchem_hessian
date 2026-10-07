from pathlib import Path

import numpy as np
import pytest
from ase.calculators.calculator import CalculationFailed
from ase.io import read
from ase.vibrations import VibrationsData

from qchem_hessian import QChemHessian
from qchem_hessian.parser import HESSIAN_HA_BOHR2_TO_EV_A2

HERE = Path(__file__).parent
REF_AU = np.loadtxt(HERE / "data" / "da_pyscf_hf_sto3g_hessian_au.txt")
REF_EV_A2 = REF_AU * HESSIAN_HA_BOHR2_TO_EV_A2
TOL_EV_A2 = 1e-5 * HESSIAN_HA_BOHR2_TO_EV_A2


@pytest.fixture
def atoms():
    return read(HERE / "data" / "da_input.xyz")


def make_calc(**kwargs):
    return QChemHessian(label="run/da", method="hf", basis="sto-3g", **kwargs)


def rem_lines(path):
    text = Path(path).read_text()
    return [line.split() for line in text.split("$rem")[1].split("$end")[0].strip().splitlines()]


def test_get_property_hessian_matches_reference(fake_qchem, atoms):
    atoms.calc = make_calc()
    hessian = atoms.calc.get_property("hessian", atoms)
    assert hessian.shape == (48, 48)
    np.testing.assert_allclose(hessian, REF_EV_A2, rtol=0, atol=TOL_EV_A2)
    np.testing.assert_array_equal(atoms.calc.get_hessian(atoms), hessian)


def test_hessian_is_cached_and_energy_comes_free(fake_qchem, atoms):
    atoms.calc = make_calc()
    atoms.calc.get_hessian(atoms)
    atoms.calc.get_hessian(atoms)
    energy = atoms.get_potential_energy()
    assert len(fake_qchem.calls) == 1
    assert "run/da_freq" not in fake_qchem.calls[0]  # ASE runs in the label directory
    assert "da_freq.inp da_freq.out" in fake_qchem.calls[0]
    assert np.isclose(energy, -230.0358576408 * 27.211386, rtol=1e-6)


def test_forces_after_hessian_run_force_job_and_keep_hessian(fake_qchem, atoms):
    atoms.calc = make_calc()
    atoms.calc.get_hessian(atoms)
    atoms.get_forces()
    assert len(fake_qchem.calls) == 2
    assert {"energy", "forces", "hessian"} <= set(atoms.calc.results)
    force_rem = rem_lines("run/da.inp")
    assert ["JOBTYPE", "FORCE"] in force_rem
    assert not any(line[0] == "VIBMAN_PRINT" for line in force_rem)


def test_moving_atoms_invalidates_hessian(fake_qchem, atoms):
    atoms.calc = make_calc(geometry_tolerance=1.0)
    atoms.calc.get_hessian(atoms)
    atoms.positions[0, 0] += 1e-3
    atoms.calc.get_hessian(atoms)
    assert len(fake_qchem.calls) == 2


def test_freq_input_rem(fake_qchem, atoms):
    atoms.calc = make_calc()
    atoms.calc.get_hessian(atoms)
    rem = rem_lines("run/da_freq.inp")
    keys = [line[0] for line in rem]
    assert len(keys) == len(set(keys))
    assert ["JOBTYPE", "FREQ"] in rem
    assert ["VIBMAN_PRINT", "6"] in rem
    assert ["NO_REORIENT", "TRUE"] in rem
    # The calculator's own parameters are unchanged afterwards.
    assert atoms.calc.parameters["jobtype"] is None
    assert "vibman_print" not in atoms.calc.parameters


def test_rem_keys_are_case_insensitive(fake_qchem, atoms):
    with pytest.raises(ValueError, match="given twice"):
        make_calc(JOBTYPE="force", jobtype="sp")
    atoms.calc = make_calc(JOBTYPE="force", VIBMAN_PRINT=5, NO_REORIENT=False)
    atoms.calc.get_hessian(atoms)
    rem = rem_lines("run/da_freq.inp")
    assert [line[0] for line in rem].count("JOBTYPE") == 1
    assert ["JOBTYPE", "FREQ"] in rem
    assert ["VIBMAN_PRINT", "5"] in rem
    assert ["NO_REORIENT", "FALSE"] in rem


def test_stale_or_foreign_output_raises(fake_qchem, atoms):
    atoms.positions[3] += 0.05  # the fake Q-Chem still returns the sample geometry
    atoms.calc = make_calc()
    with pytest.raises(CalculationFailed, match="stale output"):
        atoms.calc.get_hessian(atoms)


def test_abnormal_termination_raises(fake_qchem, atoms):
    fake_qchem.set_mode("crash")
    atoms.calc = make_calc()
    with pytest.raises(CalculationFailed, match="did not terminate normally"):
        atoms.calc.get_hessian(atoms)


def test_get_vibrations(fake_qchem, atoms):
    atoms.calc = make_calc()
    vib = atoms.calc.get_vibrations(atoms)
    assert isinstance(vib, VibrationsData)
    freqs = vib.get_frequencies()
    assert np.isclose(freqs[0].imag, 1055.42, atol=1.0)
    # Stiff modes agree with Q-Chem's (projected) frequencies.
    np.testing.assert_allclose(freqs[-3:].real, [3810.27, 3813.51, 3816.93], atol=0.5)


def test_deprecated_alias():
    import qchem_hessian

    with pytest.warns(DeprecationWarning, match="QChemHessian"):
        assert qchem_hessian.QChemAnalyticHessian is QChemHessian
    assert QChemHessian.name == "QChemHessian"
