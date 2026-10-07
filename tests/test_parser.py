from pathlib import Path

import numpy as np

from qchem_analytic_hessian.parser import parse_qchem_hessian_output

SAMPLE = Path(__file__).with_name("da.qcin.out")


def test_parse_sample_qchem_output():
    data = parse_qchem_hessian_output(SAMPLE, natoms=16)
    assert data.cartesian_hessian_ev_a2.shape == (48, 48)
    assert data.mass_weighted_hessian_au.shape == (48, 48)
    assert data.projected_mass_weighted_hessian_au is not None
    assert data.projected_eigenvectors is not None
    assert data.frequencies_cm1.size == 42
    assert np.isclose(data.frequencies_cm1[0], -1055.42, atol=0.05)
