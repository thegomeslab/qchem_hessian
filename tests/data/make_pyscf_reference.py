"""Regenerate the independent reference Hessian used by tests/test_parser.py.

HF/STO-3G analytic Hessian from PySCF at the *input* (ASE-frame) geometry of
tests/da.qcin.out.  Units: Hartree/Bohr^2, shape (3N, 3N), atom-major.

    python tests/data/make_pyscf_reference.py   # needs pyscf; ~1 min
"""

from pathlib import Path

import numpy as np
from ase.io import read
from pyscf import gto, scf
from pyscf.hessian import rhf

here = Path(__file__).parent
atoms = read(here / "da_input.xyz")
symbols = atoms.get_chemical_symbols()
mol = gto.M(
    atom=[(s, tuple(p)) for s, p in zip(symbols, atoms.positions, strict=True)],
    basis="sto-3g",
    unit="Angstrom",
    verbose=0,
)
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
energy = mf.kernel()
assert abs(energy - (-230.0358576408)) < 1e-8, energy  # Q-Chem SCF energy in da.qcin.out
n = len(atoms)
hessian = rhf.Hessian(mf).kernel().transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
np.savetxt(
    here / "da_pyscf_hf_sto3g_hessian_au.txt",
    hessian,
    fmt="%.17e",
    header="PySCF HF/STO-3G analytic Hessian, Eh/bohr^2, (48, 48), atom-major, "
    "ASE/input frame of da_input.xyz",
)
print(f"E = {energy:.10f} Eh; saved {hessian.shape}")
