# qchem-hessian

An extension of ASE's `ase.calculators.qchem.QChem` calculator that adds the
`hessian` property, computed by a Q-Chem `JOBTYPE FREQ` job instead of by finite
differences of forces.

Energies and forces work exactly as in the ASE calculator. A Hessian request runs
a FREQ job, parses Q-Chem's mass-weighted Hessian, removes the mass weighting with
the masses Q-Chem used, and returns a Cartesian Hessian:

- shape `(3N, 3N)`, atom-major (`x1, y1, z1, x2, ...`), units eV/Å²;
- in the Cartesian frame of the ASE `Atoms` object (the output geometry is checked
  against `atoms`, and any reorientation by Q-Chem is undone);
- unprojected (translations/rotations are not removed), as needed by optimizers;
- cached with ASE's normal state checking, so it is reused until the atoms change.

## Install

```bash
python -m pip install -e .            # add [dev] for pytest, ruff, mypy
```

Q-Chem must be on `PATH` as `qchem` (the same requirement as ASE's calculator).

## Use

```python
from ase.io import read
from qchem_hessian import QChemHessian

atoms = read("ts_guess.xyz")
atoms.calc = QChemHessian(
    label="qchem/ts_guess",
    method="wb97m-v",
    basis="def2-tzvpd",
    nt=8,
    charge=0,
    multiplicity=1,
    scf_convergence=8,
    thresh=12,
)

energy = atoms.get_potential_energy()  # FORCE/SP job, as in ASE
hessian = atoms.calc.get_hessian(atoms)  # FREQ job -> (3N, 3N) eV/Å²
same = atoms.calc.get_property("hessian", atoms)  # cached, no new job
vib = atoms.calc.get_vibrations(atoms)  # ase.vibrations.VibrationsData
print(vib.get_frequencies()[:3])
```

Parsed extras (mass-weighted and projected Hessians, eigenvectors, frequencies,
Q-Chem masses, output geometry, whether Q-Chem reported an analytic Hessian) are in
`atoms.calc.hessian_data`. `parse_qchem_hessian_output()` can also be used on its
own for existing outputs.

### Files and options

The FREQ job is written to `<label>_freq.inp/.out` so it does not overwrite the
FORCE job files (`hessian_label_suffix=""` turns this off). `VIBMAN_PRINT`
defaults to 6 (minimum 4, which is needed to print the Hessian). `NO_REORIENT TRUE`
is added unless set. REM keywords are case-insensitive.

The FREQ job also gives the energy. Q-Chem does not print a gradient in FREQ jobs,
so `get_forces()` at the same geometry runs a separate FORCE job.

## Using the Hessian in other ASE-based tools

ASE has no standard Hessian property: `hessian` is not in
`ase.calculators.calculator.all_properties`, and trajectory and extxyz writers drop
it. This package follows the conventions that do exist:

- `calc.get_hessian(atoms)` returning `(3N, 3N)` in eV/Å², as in
  `ase.calculators.ff.ForceField` and several MLIP calculators;
- `"hessian"` in `implemented_properties`, so `calc.get_property("hessian", atoms)`
  works and is cached;
- `ase.vibrations.VibrationsData` via `get_vibrations()`.

For example, with Sella:

```python
from sella import Sella

opt = Sella(atoms, order=1, hessian_function=atoms.calc.get_hessian)
```

Store Hessians yourself (e.g. `np.save`) if you need them on disk.

## Notes and limitations

- Q-Chem prints the mass-weighted Hessian with six decimals. The Cartesian
  Hessian therefore carries an absolute rounding error of up to about
  `5e-7 · max(mass)` Eh/bohr² (6e-6 Eh/bohr² ≈ 6e-4 eV/Å² for C), which limits
  the accuracy of very soft modes, especially with heavy atoms.
- If a method has no analytic Hessian in Q-Chem, Q-Chem may compute it by finite
  differences internally. `hessian_data.analytic` records whether the output
  reports an analytic Hessian, and a warning is issued if not.

## Tests

```bash
python -m pytest
```

The tests use a fake `qchem` executable and a bundled HF/STO-3G output
(`tests/da.qcin.out`). The parsed Hessian is compared with an independent PySCF
analytic Hessian (`tests/data/make_pyscf_reference.py`).
