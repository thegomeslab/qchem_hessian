# qchem-analytic-hessian

Small ASE extension package that subclasses `ase.calculators.qchem.QChem` and adds an analytic Cartesian Hessian through a Q-Chem `JOBTYPE FREQ` calculation.

The main class is `qchem_analytic_hessian.QChemAnalyticHessian`. It keeps the normal ASE Q-Chem energy/force behavior, but adds:

- `get_hessian(atoms) -> np.ndarray`, returning a Cartesian Hessian in `eV/A^2`.
- `hessian` as an ASE calculator property for tools that call `calc.get_property("hessian", atoms)`.
- Parsing of Q-Chem's printed mass-weighted Hessian, projected mass-weighted Hessian, eigenvectors, frequencies, and Q-Chem masses when available.
- Automatic `JOBTYPE FREQ` and `VIBMAN_PRINT >= 4` for Hessian jobs; the default is `vibman_print=6`.
- Default `NO_REORIENT TRUE` so the Hessian is reported in the same Cartesian frame as the ASE `Atoms` object.

## Install

From this directory:

```bash
python -m pip install -e .
```

Install in the same environment as `mliptools` so that `mliptools --calculator qchem` can import the class.

## Minimal use

```python
from ase.io import read
from qchem_analytic_hessian import QChemAnalyticHessian

atoms = read("ts_guess.xyz")
atoms.calc = QChemAnalyticHessian(
    label="qchem/ts_guess",
    method="wb97m-v",
    basis="def2-tzvpd",
    nt=8,
    charge=0,
    multiplicity=1,
    scf_convergence=8,
    thresh=12,
)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
hessian_ev_a2 = atoms.calc.get_hessian(atoms)
```

For `mliptools`, use JSON calculator kwargs:

```bash
mliptools freq ts_guess.xyz \
  --calculator qchem \
  --calculator-kwargs '{"method":"hf", "basis":"sto-3g", "nt":8, "label":"qchem/freq"}'
```

or in a workflow YAML:

```yaml
calculator: qchem
calculator_kwargs:
  method: wb97m-v
  basis: def2-tzvpd
  nt: 8
  label: qchem/run
  scf_convergence: 8
  thresh: 12
  vibman_print: 6
```

## Notes

`get_hessian()` runs a separate frequency calculation with a label suffix, so a calculator with label `qchem/run` writes Hessian inputs/outputs as `qchem/run_freq.inp` and `qchem/run_freq.out` by default. The unprojected mass-weighted Hessian is converted back to a Cartesian Hessian using the masses printed by Q-Chem when those masses are available.

If a Q-Chem method does not have an analytic Hessian, Q-Chem may fall back to numerical derivatives internally. This package does not force a lower-level finite-difference calculation in Python; it delegates the Hessian calculation to Q-Chem.
