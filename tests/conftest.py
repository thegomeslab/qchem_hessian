"""A fake ``qchem`` executable so the calculator can be tested without Q-Chem.

FREQ inputs get the bundled sample output (tests/da.qcin.out); other jobs get a
minimal output with the sample's energy and, for FORCE jobs, a zero gradient.
Every call is appended to ``calls.log`` next to the executable.
"""

import os
import stat
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent

MOCK = r"""#!{python}
import re, sys
from pathlib import Path
args = [a for a in sys.argv[1:] if not a.startswith("-")]
inp, out = Path(args[0]), Path(args[1])
text = inp.read_text()
log = Path(__file__).with_name("calls.log")
with log.open("a") as f:
    f.write(" ".join(sys.argv[1:]) + "\n")
mode = Path(__file__).with_name("mode").read_text().strip()
if re.search(r"^\s*JOBTYPE\s+FREQ\s*$", text, re.I | re.M):
    body = Path({sample!r}).read_text()
    if mode == "crash":
        body = body.split("Thank you very much")[0]
    out.write_text(body)
else:
    natoms = len(text.split("$molecule")[1].split("$end")[0].strip().splitlines()) - 1
    lines = [" Total energy in the final basis set =     -230.0358576408"]
    if re.search(r"JOBTYPE\s+FORCE", text, re.I):
        lines.append(" Gradient of SCF Energy")
        for start in range(0, natoms, 6):
            cols = range(start, min(start + 6, natoms))
            lines.append("     " + "".join(f"{{c + 1:12d}}" for c in cols))
            for k in range(3):
                lines.append(f"{{k + 1:5d}}" + "".join(f"{{0.0:12.7f}}" for _ in cols))
        lines.append(" Max gradient component =       0.000E+00")
    out.write_text("\n".join(lines) + "\n")
"""


class FakeQChem:
    def __init__(self, bindir: Path):
        self.bindir = bindir

    def set_mode(self, mode: str) -> None:
        (self.bindir / "mode").write_text(mode)

    @property
    def calls(self) -> list[str]:
        log = self.bindir / "calls.log"
        return log.read_text().splitlines() if log.exists() else []


@pytest.fixture
def fake_qchem(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "qchem"
    exe.write_text(MOCK.format(python=sys.executable, sample=str(HERE / "da.qcin.out")))
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.chdir(tmp_path)
    fake = FakeQChem(bindir)
    fake.set_mode("ok")
    return fake
