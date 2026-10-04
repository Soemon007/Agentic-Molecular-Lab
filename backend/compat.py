"""Import-time shims so PyTDC 1.1.15 runs on Python 3.13 / current RDKit.

Omnigent needs Python >=3.12, but PyTDC pins rdkit<2024.3.1 and an old scikit-learn. We
install PyTDC with --no-deps and shim the one removed RDKit module it imports (`rdkit.six`).
numpy must stay <2.4: numpy 2.5 makes `float(array([x]))` an error, which TDC's drd2()
swallows and turns into a silent 0.0 score.
Import this module before importing `tdc`.
"""
import sys
import types

if "rdkit.six" not in sys.modules:
    try:
        import rdkit.six  # noqa: F401
    except ImportError:
        six = types.ModuleType("rdkit.six")
        six.iteritems = lambda d: iter(d.items())
        sys.modules["rdkit.six"] = six
        import rdkit
        rdkit.six = six
