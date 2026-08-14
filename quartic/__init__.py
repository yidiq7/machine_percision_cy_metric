"""Machine-precision Calabi-Yau metric on the Fermat quartic."""
import sys
from pathlib import Path

# The vendored directory MLGeometry/ has no __init__.py, so it shadows the real
# package as a namespace portion. Put MLGeometry/src first to remove the clash.
_SRC = Path(__file__).resolve().parents[1] / 'MLGeometry' / 'src'
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from . import blocks
from . import sections

__all__ = ['blocks', 'sections']
