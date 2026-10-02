# https://github.com/xinntao/BasicSR
# flake8: noqa
#
# Only the subpackages SegESR actually uses (data, utils, ...) are vendored, and the
# upstream `metrics` package is not included. The original eager imports
# (`from .metrics import *`, `.models`, `.train`, `.test`, ...) would therefore fail on
# `import basicsr`, so subpackages are imported on demand instead, e.g.
# `from basicsr.data.degradations import ...`.
