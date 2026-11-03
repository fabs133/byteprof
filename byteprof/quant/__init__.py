"""Weight-only quantization of Bolmo's local stack: pure selection and byte accounting.

Nothing in this package imports torch or torchao at import time; `integrate.py` (torchao) is
imported only when `[quantize] target != "none"`.
"""
