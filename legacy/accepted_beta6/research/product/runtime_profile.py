"""Versioned numerical dispatch policy, separate from the frozen formulas."""
import os

PROFILE_ID = "numpy235-historical-avx-ceiling-v1"
DISABLED_FEATURES = "AVX2,FMA3,AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL"


def configure():
    # Process-local, before NumPy import. Do not change machine/user settings.
    os.environ["NPY_DISABLE_CPU_FEATURES"] = DISABLED_FEATURES
    os.environ["MICROSTRUCTAI_RUNTIME_PROFILE"] = PROFILE_ID


def describe():
    import numpy as np
    from numpy._core import _multiarray_umath as implementation
    flags = implementation.__cpu_features__
    disabled = DISABLED_FEATURES.split(",")
    verified = (np.__version__ == "2.3.5"
                and os.environ.get("MICROSTRUCTAI_RUNTIME_PROFILE") == PROFILE_ID
                and os.environ.get("NPY_DISABLE_CPU_FEATURES") == DISABLED_FEATURES
                and not any(flags.get(name, False) for name in disabled))
    return {"declared_profile": os.environ.get("MICROSTRUCTAI_RUNTIME_PROFILE", "unconfigured"),
            "expected_profile": PROFILE_ID, "profile_active": verified,
            "numpy_disable_cpu_features": os.environ.get("NPY_DISABLE_CPU_FEATURES", ""),
            "numpy_dispatch_enabled": [name for name in implementation.__cpu_dispatch__ if flags.get(name, False)],
            "scope": "Runtime configuration, not a formula or tolerance change; cross-hardware acceptance still required"}
