"""Utilities for the AloePri stage-A vocabulary permutation prototype."""

import os
import platform

# Python 3.13+ platform.machine() queries WMI before Torch loads its DLLs.
# Use platform's WinAPI/environment fallbacks when WMI can stall under Windows
# memory pressure. This changes platform detection only in this process.
if os.name == "nt" and hasattr(platform, "_wmi"):
    platform._wmi = None
