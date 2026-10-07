"""Build the vampyhost C++ extension.

All package metadata lives in pyproject.toml; this file only defines
the extension module, which cannot be declared statically.
"""

import sys

import numpy
from setuptools import Extension, setup

sdkdir = "vamp-plugin-sdk/src/vamp-hostsdk/"
vpydir = "native/"

sdkfiles = [
    "Files",
    "PluginBufferingAdapter",
    "PluginChannelAdapter",
    "PluginHostAdapter",
    "PluginInputDomainAdapter",
    "PluginLoader",
    "PluginSummarisingAdapter",
    "PluginWrapper",
    "RealTime",
]
vpyfiles = ["PyPluginObject", "PyRealTime", "VectorConversion", "vampyhost"]

sources = [sdkdir + f + ".cpp" for f in sdkfiles]
sources += [vpydir + f + ".cpp" for f in vpyfiles]

# -fPIC/-O2 already come from Python's sysconfig; -stdlib=libc++ has been
# the macOS default for a decade. Only the Linux strict-aliasing
# workaround for this codebase is still needed. Keeping this empty on
# Windows also avoids passing gcc-only flags to MSVC.
extra_compile_args = []
if sys.platform == "linux":
    extra_compile_args.append("-fno-strict-aliasing")

vampyhost = Extension(
    "vampyhost",
    language="c++",
    sources=sources,
    define_macros=[("_USE_MATH_DEFINES", 1)],
    include_dirs=["vamp-plugin-sdk", numpy.get_include()],
    extra_compile_args=extra_compile_args,
)

setup(ext_modules=[vampyhost])
