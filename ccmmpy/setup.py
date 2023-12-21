import os
import pybind11
from distutils.core import setup, Extension


cpp_args = ["-std=c++11", "-DNDEBUG", "-O3", "-Wall"]
package_name = "ccmmpy"

ext_modules = [
    Extension(
        f"{package_name}._{package_name}",
        ["cpp/src/" + file for file in os.listdir("cpp/src")],
        include_dirs=["pybind11/include",
                      "cpp/include",
                      pybind11.get_include()],
        language="c++",
        extra_compile_args=cpp_args,
    ),
]

setup(
    ext_modules=ext_modules,
    packages=[package_name],
    zip_safe=False
)
