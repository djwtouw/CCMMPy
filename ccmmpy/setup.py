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
    name=package_name,
    version="0.1.1",
    description=("Implementation of the CCMM algorithm to minimize the convex "
                 "clustering loss function."),
    author="D.J.W. Touw",
    author_email="touw@ese.eur.nl",
    ext_modules=ext_modules,
    packages=[package_name],
    setup_requires=[
        "pybind11>=2.10",
    ],
    install_requires=[
        "matplotlib>=3.7",
        "numpy>=1.23",
        "pandas>=1.5",
        "pybind11>=2.10",
        "scikit-learn>=1.2",
        "seaborn>=0.12",
    ],
    zip_safe=False
)
