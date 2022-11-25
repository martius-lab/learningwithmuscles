from setuptools import find_packages, setup

setup(
    name="warmup",
    version="1.0.0",
    install_requires=["gym"],
    author="Pierre Schumacher, MPI-IS Tuebingen, Autonomous Learning",
    author_email="pierre.schumacher@tuebingen.mpg.de",
    license="MIT",
    packages=["warmup"],
    include_package_data=True,
    package_data={
        "warmup": ["param_files/*.json", "xml_files/Geometry/*.stl"]
    },
)
