from setuptools import setup, find_packages

setup(
    name="malbecs",
    version="0.0.1",
    description="Utilities for datathon cajamar",
    author='Malbecs',
    packages=find_packages(include=["smarts"]),
    install_requires=[
        "pandas"
    ]
)
