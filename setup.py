from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="colorectal_cancer_prediction",
    version="0.1",
    author="Nilesh",
    packages=find_packages(),
    install_requires=requirements,
)
