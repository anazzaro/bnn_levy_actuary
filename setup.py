from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bnn_levy_actuary",
    version="0.1.1",
    author="Antonio Nazzaro",
    author_email="info@antonionazzaro.it",
    description="Tempered stable Lévy-regularized Bayesian neural networks for fair reserve estimation under Solvency II.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anazzaro/bnn_levy_actuary",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "bnn_levy_actuary": ["DATI_EIOPA_30NOVEMBRE2024.csv"],
    },
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "torch>=1.9.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
