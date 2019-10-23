from setuptools import setup


setup(
    name="edi",
    version="0.1.0",
    py_modules=["edi"],
    install_requires=[
        "Click",
        "colorama",
        "aiohttp",
        "tqdm",
        "aiodns",
        "iso8601",
        "pypeln",
        "pyjq",
    ],
    entry_points={"console_scripts": ["edi = edi.cli:cli"]},
)
