from typing import List

from setuptools import setup


def clean_requirements(requirements: List[str]) -> List[str]:
    valid_requirements = []
    for requirement in requirements:
        requirement = requirement.rstrip()
        if not requirement:
            continue
        if requirement.startswith("#"):
            continue
        valid_requirements.append(requirement)
    return valid_requirements


version = None
author = None
author_email = None
with open("nmanga/constants.py") as f:
    for line in f:
        if line.find("__version__") >= 0 and version is None:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
        if line.find("__author__") >= 0 and author is None:
            author = line.split("=")[1].strip()
            author = author.strip('"')
            author = author.strip("'")
        if line.find("__author_email__") >= 0 and author_email is None:
            author_email = line.split("=")[1].strip()
            author_email = author_email.strip('"')
            author_email = author_email.strip("'")

if version is None:
    raise Exception("Version not found")
if author is None:
    raise Exception("Author not found")
if author_email is None:
    raise Exception("Author email not found")

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    requirements = clean_requirements(f.readlines())
with open("requirements-dev.txt") as f:
    dev_requirements = clean_requirements(f.readlines())


setup_args = dict(
    name="nmanga",
    version=version,
    description=("A collection of CLI function to process a pirated manga."),
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/noaione/nao-manga-rls",
    author=author,
    author_email=author_email,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Other Audience",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Utilities",
    ],
    keywords="manga colorlevel spreads auto-splitting processing cbz comic",
    packages=["nmanga", "nmanga.cli", "nmanga.templates"],
    install_requires=requirements,
    extras_require={"dev": dev_requirements},
    project_urls={
        "Bug Reports": "https://github.com/noaione/nao-manga-rls/issues",
        "Source": "https://github.com/noaione/nao-manga-rls",
    },
    entry_points={"console_scripts": ["nmanga=nmanga.cmd:main"]},
    python_requires=">=3.7",
)

setup(**setup_args)
