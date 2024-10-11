from setuptools import find_packages, setup
import pathlib
with open("README.md", "r") as f:
    long_description = f.read()

HERE = pathlib.Path(__file__).parent

setup(
    name="predacons",
    version="0.0.126",
    description="A python library based on transformers for transfer learning",
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shouryashashank/predacons",
    author="shouryashashank",
    author_email="shouryashashank@gmail.com",
    license="AGPLv3+",
    classifiers=[
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",   
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],  
    install_requires=["bson >= 0.5.10",
                    "pandas >= 1.5.3",
                    "numpy >= 1.24.1",
                    "regex >= 2021.4.4",
                    "PyPDF2 >= 3.0.1",
                    "docx >= 0.2.4",
                    "python-docx >= 1.0.1",
                    "transformers >= 4.29.1",
                    "einops >= 0.7.0",
                    "openai >= 1.12.0",
                    # "torch >= 2.1.2", removed torch from install_requires so that it does not cuda version of torch
                    "typing-extensions >=4.9.0",
                    # "bitsandbytes --index-url https://pypi.org/simple/",removed bitsandbytes from install_requires so that it does not cuda version of torch
                    "peft >= 0.8.2",
                    "trl >= 0.8.1"],

    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.10",
)
