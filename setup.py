from setuptools import find_packages, setup


def get_version():
    version = {}
    with open("grimoire/_version.py") as f:
        exec(f.read(), version)
    return version["__version__"]


def get_long_description():
    with open("README.md", encoding="utf-8") as f:
        return f.read()


setup(
    name="grimoire-rl",
    version=get_version(),
    description="Simple, multi-GPU LLM fine-tuning library",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/Schneewolf-Labs/grimoire",
    project_urls={
        "Source": "https://github.com/Schneewolf-Labs/grimoire",
        "Changelog": "https://github.com/Schneewolf-Labs/grimoire/blob/main/CHANGELOG.md",
    },
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    packages=find_packages(include=["grimoire*"]),
    # NOTE: torch is intentionally excluded — it's version/CUDA-specific and
    # users must install it themselves. Using --force-reinstall with torch in
    # install_requires will nuke your entire CUDA stack.
    install_requires=[
        "accelerate>=0.24.0",
        "peft>=0.6.0",
        "datasets>=2.14.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "quantization": ["bitsandbytes>=0.41.0"],
        "logging": ["wandb>=0.15.0"],
        "liger": ["liger-kernel>=0.5.0"],
        "yaml": ["PyYAML>=6.0"],
        # ruff is capped: its default rule set grows between minor releases, so an open
        # bound lets a ruff release turn CI red on a commit nobody made.
        "dev": ["pytest>=7.0", "ruff>=0.1.0,<0.17", "PyYAML>=6.0"],
    },
)
