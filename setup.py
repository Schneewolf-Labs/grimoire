from setuptools import setup, find_packages


def get_version():
    version = {}
    with open("grimoire/_version.py") as f:
        exec(f.read(), version)
    return version["__version__"]


setup(
    name="grimoire-rl",
    version=get_version(),
    description="Simple, multi-GPU LLM fine-tuning library",
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
        "dev": ["pytest>=7.0", "ruff>=0.1.0"],
    },
)
