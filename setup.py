from setuptools import setup, find_packages

setup(
    name="grimoire",
    version="0.1.0",
    description="Simple, multi-GPU LLM fine-tuning library",
    python_requires=">=3.10",
    packages=find_packages(include=["grimoire*"]),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.24.0",
        "peft>=0.6.0",
        "datasets>=2.14.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "quantization": ["bitsandbytes>=0.41.0"],
        "logging": ["wandb>=0.15.0"],
        "dev": ["pytest>=7.0", "ruff>=0.1.0"],
    },
)
