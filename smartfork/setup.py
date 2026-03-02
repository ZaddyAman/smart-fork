"""Setup configuration for SmartFork."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="smartfork",
    version="0.1.0",
    description="AI Session Intelligence for Kilo Code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SmartFork Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "chromadb>=0.4.18",
        "sentence-transformers>=2.2.2",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "typer>=0.9.0",
        "rich>=13.7.0",
        "watchdog>=3.0.0",
        "loguru>=0.7.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "smartfork=smartfork.cli:app",
        ]
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
