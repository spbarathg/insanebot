from setuptools import setup, find_packages

setup(
    name="trading-bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "aiohttp>=3.8.0",
        "asyncio>=3.4.3",
        "python-dotenv>=0.19.0",
        "pydantic>=1.9.0",
        "loguru>=0.5.3",
        "solana>=0.30.0",
        "anchorpy>=0.18.0",
        "prometheus-client>=0.12.0",
        "pytest>=6.2.5",
        "pytest-asyncio>=0.16.0",
        "pytest-cov>=2.12.0",
    ],
    python_requires=">=3.7",
) 