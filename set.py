from setuptools import setup, find_packages

setup(
    name="MLLibrary",  # 库的名称，用户通过 `pip install MLLibrary` 安装
    version="0.1.0",  # 版本号
    author="Bobo",
    author_email="2308087369@qq.com",
    description="A custom machine learning library for data processing, model selection, and evaluation.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/2308087369/MLLibrary",  # 项目链接
    packages=find_packages(exclude=["tests*", "examples*"]),  # 自动发现库
    install_requires=[
        "numpy>=2.2.1",
        "pandas>=2.2.3",
        "scikit-learn>=1.6.1",
        "scipy>=1.15.1",
        "torch>=2.5.1",
        "xgboost>=2.1.3",
        "tqdm>=4.67.1",
        "hyperopt>=0.2.7",
        "matplotlib>=3.5.1",
        "seaborn>=0.12.2",
        "openpyxl>=3.1.5",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires="3.12",
)
