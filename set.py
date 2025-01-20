from setuptools import setup, find_packages

setup(
    name="MLLibrary",  # 库的名称
    version="0.1.0",  # 初始版本号
    description="A custom machine learning library with preprocessing, modeling, and evaluation tools.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Bobo",  
    author_email="bo_chen@u.nus.edu",  
    url="https://github.com/your_username/MLLibrary",  # 替换为你的 GitHub 仓库地址
    packages=find_packages(),  # 自动查找所有子包
    install_requires=[
        "hyperopt>=0.2.7",
        "joblib>=1.4.2",
        "numpy>=2.2.1",
        "openpyxl>=3.1.5",
        "pandas>=2.2.3",
        "scikit-learn>=1.6.1",
        "scipy>=1.15.1",
        "torch>=2.5.1",
        "torchaudio>=2.5.1",
        "torchvision>=0.20.1",
        "tqdm>=4.67.1",
        "xgboost>=2.1.3",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    python_requires="3.12",
    extras_require={
        "gpu": ["nvidia-cudnn-cu12>=9.0"],
        "visualization": ["matplotlib>=3.6.0"],
    },
    include_package_data=True,
    license="MIT",
)
