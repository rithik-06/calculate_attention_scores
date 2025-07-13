from setuptools import setup, find_packages

setup(
    name="attention_score_test",
    version="0.1.0",
    description="A simple project to calculate attention scores (scaled dot-product attention)",
    author="Your Name",
    packages=find_packages(),
    install_requires=["numpy>=1.18.0"],
    python_requires=">=3.6",
    url="https://github.com/yourusername/attention_score_test",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 