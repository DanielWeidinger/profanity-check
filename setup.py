import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()


def package_files(directory):
    paths = []
    for (path, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


sentence_transformers_files = package_files(
    'profanity_protector/data/sentence-transformers_all-MiniLM-L6-v2')

setuptools.setup(
    name="profanity-protector",
    version="1.0.0",
    author="Daniel Weidinger",
    author_email="daniel.weidinger@codl.at",
    description="An even more robust fork of 'profanity-check'.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DanielWeidinger/profanity-protector",
    packages=setuptools.find_packages(),
    install_requires=['joblib>=0.14.1', 'scikit-learn>=0.20.2',
                      'sentence_transformers==2.2.2'],
    package_data={'profanity_protector': [
        'data/model.joblib'] + sentence_transformers_files},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
