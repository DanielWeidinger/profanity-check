import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="profanity-protector",
    version="1.0.0",
    author="Daniel Weidinger",
    author_email="daniel.weidinger@codl.at",
    description="A fast, robust library to check for offensive language in strings.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DanielWeidinger/profanity-protector",
    packages=setuptools.find_packages(),
    install_requires=['joblib>=0.14.1', 'scikit-learn>=0.20.2',
                      'sentence_transformers==2.2.2'],
    package_data={'profanity_protector': [
        'data/model.joblib', 'data/sentence-transformers_all-MiniLM-L6-v2']},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
