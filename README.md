# blob-matcher

[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-blob--matcher-black?logo=github)](https://github.com/the-sauer/blob-matcher)

This repository contains the *BlobMatcher* feature matching module for *BlobBoards*.

## Usage

Make sure you have Python3.9 or later installed. Install the BlobMatcher with
```sh
pip install --update git+https://github.com/the-sauer/log-polar-descriptors@master
```
and import it with
```python
from blob_matcher import BlobMatcher
```

See the Documentation of `BlobMatcher` for details on the usage.

Alternatively, clone this repository and install it from the local source
```sh
git clone git@github.com:the-sauer/log-polar-descriptors.git
cd log-polar-descriptors
pip install -e .
```

### Docker

Build the image with
```sh
docker build -t blob-matcher .
```

Then run the container with
```sh
docker run --mount type=bind,src=./data,dst=/blobinator/data -i -t blob-matcher /bin/bash
```
make sure you have a data directory.

## Train the descriptor

### Obtaining the dataset

**TODO**

### Train

Run
```sh
blob-matcher-train
```

### Options

**TODO**

## Evaluate the Descriptor
