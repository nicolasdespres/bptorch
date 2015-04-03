# bptorch

Library containing CPU/Cuda modules for torch:

- Binary Hierarchical SoftMax

## TODO

- Get rid of the folly library dependency

## Requirements

- The facebook [folly](https://github.com/facebook/folly) library

## Installation

You can install the package by opening a terminal, changing directory into the folder and typing:

- Standard location for the folly library

     luarocks make

- Specifying the folly library path:

     CMAKE_INCLUDE_PATH=$FOLLYDIR/include CMAKE_LIBRARY_PATH=$FOLLYDIR/lib luarocks make
