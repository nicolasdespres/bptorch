examplepackage.torch
====================

A hello-world for torch packages

You can install the package by opening a terminal, changing directory into the folder and typing:

luarocks make

With external folly library
VERBOSE=1 CMAKE_INCLUDE_PATH=$HOME/torch7/bptorch/folly/usr/include CMAKE_LIBRARY_PATH=$HOME/torch7/bptorch/folly/usr/lib luarocks make