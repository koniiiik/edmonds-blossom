===========================
Edmonds's blossom algorithm
===========================

This is my implementation of the blossom algorithm for min-weight maximum
matchings on general graphs, made as an assignment for an approximation
algorithms course. It is written in Python and has been tested on CPython
2.7, 3.2 and 3.3 and PyPy 2.0.

The code is not really optimized, it takes a little more than a minute to
calculate the result for a graph on 1000 vertices with 10495 edges on
PyPy; the time it takes for a complete graph on 1002 vertices on my laptop
is about 45 minutes on PyPy and nearly six hours on CPython 3.3.
