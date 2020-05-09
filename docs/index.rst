.. RapidFuzz documentation master file, created by
   sphinx-quickstart on Sat May  9 19:17:06 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RapidFuzz
=====================================

Rapid fuzzy string matching in Python and C++ using the Levenshtein Distance.

------------------


RapidFuzz is a fast string matching library for Python and C++, which is using the string similarity calculations from `FuzzyWuzzy <https://github.com/seatgeek/fuzzywuzzy>`_. However there are two aspects that set RapidFuzz apart from FuzzyWuzzy:

1) It is MIT licensed so it can be used whichever License you might want to choose for your project, while you're forced to adopt the GPL license when using FuzzyWuzzy

2) It is mostly written in C++ and on top of this comes with a lot of Algorithmic improvements to make string matching even faster, while still providing the same results. More details on these performance improvements in form of benchmarks can be found in `Benchmarks.md <https://github.com/maxbachmann/rapidfuzz/blob/master/Benchmarks.md>`_.



Package Reference
##########################

.. toctree::
   :maxdepth: 2

   _packages/installation.rst
   _packages/usage.rst
