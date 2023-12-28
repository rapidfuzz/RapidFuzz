Installation
============

While there are several ways of install RapidFuzz, the recommended methods
are either by using ``pip`` (the Python package manager) or
``conda`` (an open-source, cross-platform, package manager)

using pip
---------

RapidFuzz can be installed with ``pip``:

.. code-block:: sh

   pip install rapidfuzz

There are pre-built binaries (wheels) of RapidFuzz for MacOS (10.9 and later), Linux x86_64 and Windows.

.. raw:: html

   <div class="admonition error">
     <p class="admonition-title">failure "ImportError: DLL load failed"</p>
     <p>
     If you run into this error on Windows the reason is most likely, that the
     <a class="reference external" href="https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads">Visual C++ 2019 redistributable</a> is not installed, which is required to
     find C++ Libraries (The C++ 2019 version includes the 2015, 2017 and 2019 version).</p>
   </div>


using conda
-----------

RapidFuzz can be installed with ``conda``:

.. code-block:: sh

   conda install -c conda-forge rapidfuzz


from git
--------

RapidFuzz can be directly used from GitHub by cloning the
repository. This requires a C++17 capable compiler.

.. code-block:: sh

   git clone --recursive https://github.com/rapidfuzz/rapidfuzz.git
   cd rapidfuzz
   pip install .
