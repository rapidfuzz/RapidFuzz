rapidfuzz.utils
===============

default_process
---------------
.. autofunction:: rapidfuzz.utils.default_process


default_process_with_marks
--------------------------
.. autofunction:: rapidfuzz.utils.default_process_with_marks

Pass this processor explicitly to preserve marks used in scripts such as
Sinhala, Tamil, and Devanagari, as well as joining controls within words.

.. code-block:: python

   from rapidfuzz import process, utils

   process.extract(
       "කුරුණෑගල",
       ["කොළඹ", "කුරුණෑගල"],
       processor=utils.default_process_with_marks,
       score_cutoff=80,
   )
   # [('කුරුණෑගල', 100.0, 1)]

Both backends process strings in Python using the interpreter's Unicode
character database. If your input may mix canonically equivalent spellings,
normalize it separately.
