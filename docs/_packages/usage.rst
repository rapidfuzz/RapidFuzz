Usage
=====================================

.. code-block:: console

   > from rapidfuzz import fuzz
   > from rapidfuzz import process


Simple Ratio
#####################

.. code-block:: console

   > fuzz.ratio("this is a test", "this is a test!")
   96.55171966552734


Partial Ratio
#####################

.. code-block:: console

   > fuzz.partial_ratio("this is a test", "this is a test!")
   100.0


Token Sort Ratio
#####################

.. code-block:: console

   > fuzz.ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
   90.90908813476562
   > fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
   100.0


Token Set Ratio
#####################

.. code-block:: console

   > fuzz.token_sort_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
   83.8709716796875
   > fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
   100.0


Process
#####################

.. code-block:: console

   > choices = ["Atlanta Falcons", "New York Jets", "New York Giants", "Dallas Cowboys"]
   > process.extract("new york jets", choices, limit=2)
   [('new york jets', 100), ('new york giants', 78.57142639160156)]
   > process.extractOne("cowboys", choices)
   ("dallas cowboys", 90)



.. toctree::
   :maxdepth: 2
   :caption: Contents:
