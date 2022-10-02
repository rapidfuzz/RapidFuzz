Jaro
----

Functions
^^^^^^^^^

distance
~~~~~~~~
.. autofunction:: rapidfuzz.distance.Jaro.distance

normalized_distance
~~~~~~~~~~~~~~~~~~~
.. autofunction:: rapidfuzz.distance.Jaro.normalized_distance

similarity
~~~~~~~~~~
.. autofunction:: rapidfuzz.distance.Jaro.similarity

normalized_similarity
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: rapidfuzz.distance.Jaro.normalized_similarity

Performance
^^^^^^^^^^^
The following image shows a benchmark of the Jaro similarity in RapidFuzz
and jellyfish. Jellyfish uses an implementation with a time complexity of ``O(NM)``,
while RapidFuzz has a time complexity of ``O([N/64]M)``.

.. image:: img/jaro.svg
    :align: center
