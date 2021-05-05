# distutils: language=c++
# cython: language_level=3
# cython: binding=True

{%- from "cpp_common.j2" import similarity_func %}

from rapidfuzz.utils import default_process
from cpp_common cimport proc_string, is_valid_string, convert_string, hash_array, hash_sequence
from array import array
from libc.stdlib cimport malloc, free

cdef inline proc_string conv_sequence(seq):
    if is_valid_string(seq):
        return convert_string(seq)
    elif isinstance(seq, array):
        return hash_array(seq)
    else:
        return hash_sequence(seq)

{{ similarity_func(
ratio_name="ratio",
processor_default="None",
description="""
calculates a simple ratio between two strings. This is a simple wrapper
for string_metric.normalized_levenshtein using the weights:
- weights = (1, 1, 2)
""",
see_also="",
notes = """
.. image:: img/ratio.svg
""",
examples="""
>>> fuzz.ratio(\"this is a test\", \"this is a test!\")
96.55171966552734
"""
)
}}

{{ similarity_func(
ratio_name="partial_ratio",
processor_default="None",
description="""
calculates the fuzz.ratio of the optimal string alignment
""",
notes = """
.. image:: img/partial_ratio.svg
""",
examples="""
>>> fuzz.partial_ratio(\"this is a test\", \"this is a test!\")
100.0
"""
)
}}

{{ similarity_func(
ratio_name="token_sort_ratio",
processor_default="True",
description="""
sorts the words in the strings and calculates the fuzz.ratio between them
""",
notes = """
.. image:: img/token_sort_ratio.svg
""",
examples="""
>>> fuzz.token_sort_ratio(\"fuzzy wuzzy was a bear\", \"wuzzy fuzzy was a bear\")
100.0
"""
)
}}

{{ similarity_func(
ratio_name="token_set_ratio",
processor_default="True",
description="""
Compares the words in the strings based on unique and common words between them
using fuzz.ratio
""",
notes = """
.. image:: img/token_set_ratio.svg
""",
examples="""
>>> fuzz.token_sort_ratio(\"fuzzy was a bear\", \"fuzzy fuzzy was a bear\")
83.8709716796875
>>> fuzz.token_set_ratio(\"fuzzy was a bear\", \"fuzzy fuzzy was a bear\")
100.0
"""
)
}}

{{ similarity_func(
ratio_name="token_ratio",
processor_default="True",
description="""
Helper method that returns the maximum of fuzz.token_set_ratio and fuzz.token_sort_ratio
(faster than manually executing the two functions)
""",
notes = """
.. image:: img/token_ratio.svg
"""
)
}}

{{ similarity_func(
ratio_name="partial_token_sort_ratio",
processor_default="True",
description="""
sorts the words in the strings and calculates the fuzz.partial_ratio between them
""",
notes = """
.. image:: img/partial_token_sort_ratio.svg
"""
)
}}

{{ similarity_func(
ratio_name="partial_token_set_ratio",
processor_default="True",
description="""
Compares the words in the strings based on unique and common words between them
using fuzz.partial_ratio
""",
notes = """
.. image:: img/partial_token_set_ratio.svg
"""
)
}}

{{ similarity_func(
ratio_name="partial_token_ratio",
processor_default="True",
description="""
Helper method that returns the maximum of fuzz.partial_token_set_ratio and
fuzz.partial_token_sort_ratio (faster than manually executing the two functions)
""",
notes = """
.. image:: img/partial_token_ratio.svg
"""
)
}}

{{ similarity_func(
ratio_name="WRatio",
processor_default="True",
description="""
Calculates a weighted ratio based on the other ratio algorithms
""",
notes = """
.. image:: img/WRatio.svg
"""
)
}}

{{ similarity_func(
ratio_name="QRatio",
processor_default="True",
description="""
Calculates a quick ratio between two strings using fuzz.ratio.
The only difference to fuzz.ratio is, that this preprocesses
the strings by default.
""",
examples="""
>>> fuzz.QRatio(\"this is a test\", \"THIS is a test!\")
100.0
"""
)
}}