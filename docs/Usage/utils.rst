rapidfuzz.utils
===============

default_process
---------------
.. autofunction:: rapidfuzz.utils.default_process

Unicode normalization
---------------------
Visually identical strings can use different Unicode representations. For
example, ``"caf\u00e9"`` contains a single character for the accented letter,
while ``"cafe\u0301"`` uses an ``e`` followed by a combining acute accent.
RapidFuzz does not normalize strings automatically. Use
:func:`unicodedata.normalize` with ``processor`` to normalize strings before
comparing them:

.. code-block:: pycon

    >>> from functools import partial
    >>> from unicodedata import normalize
    >>> from rapidfuzz import fuzz, process
    >>> from rapidfuzz.distance import Levenshtein
    >>> composed = "caf\u00e9"
    >>> decomposed = "cafe\u0301"
    >>> nfc = partial(normalize, "NFC")
    >>> round(fuzz.ratio(composed, decomposed), 2)
    66.67
    >>> fuzz.ratio(composed, decomposed, processor=nfc)
    100.0
    >>> Levenshtein.distance(composed, decomposed, processor=nfc)
    0
    >>> match = process.extractOne(composed, [decomposed], processor=nfc)
    >>> match[0] == decomposed, match[1], match[2]
    (True, 100.0, 0)

NFC makes canonically equivalent strings equal while preserving accents.
NFKC additionally folds compatibility distinctions, such as ``"①"`` into
``"1"``. Choose the normalization form according to the distinctions your
application needs to preserve.

When combining normalization with ``default_process``, normalize first:

.. code-block:: pycon

    >>> from rapidfuzz import utils
    >>> def normalize_and_process(value):
    ...     return utils.default_process(nfc(value))
    ...
    >>> fuzz.ratio(composed, decomposed, processor=normalize_and_process)
    100.0

``default_process`` can remove combining marks, so normalize first to
compose accents where possible before filtering. Search results contain
the original choices. Edit and alignment positions obtained with a
processor refer to the processed strings, whose lengths and positions can
differ from the originals.
