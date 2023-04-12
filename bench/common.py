from __future__ import annotations

import json
import timeit
from urllib.request import urlopen

from packaging.version import Version


def find_versions(package_name):
    url = f"https://pypi.org/pypi/{package_name}/json"
    data = json.load(urlopen(url))
    versions = list(data["releases"].keys())
    versions.sort(key=Version, reverse=True)
    return versions


def benchmark(name, func, setup, lengths, count):
    print(f"starting {name}")
    start = timeit.default_timer()
    results = []
    from tqdm import tqdm

    for length in tqdm(lengths):
        test = timeit.Timer(func, setup=setup.format(length, count))
        results.append(min(test.timeit(number=1) for _ in range(7)) / count)
    stop = timeit.default_timer()
    print(f"finished {name}, Runtime: ", stop - start)
    return results
