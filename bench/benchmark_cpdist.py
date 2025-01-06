from __future__ import annotations

import importlib
import random
import string
from timeit import timeit

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

random.seed(18)

plt.rc("font", size=13)  # controls default text sizes
plt.rc("axes", titlesize=18)  # fontsize of the axes title
plt.rc("axes", labelsize=15)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=15)  # fontsize of the tick labels
plt.rc("ytick", labelsize=15)  # fontsize of the tick labels
plt.rc("legend", fontsize=15)  # legend fontsize

LIBRARIES = (
    "ratio",
    "partial_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "partial_token_sort_ratio",
    "partial_token_set_ratio",
    "QRatio",
    "WRatio",
)


def load_func(target):
    modname, funcname = target.rsplit(".", maxsplit=1)

    module = importlib.import_module(modname)
    return getattr(module, funcname)


def get_platform():
    import platform

    uname = platform.uname()
    pyver = platform.python_version()
    return f"Python {pyver} on {uname.system} ({uname.machine})"


def benchmark():
    words = ["".join(random.choice(string.ascii_letters + string.digits) for _ in range(8)) for _ in range(1000000)]
    sample_rate = len(words) // 2
    words1 = words[:sample_rate]
    words2 = words[sample_rate::]
    total = len(words1)

    print("System:", get_platform())
    print("Words :", len(words1))
    print("Sample:", len(words2))
    print("Total : %s calls\n" % total)

    def wrap_cpdist(scorer):
        from rapidfuzz.process import cpdist

        def func():
            cpdist(words1, words2, scorer=scorer, workers=4)

        return func

    def wrap_iterate(scorer):
        def func():
            for query, choice in zip(words1, words2):
                scorer(query, choice)

        return func

    fuzz = []
    rfuzz = []

    header_list = ["Function", "RapidFuzz_cpdist_4_workers", "RapidFuzz_fuzz", "SpeedImprovement"]
    row_format = "{:>25}" * len(header_list)
    print(row_format.format(*header_list))
    for target in LIBRARIES:
        rscorer = load_func("rapidfuzz.fuzz." + target)
        sec = timeit(
            "func()",
            globals={"func": wrap_iterate(rscorer)},
            number=1,
        )
        calls = total / sec
        fuzz.append(calls)

        rsec = timeit("func()", globals={"func": wrap_cpdist(rscorer)}, number=1)
        rcalls = total / rsec
        rfuzz.append(rcalls)

        print(row_format.format(target, f"{rcalls//1000}k", f"{calls//1000}k", f"{int(100 * sec/rsec)}%"))

    labels = LIBRARIES

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(17, 10))
    rects1 = ax.bar(x - width / 2, fuzz, width, label="RapidFuzz_fuzz", color="xkcd:coral")
    rects2 = ax.bar(x + width / 2, rfuzz, width, label="RapidFuzz_cpdist_4_workers", color="#6495ED")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("evaluated word pairs [inputs/s]")
    ax.set_xlabel("Scorer")
    ax.set_title("The number of word pairs evaluated per second\n(the larger the better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: format(int(x), ",")))
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{int(height):,}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    benchmark()
