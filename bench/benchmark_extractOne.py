##### LIST OF LIBRARIES IN THIS SCIPRT ####################
#
# edlib        https://github.com/Martinsos/edlib
# distance     https://github.com/doukremt/distance
# jellyfish    https://github.com/jamesturk/jellyfish
# editdistance https://github.com/aflc/editdistance
# Levenshtein  https://github.com/ztane/python-Levenshtein
# polyleven    https://github.com/fujimotos/polyleven
#
###########################################################

import importlib
import os.path
import sys
import random
from timeit import timeit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

PROCESSOR = {
    "fuzz.ratio":False,
    "string_metric.normalized_hamming": False,
    "fuzz.quick_ratio":False,
    "fuzz.partial_ratio":False,
    "fuzz.token_sort_ratio":True,
    "fuzz.token_set_ratio":True,
    "fuzz.partial_token_sort_ratio":True,
    "fuzz.partial_token_set_ratio":True,
    "fuzz.QRatio":True,
    "fuzz.WRatio":True
}

LIBRARIES = (
    "fuzz.ratio",
    "fuzz.partial_ratio",
    "fuzz.token_sort_ratio",
    "fuzz.token_set_ratio",
    "fuzz.partial_token_sort_ratio",
    "fuzz.partial_token_set_ratio",
    "fuzz.QRatio",
    "fuzz.WRatio"
)

WORD_TXT = 'data/words.txt'

def load_word():
    basedir = os.path.dirname(sys.argv[0])
    path = os.path.join(basedir, WORD_TXT)

    with open(path, encoding="utf-8") as fp:
        return [x.strip() for x in fp]*10

def load_func(target):
    modname, funcname = target.rsplit('.', maxsplit=1)

    module = importlib.import_module(modname)
    return getattr(module, funcname)

def get_platform():
    import platform
    uname = platform.uname()
    pyver = platform.python_version()
    return 'Python %s on %s (%s)' % (pyver, uname.system, uname.machine)

def benchmark():
    words = load_word()
    sample_rate = len(words) // 10
    sample = [word for word in words[::sample_rate]]
    total = len(words) * len(sample)

    print('System:', get_platform())
    print('Words :', len(words))
    print('Sample:', len(sample))
    print('Total : %s calls\n' % total)

    print('%-30s %s' % ('#', 'speed difference'))

    def wrap(f, scorer, processor, score_cutoff=0):
        def func():
            if not processor:
                return len([f(x, words, scorer=scorer, processor=None, score_cutoff=score_cutoff) for x in sample])
            return len([f(x, words, scorer=scorer, score_cutoff=score_cutoff) for x in sample])
        return func


    fuzz = []
    rfuzz = []
    for target in LIBRARIES:
 
        func = load_func("fuzzywuzzy.process.extractOne")
        scorer = load_func("fuzzywuzzy." + target)
        sec = timeit('func()', globals={'func': wrap(func, scorer, PROCESSOR[target], 0)}, number=1)
        calls = total / sec
        fuzz.append(calls)

        rfunc = load_func("rapidfuzz.process.extractOne")
        rscorer = load_func("rapidfuzz." + target)
        rsec = timeit('func()', globals={'func': wrap(rfunc, rscorer, PROCESSOR[target], 0)}, number=1)
        rcalls = total / rsec
        rfuzz.append(rcalls)

        print('%-30s  %i %%' % (target, 100 * sec/rsec))


    labels = LIBRARIES

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(17,10))
    rects1 = ax.bar(x - width/2, fuzz, width, label='FuzzyWuzzy', color="xkcd:coral")
    rects2 = ax.bar(x + width/2, rfuzz, width, label='RapidFuzz', color='#6495ED')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('evaluated word pairs [inputs/s]')
    ax.set_xlabel('Scorer')
    ax.set_title('The number of word pairs evaluated per second\n(the larger the better)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    #ax.ticklabel_format(style='plain', axis='y')
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:,}'.format(round(height)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    benchmark()
