from timeit import timeit
import math
import csv

iterations = 100000


reader = csv.DictReader(open('titledata.csv'), delimiter='|')
titles = [i['custom_title'] for i in reader]
title_blob = '\n'.join(titles)


cirque_strings = [
    "cirque du soleil - zarkana - las vegas",
    "cirque du soleil ",
    "cirque du soleil",
    "cirque du soleil las vegas",
    "zarkana las vegas",
    "las vegas cirque du soleil at the bellagio",
    "zarakana - cirque du soleil - bellagio"
]

choices = [
    "",
    "new york yankees vs boston red sox",
    "",
    "zarakana - cirque du soleil - bellagio",
    None,
    "cirque du soleil las vegas",
    None
]

mixed_strings = [
    "Lorem Ipsum is simply dummy text of the printing and typesetting industry.",
    "C\\'est la vie",
    u"Ça va?",
    u"Cães danados",
    u"\xacCamarões assados",
    u"a\xac\u1234\u20ac\U00008000"
]

common_setup = "from {} import fuzz, process; "


def print_result_from_timeit(stmt='pass', stmt_cpp='pass', setup='pass', number=1000000):
    """
    Clean function to know how much time took the execution of one statement
    """
    units = ["s", "ms", "us", "ns"]

    setup_fuzzywuzzy = setup.format("fuzzywuzzy")
    duration_fuzzywuzzy = timeit(stmt, setup_fuzzywuzzy, number=int(number))
    avg_duration = duration_fuzzywuzzy / float(number)
    thousands = int(math.floor(math.log(avg_duration, 1000)))

    print("Total time FuzzyWuzzy: %fs. Average run: %.3f%s." % (
        duration_fuzzywuzzy, avg_duration * (1000 ** -thousands), units[-thousands]))
    

    setup_rapidfuzz = setup.format("rapidfuzz")
    duration_rapidfuzz = timeit(stmt_cpp, setup_rapidfuzz, number=int(number))
    avg_duration = duration_rapidfuzz / float(number)
    thousands = int(math.floor(math.log(avg_duration, 1000)))

    print("Total time RapidFuzz: %fs. Average run: %.3f%s." % (
        duration_rapidfuzz, avg_duration * (1000 ** -thousands), units[-thousands]))
    
    relative_duration = duration_fuzzywuzzy / duration_rapidfuzz
    print("RapidFuzz is %.3f times faster than FuzzyWuzzy" % relative_duration)

    print()


# benchmarking the core matching methods...

for s in cirque_strings:
    print('Test fuzz.ratio for string: "%s"' % s)
    print('-------------------------------')
    print_result_from_timeit('fuzz.ratio(u\'cirque du soleil\', u\'%s\')' % s,
                             'fuzz.ratio(u\'cirque du soleil\', u\'%s\', preprocess=False)' % s,
                             common_setup, number=iterations / 100)

for s in cirque_strings:
    print('Test fuzz.partial_ratio for string: "%s"' % s)
    print('-------------------------------')
    print_result_from_timeit('fuzz.partial_ratio(u\'cirque du soleil\', u\'%s\')' % s,
                            'fuzz.partial_ratio(u\'cirque du soleil\', u\'%s\', preprocess=False)' % s,
                             common_setup, number=iterations / 100)

for s in cirque_strings:
    print('Test fuzz.WRatio for string: "%s"' % s)
    print('-------------------------------')
    print_result_from_timeit('fuzz.WRatio(u\'cirque du soleil\', u\'%s\')' % s,
                            'fuzz.WRatio(u\'cirque du soleil\', u\'%s\')' % s,
                             common_setup, number=iterations / 100)

print('Test process.extract(scorer =  fuzz.WRatio) for string: "%s"' % s)
print('-------------------------------')
stmt = 'process.extract("%s", choices, scorer =  fuzz.WRatio)' % s
print_result_from_timeit(stmt, stmt,
                             common_setup + " import string,random; random.seed(18);"
                             " choices = [\'\'.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(30)) for s in range(5000)]",
                              number=10)

print('Test process.extract(scorer =  fuzz.WRatio, score_cutoff=70) for string: "%s"' % s)
print('-------------------------------')
stmt = 'process.extract("%s", choices, scorer =  fuzz.WRatio)' %s
stmt2 = 'process.extract("%s", choices, scorer =  fuzz.WRatio, score_cutoff=70)' % s
print_result_from_timeit(stmt, stmt2,
                             common_setup + " import string,random; random.seed(18);"
                             " choices = [\'\'.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(30)) for s in range(5000)]",
                              number=10)

print('Test process.extractOne(scorer =  fuzz.WRatio) for string: "%s"' % s)
print('-------------------------------')
stmt = 'process.extractOne("%s", choices)' % s
print_result_from_timeit(stmt, stmt,
                             common_setup + " import string,random; random.seed(18);"
                             " choices = [\'\'.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(30)) for s in range(5000)]",
                              number=10)

print('Test process.extractOne(scorer =  fuzz.WRatio, score_cutoff=70) for string: "%s"' % s)
print('-------------------------------')
stmt = 'process.extractOne("%s", choices, score_cutoff=70)' % s
print_result_from_timeit(stmt, stmt,
                             common_setup + " import string,random; random.seed(18);"
                             " choices = [\'\'.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(30)) for s in range(5000)]",
                              number=10)

s = 'New York Yankees'
test = 'import functools\n'
test += 'title_blob = """%s"""\n' % title_blob
test += 'title_blob = title_blob.strip()\n'
test += 'titles = title_blob.split("\\n")\n'

print('Real world ratio(): "%s"' % s)
print('-------------------------------')
test += 'prepared_ratio = functools.partial(fuzz.ratio, "%s")\n' % s
test += 'titles.sort(key=prepared_ratio)\n'
print_result_from_timeit(test, test,
                         common_setup,
                         number=100)