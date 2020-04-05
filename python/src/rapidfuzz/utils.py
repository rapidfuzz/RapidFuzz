import re

def default_process(s):
	alnum_re = re.compile(r"(?ui)\W")
	return alnum_re.sub(" ", s).strip().lower()