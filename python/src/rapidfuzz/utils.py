
def default_process(s):
	return s.replace('\x00', ' ').strip().lower()