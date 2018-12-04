def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step