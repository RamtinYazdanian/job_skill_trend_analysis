import os
import errno

def remove_nones(l):
    return [x for x in l if x is not None]

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST and exception.errno != errno.EPERM:
            raise

def invert_dict(d):
    return {d[k]: k for k in d}

def capitalise_initials(s):
    if ' ' not in s:
        return s[0].upper() + s[1:].lower()
    l = s.lower().split(' ')
    return ' '.join([capitalise_initials(l[0])] + l[1:])

