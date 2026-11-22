"""
io utils.
"""

import os as _os


def make_meaningful(line, comment="#"):
    """
    Tries to turn a str input a meaningful line. Returns False if it is not
    possible

    Parameters
    -----------
    line: str
    comment: str
      str that defined comment section. "#" is default

    Returns
    --------

    """
    line = line.strip()
    if len(line) == 0 or line.startswith(comment):
        return False
    else:
        return line


def next_line(f, comment="#"):
    """
    Strips and returns next meaningful line from the opened, readable file.
    Returns None at the end of the file.

    Parameters
    -----------
    f: _io.TextIoWrapper
    comment: str
      A symbol that defines comment section. "#" is default.

    Returns
    --------
    next_meaningful_str: str
    """
    for line in iter(f.readline, ""):
        line = line.strip()  # noqa: PLW2901
        if line and not line.startswith(comment):
            return line
    return None


def form_lines(*args):
    """
    Formulate a string, taking each args as a line.

    Parameters
    -----------
    *args: str

    Returns
    --------
    line_separated_str: str
    """
    return "".join(a + "\n" for a in args)


def abs_fname(fname):
    """
    Checks if fname is absolute. If not, returns abspath. Tilde safe.

    Parameters
    ----------
    fname: str

    Returns
    --------
    abs_fname: str
      Maybe same to fname, maybe not.
    """
    return _os.path.abspath(_os.path.expanduser(fname))


def expand_tabs(fname, overwrite=True, tab_expand=2):
    """Replaces tabs in a text file with spaces

    Parameters
    ----------
    fname : string
      Filename
    overwrite : bool
      Inplace modification, otherwise new file with prefix copy is created
    tab_expand : int
      Determines how many spaces to be set for each tab

    Returns
    -------
    None
    """
    if overwrite:
        out_name = fname
    else:
        directory, file = _os.path.split(fname)
        out_name = _os.path.join(directory, "copy_" + file)

    with open(fname) as f:
        file_contents = f.read()
    with open(out_name, "w") as f:
        f.write(file_contents.replace("\t", " " * tab_expand))


def dict_to_spline(spline_dictionary):
    """Create a list of splines from a list of dictionaries

    Parameters
    ----------
    spline_dictionary : list
      List of dictionaries

    Returns
    -------
    spline_lits : list
      List of splines in called format (NAME_TO_TYPE)
    """
    from splinepy.settings import NAME_TO_TYPE as _NAME_TO_TYPE
    from splinepy.spline import Spline as _Spline

    spline_list = [_Spline(**spd) for spd in spline_dictionary]
    return [_NAME_TO_TYPE[spl.name](spline=spl) for spl in spline_list]
