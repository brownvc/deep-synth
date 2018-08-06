import gzip
import os
import os.path
import sys
import pickle
from contextlib import contextmanager

def ensuredir(dirname):
    """
    Ensure a directory exists
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def zeropad(num, n):
    """
    Turn a number into a string that is zero-padded up to length n
    """
    sn = str(num)
    while len(sn) < n:
        sn = '0' + sn
    return sn

def pickle_dump_compressed(object, filename, protocol=pickle.HIGHEST_PROTOCOL):
    """
    Pickles + compresses an object to file
    """
    file = gzip.GzipFile(filename, 'wb')
    file.write(pickle.dumps(object, protocol))
    file.close()

def pickle_load_compressed(filename):
    """
    Loads a compressed pickle file and returns reconstituted object
    """
    file = gzip.GzipFile(filename, 'rb')
    buffer = b""
    while True:
        data = file.read()
        if data == b"":
            break
        buffer += data
    object = pickle.loads(buffer)
    file.close()
    return object

def get_data_root_dir():
    """
    Gets the root dir of the dataset
    Check env variable first,
    if not set, use the {code_location}/data
    """
    env_path = os.environ.get("SCENESYNTH_DATA_PATH")
    if env_path:
    #if False: #Debug purposes
        return env_path
    else:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        return f"{root_dir}/data"

@contextmanager
def stdout_redirected(to=os.devnull):
    """
    From https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
    Suppress C warnings
    """
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different
