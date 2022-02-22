import time
import codecs
import numpy as np


class Timer:
    def __init__(self):
        self.start_time = None
        self.job = None

    def start(self, job, verbal=False):
        self.job = job
        self.start_time = time.time()
        if verbal:
            print("[I] {job} started.".format(job=self.job))

    def stop(self):
        if self.job is None:
            return None
        elapsed_time = time.time() - self.start_time
        print("[I] {job} finished in {elapsed_time:0.3f} s."
              .format(job=self.job, elapsed_time=elapsed_time))
        self.job = None


class Log:
    verbose = True

    @staticmethod
    def log(text):
        if Log.verbose:
            print(text)


class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
        ERROR: Unable to locate file {}.

        FIX: Have you tried running python build_data first?
        This will build vocab file from your train, test and dev sets and
        trim your word vectors.""".format(filename)

        super(MyIOError, self).__init__(message)


def load_vocab(filename):
    try:
        d = dict()
        with codecs.open(filename, encoding='utf-8') as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx + 1  # preserve idx 0 for pad_tok

    except IOError:
        raise MyIOError(filename)
    return d


def get_trimmed_w2v_vectors(filename):
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)
