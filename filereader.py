import dill


def fileload(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield dill.load(f)
            except EOFError:
                break