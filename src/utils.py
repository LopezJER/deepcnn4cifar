import pickle


def unpickle(file_path):
    """Unpickle a file and return its content."""
    with open(file_path, "rb") as fo:
        return pickle.load(fo, encoding="bytes")
