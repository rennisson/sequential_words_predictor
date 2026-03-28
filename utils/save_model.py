from pathlib import Path
import pickle


def get_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_file(path, object):
    with open(path, "wb") as f:
        pickle.dump(object, f)


def save_model(marginal_probs, N, conditional_probs, words_frequencies):
    print(f"Saving model...")

    folder_path = Path(__file__).parent.parent / "pkl_files"

    file_path = folder_path / "cond_probs.pkl"
    save_file(file_path, conditional_probs)

    file_path = folder_path / "words_frequencies.pkl"
    save_file(file_path, words_frequencies)

    file_path = folder_path / "marginal_probs.pkl"
    save_file(file_path, marginal_probs)

    file_path = folder_path / "len_vocabulary.pkl"
    save_file(file_path, N)
