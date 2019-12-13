import pickle


def load_wv(wv_path):
    """Load word vector dictionary

    Arguments:
        wv_path {str} -- path of pickle file
    """
    with open(wv_path, 'rb') as file:
        wv_dict = pickle.load(file)

    return wv_dict
