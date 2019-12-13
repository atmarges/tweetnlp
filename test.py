from tweetnlp.utils.DataLoader import DataLoader

dataset_path = 'D:/HDD/Documents/Programming/Python/Notebooks/datasets/emoji_sentiment_small/dataset_emoji10n_test.tsv'

loader = DataLoader(dataset_path)

x_train, x_test, y_train, y_test = loader.load_data(
    verbose=True, test_size=0.25)
