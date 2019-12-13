# tweetnlp
A python package containing helper tools for loading tweet data for NLP and machine learning

When doing NLP and machine learning on social media data such as tweets, 
a big chunk of the time spent goes to loading and preprocessing datasets. 
The goal of this project is to create a tool that allows data scientists 
to focus more on optimizing machine learning models.

## Usage
```python
from tweetnlp.utils.DataLoader import DataLoader

dataset_path = 'path/to/dataset/dataset.tsv'

loader = DataLoader(dataset_path)

x_train, x_test, y_train, y_test = loader.load_data(test_size=0.25)

```
