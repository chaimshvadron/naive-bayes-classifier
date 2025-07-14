import pandas as pd

class DataLoader:
    
    def load_data_csv(self, file_path):
        data = pd.read_csv(file_path, dtype=str)
        return data
    
    def load_and_split_csv(self, file_path, test_size=0.3, random_state=None):
        data = self.load_data_csv(file_path)
        # sample test set
        test = data.sample(frac=test_size, random_state=random_state)
        # remaining as train set
        train = data.drop(test.index)
        return train, test
