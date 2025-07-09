import pandas as pd

class DataLoader:
    
    def load_data_csv(self, file_path):
        data = pd.read_csv(file_path)
        for col in data.columns:
            if data[col].dtype == 'bool':
                data[col] = data[col].astype(str)
        return data
