from torch.utils.data import Dataset


class AliExpressDataset(Dataset):
    def __init__(self, df, categorical_columns, numerical_columns):
        self.df = df
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

        self.categorical_df = self.df[categorical_columns]
        self.numerical_df = self.df[numerical_columns]

        self.click_df = self.df["click"]
        self.cv_df = self.df["conversion"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        categorical_features = self.categorical_df[idx].to_numpy()
        numerical_features = self.numerical_df[idx].to_numpy()

        click_label = self.click_df[idx]
        cv_label = self.cv_df[idx]

        return categorical_features, numerical_features, click_label, cv_label
