import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, embedding_sizes):
        super().__init__()
        self.embedding_layers = nn.ModuleList(
            [
                nn.Embedding(num_unique, emb_dim)
                for num_unique, emb_dim in embedding_sizes
            ]
        )

    def forward(self, categorical_feats, numerical_feats):
        cat_embs = [
            embedding_layer(categorical_feats[:, i])
            for i, embedding_layer in enumerate(self.embedding_layers)
        ]
        cat_embs = torch.concat(cat_embs, dim=-1)
        concat_feats = torch.concat((cat_embs, numerical_feats), dim=-1)
        return concat_feats


class CTRTower(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dims[0], out_features=1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.mlp(x)
        p_ctr = self.sigmoid(logits)
        return p_ctr


class CVRTower(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dims[0], out_features=1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.mlp(x)
        p_ctr = self.sigmoid(logits)
        return p_ctr


class ESMM(nn.Module):
    def __init__(
        self, embedding_sizes, num_numerical, hidden_dims, task_tower_hidden_dims
    ):
        super().__init__()
        self.embedding_sizes = embedding_sizes
        cat_dim = sum(t[1] for t in self.embedding_sizes)
        self.input_dim = cat_dim + num_numerical

        self.feature_extractor = FeatureExtractor(self.embedding_sizes)

        self.shared_bottom = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )

        self.ctr_tower = CTRTower(hidden_dims[1], task_tower_hidden_dims)
        self.cvr_tower = CVRTower(hidden_dims[1], task_tower_hidden_dims)

    def forward(self, categorical_feats, numerical_feats):
        concat_feat = self.feature_extractor(categorical_feats, numerical_feats)

        shared_feat = self.shared_bottom(concat_feat)

        p_ctr = self.ctr_tower(shared_feat)
        p_cvr = self.cvr_tower(shared_feat)

        return p_ctr, p_cvr
