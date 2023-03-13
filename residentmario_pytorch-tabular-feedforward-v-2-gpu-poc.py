import pandas as pd

from pathlib import Path



path = Path('rossmann')

train_df = pd.read_pickle('../input/rossman-fastai-sample/train_clean').drop(['index', 'Date'], axis='columns')

test_df = pd.read_pickle('../input/rossman-fastai-sample/test_clean')
train_df.head()
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.pipeline import FeatureUnion, Pipeline

import numpy as np





cat_vars = [

    'Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'CompetitionMonthsOpen',

    'Promo2Weeks', 'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear',

    'State', 'Week', 'Events', 'Promo_fw', 'Promo_bw', 'StateHoliday_fw', 'StateHoliday_bw',

    'SchoolHoliday_fw', 'SchoolHoliday_bw'

]

cont_vars = [

    'CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',

    'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 

    'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',

    'AfterStateHoliday', 'BeforeStateHoliday', 'Promo', 'SchoolHoliday'

]

target_var= 'Sales'





class ColumnFilter:

    def fit(self, X, y):

        return self

    

    def transform(self, X):

        return X.loc[:, cat_vars + cont_vars]

        



class GroupLabelEncoder:

    def __init__(self):

        self.labeller = LabelEncoder()

    

    def fit(self, X, y):

        self.encoders = {col: None for col in X.columns if col in cat_vars}

        for col in self.encoders:

            self.encoders[col] = LabelEncoder().fit(

                X[col].fillna(value='N/A').values

            )

        return self

    

    def transform(self, X):

        X_out = []

        categorical_part = np.hstack([

            self.encoders[col].transform(X[col].fillna(value='N/A').values)[:, np.newaxis]

            for col in cat_vars

        ])

        return pd.DataFrame(categorical_part, columns=cat_vars)





class GroupNullImputer:

    def fit(self, X, y):

        return self

        

    def transform(self, X):

        return X.loc[:, cont_vars].fillna(0)





class Preprocessor:

    def __init__(self):

        self.cf = ColumnFilter()

        self.gne = GroupNullImputer()

        

    def fit(self, X, y=None):

        self.gle = GroupLabelEncoder().fit(X, y=None)

        return self

    

    def transform(self, X):

        X_out = self.cf.transform(X)

        X_out = np.hstack((self.gle.transform(X_out).values, self.gne.transform(X_out).values))

        X_out = pd.DataFrame(X_out, columns=cat_vars + cont_vars)

        return X_out





X_train_sample = Preprocessor().fit(train_df).transform(train_df.loc[:999])

y_train_sample = train_df[target_var].loc[:999]
import torch

from torch import nn

import torch.utils.data

# ^ https://discuss.pytorch.org/t/attributeerror-module-torch-utils-has-no-attribute-data/1666





class FeedforwardTabularModel(nn.Module):

    def __init__(self):

        super().__init__()

        

        self.batch_size = 512

        self.base_lr, self.max_lr = 0.001, 0.003

        self.n_epochs = 5

        self.cat_vars_embedding_vector_lengths = [

            (1115, 80), (7, 4), (3, 3), (12, 6), (31, 10), (2, 2), (25, 10), (26, 10), (4, 3),

            (3, 3), (4, 3), (23, 9), (8, 4), (12, 6), (52, 15), (22, 9), (6, 4), (6, 4), (3, 3),

            (3, 3), (8, 4), (8, 4)

        ]

        self.loss_fn = torch.nn.MSELoss()

        self.score_fn = torch.nn.MSELoss()

        

        # Layer 1: embeddings.

        self.embeddings = []

        for i, (in_size, out_size) in enumerate(self.cat_vars_embedding_vector_lengths):

            emb = nn.Embedding(in_size, out_size)

            self.embeddings.append(emb)

            setattr(self, f'emb_{i}', emb)



        # Layer 1: dropout.

        self.embedding_dropout = nn.Dropout(0.04)

        

        # Layer 1: batch normalization (of the continuous variables).

        self.cont_batch_norm = nn.BatchNorm1d(16, eps=1e-05, momentum=0.1)

        

        # Layers 2 through 9: sequential feedforward model.

        self.seq_model = nn.Sequential(*[

            nn.Linear(in_features=215, out_features=1000, bias=True),

            nn.ReLU(),

            nn.BatchNorm1d(1000, eps=1e-05, momentum=0.1),

            nn.Dropout(p=0.001),

            nn.Linear(in_features=1000, out_features=500, bias=True),

            nn.ReLU(),

            nn.BatchNorm1d(500, eps=1e-05, momentum=0.1),

            nn.Dropout(p=0.01),

            nn.Linear(in_features=500, out_features=1, bias=True)

        ])

    

    

    def forward(self, x):

        # Layer 1: embeddings.

        inp_offset = 0

        embedding_subvectors = []

        for emb in self.embeddings:

            index = torch.tensor(inp_offset, dtype=torch.int64).cuda()

            inp = torch.index_select(x, dim=1, index=index).long().cuda()

            out = emb(inp)

            out = out.view(out.shape[2], out.shape[0], 1).squeeze()

            embedding_subvectors.append(out)

            inp_offset += 1

        out_cat = torch.cat(embedding_subvectors)

        out_cat = out_cat.view(out_cat.shape[::-1])

        

        # Layer 1: dropout.

        out_cat = self.embedding_dropout(out_cat)

        

        # Layer 1: batch normalization (of the continuous variables).

        out_cont = self.cont_batch_norm(x[:, inp_offset:])

        

        out = torch.cat((out_cat, out_cont), dim=1)

        

        # Layers 2 through 9: sequential feedforward model.

        out = self.seq_model(out)

            

        return out

        

        

    def fit(self, X, y):

        self.train()

        

        # TODO: set a random seed to invoke determinism.

        # cf. https://github.com/pytorch/pytorch/issues/11278



        X = torch.tensor(X, dtype=torch.float32)

        y = torch.tensor(y, dtype=torch.float32)

        

        # The build of PyTorch on Kaggle has a blog that prevents us from using

        # CyclicLR with ADAM. Cf. GH#19003.

        # optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)

        # scheduler = torch.optim.lr_scheduler.CyclicLR(

        #     optimizer, base_lr=base_lr, max_lr=max_lr,

        #     step_size_up=300, step_size_down=300,

        #     mode='exp_range', gamma=0.99994

        # )

        optimizer = torch.optim.Adam(self.parameters(), lr=(self.base_lr + self.max_lr) / 2)

        batches = torch.utils.data.DataLoader(

            torch.utils.data.TensorDataset(X, y),

            batch_size=self.batch_size, shuffle=True

        )

        

        for epoch in range(self.n_epochs):

            for i, (X_batch, y_batch) in enumerate(batches):

                X_batch = X_batch.cuda()

                y_batch = y_batch.cuda()

                

                y_pred = model(X_batch).squeeze()

                # scheduler.batch_step()  # Disabled due to a bug, see above.

                loss = self.loss_fn(y_pred, y_batch)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            print(

                f"Epoch {epoch + 1}/{self.n_epochs}, Loss {loss.detach().cpu().numpy()}"

            )

    

    

    def predict(self, X):

        self.eval()

        with torch.no_grad():

            y_pred = model(torch.tensor(X, dtype=torch.float32).cuda())

        return y_pred.squeeze()

    

    

    def score(self, X, y):

        y_pred = self.predict(X)

        y = torch.tensor(y, dtype=torch.float32).cuda()

        return self.score_fn(y, y_pred)
model = FeedforwardTabularModel()

model.cuda()

model.fit(X_train_sample.values, y_train_sample.values)