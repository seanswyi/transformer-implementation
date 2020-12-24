import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNN(nn.Module):
    def __init__(self, args):
        super(FeedForwardNN, self).__init__()

        self.d_model = args.d_model
        self.d_ff = args.d_ff

        self.linear1 = nn.Linear(in_features=self.d_model, out_features=self.d_ff)
        self.linear2 = nn.Linear(in_features=self.d_ff, out_features=self.d_model)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        output1 = self.dropout(F.relu(self.linear1(x)))
        output2 = self.linear2(output1)

        return output2
