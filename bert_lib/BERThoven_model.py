import torch
from torch import nn

from .BERT_utils import get_new_bert_model


class BERThoven(nn.Module):
    """
        sum_outputs: run BERT with text in both directions, then sum the outputs for the fc layer
        concat_outputs: run BERT with text in both directions, then concatenate the outputs for the fc layer
        cls: use the [CLS] output (instead of the pooled output)
    """

    def __init__(self, sum_outputs=False, concat_outputs=False, cls=False, dropout=True, dropout_prob=0.5):
        super(BERThoven, self).__init__()
        if sum_outputs and concat_outputs:
            raise RuntimeError("You can't both sum and concatenate outputs.")

        self.bert_layers = get_new_bert_model()
        bert_out_features = self.bert_layers.pooler.dense.out_features
        if concat_outputs:
            self.lin_layer = nn.Linear(bert_out_features * 2, 1)
        else:
            self.lin_layer = nn.Linear(bert_out_features, 1)

        self.droupout_layer = nn.Dropout(p=dropout_prob)

        self.droupout = dropout
        self.both_ways = sum_outputs or concat_outputs
        self.sum_outputs = sum_outputs
        self.concat_outputs = concat_outputs
        self.get_bert_output = lambda x: x[0][:, 0, :] if cls else x[1]

    def forward(self, x1, x2):
        # The 1 index is for the pooled head
        out1a = self.bert_layers(x1[0], attention_mask=x1[1])
        out1a = self.get_bert_output(out1a)
        if not self.both_ways:
            out1x = out1a
        else:
            out1b = self.bert_layers(x2[0], attention_mask=x2[1])
            out1b = self.get_bert_output(out1b)
            if self.concat_outputs:
                out1x = torch.cat((out1a, out1b), 1)
            else:
                out1x = out1a + out1b

        out2 = self.droupout_layer(out1x) if self.droupout else out1x
        out2 = self.lin_layer(out2)
        out2 = torch.sigmoid(out2)
        return out2.squeeze()
