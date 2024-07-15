import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self,
                 output_dim,
                 model_id_or_path):
        super(BertClassifier, self).__init__()
        self.feature_extractor = BertModel.from_pretrained(model_id_or_path)
        self.dropout = nn.Dropout(self.feature_extractor.config.hidden_dropout_prob)
        self.fc = nn.Linear(self.feature_extractor.config.hidden_size, output_dim)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, encoded_inputs):
        pooler_outputs = self.feature_extractor(**encoded_inputs).last_hidden_state[:, 0, :]
        outputs = self.dropout(pooler_outputs)
        return self.fc(outputs)