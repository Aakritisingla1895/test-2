# siamese_bert_model.py
import torch
import torch.nn as nn
from transformers import BertModel

class SiameseBert(nn.Module):
    def __init__(self):
        super(SiameseBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768 * 2, 1)  # Change the linear layer input size to handle concatenation

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)["last_hidden_state"]  # Get the output embeddings
        output = torch.cat((output[:, 0, :], output[:, 1, :]), dim=1)  # Concatenate embeddings for text1 and text2
        similarity_score = torch.sigmoid(self.fc(output))  # Sigmoid activation for similarity score
        return similarity_score
