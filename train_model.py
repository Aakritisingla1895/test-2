# train_model.py

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load data from CSV file
df = pd.read_csv('data.csv')

# Sample unlabeled dataset
data = []
for index, row in df.iterrows():
    data.append({"text1": row['text1'], "text2": row['text2']})

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Custom dataset class
class ParagraphDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx]
        text1 = pair["text1"]
        text2 = pair["text2"]

        # Tokenize and truncate/pad to the specified max sequence length
        inputs = self.tokenizer(text1, text2, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_seq_length)
        return {"input_ids": inputs["input_ids"].squeeze(), "attention_mask": inputs["attention_mask"].squeeze()}

# Create separate datasets and dataloaders for training and testing
train_dataset = ParagraphDataset(train_data, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = ParagraphDataset(test_data, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)  # No need to shuffle the test set

# Siamese network model
class SiameseBert(nn.Module):
    def __init__(self):
        super(SiameseBert, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(768 * 2, 1)  # Change the linear layer input size to handle concatenation

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)["last_hidden_state"]  # Get the output embeddings
        output = torch.cat((output[:, 0, :], output[:, 1, :]), dim=1)  # Concatenate embeddings for text1 and text2
        similarity_score = torch.sigmoid(self.fc(output))  # Sigmoid activation for similarity score
        return similarity_score

# Initialize the model, optimizer, and loss function
model = SiameseBert()
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = nn.BCELoss()  # Binary cross-entropy loss for binary classification

# Training loop
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1} (Training)"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        
        # Create pseudo-labels (assuming all pairs are dissimilar)
        pseudo_labels = torch.zeros_like(outputs)

        loss = criterion(outputs, pseudo_labels)
        loss.backward()
        optimizer.step()

# Save the trained model
torch.save(model.state_dict(), 'siamese_bert_model.pth')
