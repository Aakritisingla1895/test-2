# app.py

from flask import Flask, render_template, request
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


app = Flask(__name__)


# Load the pre-trained Siamese BERT model
class SiameseBert(nn.Module):
    def __init__(self):
        super(SiameseBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768 * 2, 1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)["last_hidden_state"]
        output = torch.cat((output[:, 0, :], output[:, 1, :]), dim=1)
        similarity_score = torch.sigmoid(self.fc(output))
        return similarity_score

model = SiameseBert()
model.load_state_dict(torch.load('siamese_bert_model.pth', map_location=torch.device('cpu')))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def calculate_similarity(text1, text2):
    inputs = tokenizer(text1, text2, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    with torch.no_grad():
        similarity_score = model(inputs["input_ids"], inputs["attention_mask"]).item()
    return similarity_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text1 = request.form['text1']
        text2 = request.form['text2']
        similarity_score = calculate_similarity(text1, text2)
        return render_template('index.html', text1=text1, text2=text2, similarity_score=similarity_score)

if __name__ == '__main__':
    app.run()
