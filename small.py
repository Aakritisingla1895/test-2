from flask import Flask, request, jsonify
from transformers import BertTokenizer
from siamese_bert_model import SiameseBert
import torch

app = Flask(__name__)

# Load the trained model
model = SiameseBert()
model.load_state_dict(torch.load('small_siamese_bert_model.pth'))
model.eval()

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def simulate_prediction(text1, text2):
    # Tokenize and prepare inputs
    inputs = tokenizer(text1, text2, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].squeeze()
    attention_mask = inputs["attention_mask"].squeeze()

    # Make prediction
    with torch.no_grad():
        output = model(input_ids, attention_mask)

    # Use sigmoid activation for similarity score
    similarity_score = torch.sigmoid(output).item()

    return similarity_score

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Handle POST requests
        data = request.get_json()
        text1 = data.get('text1', '')
        text2 = data.get('text2', '')

        # Use the simulate_prediction function
        similarity_score = simulate_prediction(text1, text2)

        # Return the similarity score
        return jsonify({"similarity_score": similarity_score})

    elif request.method == 'GET':
        # Handle GET requests
        text1 = "nuclear body seeks new tech"
        text2 = "terror suspects face arrest"

        # Use the simulate_prediction function
        test_similarity_score = simulate_prediction(text1, text2)

        return jsonify({"test_similarity_score": test_similarity_score})

if __name__ == '__main__':
    app.run(debug=True)
