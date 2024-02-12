# app.py

from flask import Flask, render_template, request
import torch
from flask import jsonify # <- `jsonify` instead of `json`
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
def calculate_similarity_endpoint():
   
    text1 = 'rap boss arrested over drug find rap mogul marion  suge  knight has been arrested for violating his parole after he was allegedly found with marijuana.  he was arrested in barstow  california  on saturday following an alleged traffic offence. he is expected to be transferred to a state prison while a decision is made on whether he should be released. mr knight  founder of death row records  served a 10-month jail term in 2004 for punching a man while on parole for an assault conviction. police said mr knight was stopped on saturday after performing an illegal u-turn and a search of his car allegedly found marijuana.  he is also accused of not having insurance. a 18-year-old woman in the car was arrested for providing false information and having a fake id card. she was later released. it was his second alleged violation  having previously served half of a nine-year sentence for breaking the terms of his parole. mr knight  39  was jailed in october 1996 following his involvement in a fight with a rival gang just hours before rapper tupac shakur was killed in a las vegas drive-by shooting. he was driving shakur s car at the time and was shot in the head. at the time he was on probation for assaulting two musicians. mr knight  a former bodyguard  set up death row records in the early 1990s with shakur and dr dre among his protegees. but the label has always been dogged by allegations it supports gang culture and fuels the east and west coast rap rivalry'
    text2 = 'amnesty chief laments war failure the lack of public outrage about the war on terror is a powerful indictment of the failure of human rights groups  amnesty international s chief has said.  in a lecture at the london school of economics  irene khan said human rights had been flouted in the name of security since 11 september  2001. she said the human rights movement had to use simpler language both to prevent scepticism and spread a moral message. and it had to fight poverty  not just focus on political rights for elites.  ms khan highlighted detentions without trial  including those at the us camp at guantanamo bay in cuba  and the abuse of prisoners as evidence of increasing human rights problems.  what s a new challenge is the way in which this age-old debate on security and human rights has been translated into the language of war   she said.  by using the language of war  human rights are being sidelined because we know human rights do not apply in times of war.  ms khan said such breaches were infectious and were now seen in almost very major country in the world.  the human rights movement faces a crisis of faith in the value of human rights   she said. that was accompanied by a crisis of governance  where the united nations system did not seem able to hold countries to account.  the amnesty secretary-general said a growing gap between the perceived influence of human rights group and what they could actually achieve was fuelling scepticism.  public passivity on the war against terror is the single most powerful indictment on the failures of human rights groups   she said. ms khan said the movement had failed to mobilise public outrage about what was happening to the human rights system. there needed to be a drive to use simpler language  talking about the basic morality of the issues rather than the complexity of legal processes. such efforts could make the issues more relevant to people across the world  she said.  the human rights groups also had to recognise there were new groups which had to be tackled in new ways as power dripped away from state governments. al-qaeda  for example  was not going to be impressed by a traditional amnesty letter writing campaign. more also needed to be done to develop a human rights framework for international business corporations. amnesty international members voted in 2001 to extend the organisation s work from political and civil rights to cover social and economic rights too. ms khan said the human rights movement would make itself irrelevant if it turned away from the suffering caused by economic strife.  we would be an elitist bunch working for the elites  for those who cannot read the newspaper of their choice rather than those who cannot read   she said. despite her concerns  ms khan dubbed herself a  hope-monger   saying she was confident the passions of the human rights movement could overcome the new challenges.'
    similarity_score = calculate_similarity(text1, text2)
    response_data = {'text1': text1, 'text2': text2, 'similarity score': similarity_score}
    return jsonify(response_data)

if __name__ == '__main__':
    app.run()