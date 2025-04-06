from flask import Flask, request, jsonify
import torch
import openai
import os
import numpy as np
import onnxruntime as ort
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer
from sentence_transformers import SentenceTransformer, util
from transformers.modeling_outputs import SequenceClassifierOutput
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

#Global variables
tokenizer = None
onnx_model = None
pytorch_model = None
explainer = None
sbert_model = None
dataset_embeddings = None
dataset_texts = None
client = None
resources_loaded = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_mapping = {0: "real", 1: "false", 2: "partially_true"}

#Loading ONNX and SBERT and the Tokenizer
def load_resources():
    global tokenizer, onnx_model, sbert_model, dataset_embeddings, dataset_texts, client, resources_loaded

    if resources_loaded:
        return

    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    onnx_model = ort.InferenceSession("./newbiobert_finetuned_3class.onnx")

    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    dataset_embeddings = torch.load("./combined_embeddings.pt", map_location="cpu")
    dataset_texts = torch.load("./combined_texts.pt", map_location="cpu")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
        client = openai  
    else: 
        print("No OpenAI API key found.") 

    resources_loaded = True

@app.before_request
def ensure_resources_loaded():
    if not resources_loaded:
        load_resources()

CORS(app, origins=["chrome-extension://pkbfgfhddafhlnndmcnhahnokddbgjjo"],
     allow_headers=["Content-Type"], supports_credentials=True, methods=["GET", "POST", "OPTIONS"])

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Hybrid ONNX + Explainable BioBERT API is running!"})

#Classification using ONNX
def classify_with_onnx(text):
    encoded = tokenizer(text, return_tensors="np", truncation=True, padding="max_length", max_length=128)
    inputs_onnx = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    logits = onnx_model.run(["logits"], inputs_onnx)[0]
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    pred_index = int(np.argmax(probs))
    label = label_mapping[pred_index]
    return label, {"real": round(probs[0][0]*100, 2), "false": round(probs[0][1]*100, 2), "partially_true": round(probs[0][2]*100, 2)}


#Custom Model arhitecture
class FineTuneModel(nn.Module):
    def __init__(self, num_classes=3, hidden_size=768):
        super(FineTuneModel, self).__init__()
        self.encoder = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        if attention_mask is None:
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)

        return SequenceClassifierOutput(logits=logits)

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()



# Lay loading of PyTorch model for attribution
def get_explainer():
    global pytorch_model, explainer

    if explainer is None:
        pytorch_model = FineTuneModel(num_classes=3)
        pytorch_model.load_state_dict(
            torch.load("./newbiobert_model_3class/newbiobert_model_3class/pytorch_model.bin", map_location="cpu")
        )
        pytorch_model.eval()

        #Adding the required Hugging Face attributes after several failures of the model
        config = AutoConfig.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        pytorch_model.config = config
        pytorch_model.base_model_prefix = "bert" 
        pytorch_model.device = torch.device("cpu")  
        pytorch_model.bert = pytorch_model.encoder 

        explainer = SequenceClassificationExplainer(pytorch_model, tokenizer)

    return explainer


#Explainability module
def get_explanation(text, label, confidence):
    explainer = get_explainer()
    attributions = explainer(text)
    top_words = sorted(attributions, key=lambda x: abs(x[1]), reverse=True)[:7]
    keywords = [w for w, s in top_words if s > 0.05]

    label_map = {
        "real": "True Information",
        "false": "False Information",
        "partially_true": "Partially True / Misleading Information"
    }
    mapped_label = label_map.get(label, label)

    if not client:
        return "OpenAI key not set.", keywords
    
    prompt = f"""
    The model classified the following text as '{mapped_label}' with {confidence:.1f}% confidence.

    Text: "{text}"

    The model highlighted these key words: {', '.join(keywords)}.

    Your task is to explain why the model likely made this prediction, based solely on highlighted keywords. 
    Do not introduce new keywords or contradict the model’s decision. 

    Then, add a short, informative follow-up sentence based on the text. This sentence should relate directly to the *main subject or health claim in the text*.
    """



    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=120
        )
        return response.choices[0].message.content.strip(), keywords
    except Exception as e:
        print("GPT Error:", e)
        return "Explanation failed.", keywords


#SBERT relevance check
def is_query_relevant(text, threshold=0.6):
    embedding = sbert_model.encode(text, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(embedding, dataset_embeddings)
    return torch.max(scores).item() >= threshold


#Main predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400

    #Normalize: add punctuation if missing
    if not text.endswith((".", "!", "?")):
        text += "."

    #Normalize: capitalize first letter for better model consistency
    text = text[0].upper() + text[1:]

    if not is_query_relevant(text):
        return jsonify({
            "is_relevant": False,
            "message": "Query not related to diabetes."
        })

    label, raw_probs = classify_with_onnx(text)

    #float32 values are converted to native floats
    probs = {
        "real": float(raw_probs["real"]),
        "false": float(raw_probs["false"]),
        "partially_true": float(raw_probs["partially_true"])
    }

    explanation, keywords = get_explanation(text, label, probs[label])

    return jsonify({
        "text": text,
        "is_relevant": True,
        "predicted_label": label,
        "probabilities": probs,
        "key_words": keywords,
        "explanation": explanation
    })

if __name__ == "__main__":
    app.run(debug=False)
