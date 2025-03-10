from flask import Flask, request, jsonify
import torch
import openai
from openai import OpenAI
import os
from transformers_interpret import SequenceClassificationExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# Flask app initialization
app = Flask(__name__)

# Global variables for models and embeddings
bert_model = None
tokenizer = None
sbert_model = None
dataset_embeddings = None
dataset_texts = None
cls_explainer = None
device = None
client = None
resources_loaded = False

label_mapping = {1: "false", 0: "real"}

# Function to load models and resources once
def load_resources():
    global bert_model, tokenizer, sbert_model, dataset_embeddings, dataset_texts, cls_explainer, device, client, resources_loaded
    
    if resources_loaded:
        return  # Resources already loaded
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load BERT model and tokenizer
    model_path = "./new_model"
    bert_model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    bert_model.eval()
    
    # Load SBERT model
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load dataset embeddings
    dataset_embeddings = torch.load("./dataset_embeddings.pt", map_location="cpu")
    dataset_texts = torch.load("./dataset_texts.pt", map_location="cpu")
    
    # Initialize explainability model
    cls_explainer = SequenceClassificationExplainer(bert_model, tokenizer)
    
    # OpenAI Client
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        print("Warning: OPENAI_API_KEY is not set in the environment.")
    
    resources_loaded = True

@app.before_request
def ensure_resources_loaded():
    if not resources_loaded:
        load_resources()


CORS(app, origins=["chrome-extension://pkbfgfhddafhlnndmcnhahnokddbgjjo"],
     allow_headers=["Content-Type"],
     supports_credentials=True,
     methods=["GET", "POST", "OPTIONS"])

from flask import jsonify

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not Found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal Server Error"}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the BERT Fine-tuned Model API with SBERT Filtering & Explainability!",
        "instructions": "Use the /predict endpoint with a POST request to classify text and get explanations."
    })

def is_query_relevant(user_input, min_threshold=0.6, strict_threshold=0.7):
    query_embedding = sbert_model.encode(user_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, dataset_embeddings)
    max_similarity = torch.max(similarities).item()
    return max_similarity >= min_threshold

def classify_with_bert(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding="max_length", max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

    predicted_label_index = probabilities.argmax()
    predicted_label = label_mapping[predicted_label_index]
    probabilities_percentage = (probabilities * 100).round(2).tolist()

    return predicted_label, probabilities_percentage

def merge_subword_attributions(attributions):
    merged_tokens = []
    merged_values = []
    prev_word = None
    accumulated_score = 0.0

    for token, score in attributions:
        clean_token = token.replace("##", "").strip(".")
        if token.startswith("##") and prev_word:
            prev_word += clean_token
            accumulated_score += score
        else:
            if prev_word:
                merged_tokens.append(prev_word)
                merged_values.append(accumulated_score)
            prev_word = clean_token
            accumulated_score = score

    if prev_word:
        merged_tokens.append(prev_word)
        merged_values.append(accumulated_score)

    return list(zip(merged_tokens, merged_values))

def extract_key_words(text, top_n=7, min_threshold=0.05):
    attributions = cls_explainer(text)
    merged_attributions = merge_subword_attributions(attributions)
    merged_attributions = sorted(merged_attributions, key=lambda x: abs(x[1]), reverse=True)
    filtered_attributions = [(word, score) for word, score in merged_attributions if score > 0 and abs(score) >= min_threshold]
    key_words = [word for word, score in filtered_attributions[:top_n]]
    total_attribution_score = sum(abs(score) for _, score in filtered_attributions)
    return key_words, total_attribution_score

def should_explain(input_text, total_attribution_score, alpha=0.1):
    return total_attribution_score > (alpha * len(input_text.split()))

def generate_gpt_explanation(prediction, key_words, confidence, user_text):
    if not client:
        return "OpenAI API Key not provided. Cannot generate explanation."

    label_mapping_gpt = {"real": "True Information", "false": "False Information"}
    mapped_prediction = label_mapping_gpt.get(prediction, prediction)

    if not key_words:
        return f"The model classified this text as '{mapped_prediction}' but found no strong influencing words for explanation."

    prompt = f"""
    The model classified this text as '{mapped_prediction}' with {confidence:.2f}% confidence.

    **Original sentence:** "{user_text}"

    The most important words influencing this decision were: {', '.join(key_words)}.

    Provide a **concise, context-aware explanation** (50-80 words) about why the model made this prediction.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )
        explanation = response.choices[0].message.content.strip()
        return explanation

    except Exception as e:
        print("OpenAI API Error:", e)
        return "Failed to generate an explanation due to API error."


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if not is_query_relevant(text):
        return jsonify({
            "is_relevant": False,
            "message": "The query is not related to diabetes misinformation."
        })

    prediction, probabilities = classify_with_bert(text)
    key_words, total_attribution_score = extract_key_words(text)

    explanation = (
        generate_gpt_explanation(prediction, key_words, probabilities[1], text)
        if should_explain(text, total_attribution_score)
        else "The model's attribution scores were too low for a reliable explanation."
    )

    return jsonify({
        "text": text,
        "is_relevant": True,
        "predicted_label": prediction,
        "probabilities": {"false": probabilities[1], "real": probabilities[0]},
        "key_words": key_words,
        "explanation": explanation
    })


# Run Flask app
if __name__ == "__main__":
    app.run(debug=False)
