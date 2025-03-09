from flask import Flask, request, jsonify
import torch
import openai
from openai import OpenAI
import os
from transformers_interpret import SequenceClassificationExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Load BERT model and tokenizer
model_path = "./new_model"
bert_model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load SBERT model for content filtering
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load precomputed dataset embeddings for filtering
dataset_texts = torch.load("./dataset_texts.pt", map_location=torch.device("cpu"))
dataset_embeddings = torch.load("./dataset_embeddings.pt", map_location=torch.device("cpu"))

# Initialize Transformers-Interpret for explainability
cls_explainer = SequenceClassificationExplainer(bert_model, tokenizer)

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()

# Get the OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Print a warning if the key is missing
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY is not set in the environment.")

# Label mapping for classification
label_mapping = {1: "false", 0: "real"}

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the BERT Fine-tuned Model API with SBERT Filtering & Explainability!",
        "instructions": "Use the /predict endpoint with a POST request to classify text and get explanations."
    })

# Content filtering function using SBERT
def is_query_relevant(user_input, min_threshold=0.6, strict_threshold=0.7):
    # Encode only the user query using SBERT
    query_embedding = sbert_model.encode(user_input, convert_to_tensor=True)

    # Compute cosine similarity with precomputed dataset embeddings
    similarities = util.pytorch_cos_sim(query_embedding, dataset_embeddings)

    # Get highest similarity score
    max_similarity, best_match_idx = torch.max(similarities, dim=1)
    max_similarity = max_similarity.item()

    if max_similarity >= strict_threshold:
        return True  # Fully relevant, pass to BERT

    elif max_similarity >= min_threshold:
        return True  # Medium similarity, still pass to BERT

    return False  # Irrelevant query, block it

# Classify text with BERT (Return Both Probabilities)
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

"""
def merge_subword_attributions(attributions):
    merged_tokens = []
    merged_values = []
    prev_word = None
    accumulated_score = 0.0

    for token, score in attributions:
        clean_token = token.replace("##", "")

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
"""

def merge_subword_attributions(attributions):
    """Merges subword tokens properly, preventing extra dots or spaces."""
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


#Function to extract key words using Transformers-Interpret
def extract_key_words(text, top_n=7, min_threshold=0.05):
    """Extracts top influential words from the model's attributions, keeping only positive scores."""
    attributions = cls_explainer(text)
    merged_attributions = merge_subword_attributions(attributions)

    #Sort words by absolute importance (strongest first)
    merged_attributions = sorted(merged_attributions, key=lambda x: abs(x[1]), reverse=True)

    #Keep only positive attributions (words that supported the model's prediction)
    filtered_attributions = [(word, score) for word, score in merged_attributions if score > 0 and abs(score) >= min_threshold]

    #Select the top N positive words
    key_words = [word for word, score in filtered_attributions[:top_n]]

    #Compute total attribution score
    total_attribution_score = sum(abs(score) for _, score in filtered_attributions)

    return key_words, total_attribution_score


def should_explain(input_text, total_attribution_score, alpha=0.1):
    """Determines whether an explanation should be generated based on attribution strength."""
    num_words = len(input_text.split())
    threshold = alpha * num_words 
    return total_attribution_score > threshold 


#Generate Explanation Using GPT-3.5
def generate_gpt_explanation(prediction, key_words, confidence, user_text):
    """Generates a user-friendly explanation using OpenAI's GPT model."""
    
    #Ensure that "false" is mapped correctly
    label_mapping_gpt = {
        "real": "True Information",
        "false": "False Information"
    }
    mapped_prediction = label_mapping_gpt.get(prediction, prediction)  # Use correct label names
    
    if not key_words:
        return f"The model classified this text as '{mapped_prediction}' but did not find strong influencing words for further explanation."

    prompt = f"""
    The model classified this text as '{mapped_prediction}' with {confidence:.2f}% confidence.
    
    **Original sentence:** "{user_text}"

    The most important words influencing this decision were: {', '.join(key_words)}.

    Provide a **concise, context-aware explanation** (50-80 words) about why the model made this prediction.
    - Explain how these words are used in {mapped_prediction.lower()} content.
    - Provide a real-world example of {mapped_prediction.lower()} content using these words.
    - Compare this to scientifically verified content.
    - Keep the explanation clear and understandable.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100
    )

    return response.choices[0].message.content


#Main API: Filtering + Prediction + Explainability
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        is_relevant = is_query_relevant(text)

        if not is_relevant:
            return jsonify({
                "is_relevant": False,
                "message": "The query is not related to diabetes misinformation."
            })
        prediction, probabilities = classify_with_bert(text)

        key_words, total_attribution_score = extract_key_words(text)

        if not should_explain(text, total_attribution_score):
            return jsonify({
                "text": text,
                "is_relevant": True,
                "predicted_label": prediction,
                "probabilities": {
                    "false": probabilities[1],  # False label
                    "real": probabilities[0]   # Real label
                },
                "key_words": key_words,
                "explanation": "The model's attribution scores were too low for a reliable explanation."
            })

        gpt_explanation = generate_gpt_explanation(prediction, key_words, probabilities[1], text)
        response = {
            "text": text,
            "is_relevant": True,
            "predicted_label": prediction,
            "probabilities": {
                "false": probabilities[1],  # False label
                "real": probabilities[0]   # Real label
            },
            "key_words": key_words,
            "explanation": gpt_explanation
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



#Run Flask app
if __name__ == "__main__":
    app.run(debug=False)
