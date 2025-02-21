from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
import torch
import logging

# Initialize Flask app
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load BERT Model and Tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME, output_attentions=True)
logging.info("BERT model and tokenizer loaded successfully.")

@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.json
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        
        # Extract attention
        attentions = outputs.attentions  # Tuple of tensors
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Convert tensors to lists for JSON response
        attention_scores = [attn.tolist() for attn in attentions]
        
        return jsonify({
            "tokens": tokens,
            "attention": attention_scores
        })
    
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
