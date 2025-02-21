# Bert-Implementation
# Enhanced Attention Visualization with BERT
This project implements BERT-based attention visualization to analyze how words interact within a sentence. It provides a user-friendly Streamlit interface that connects to a Flask backend, utilizing BertViz for interactive visualization of self-attention across different transformer layers.

# Features
1. Text Tokenization & Attention Processing
Users can input a sentence, which is tokenized using BERT Tokenizer.
The model extracts attention weights across all layers and heads.
2. Flask Backend for Model Processing
A Flask API processes input text, runs it through BERT, and returns tokens & attention weights.
API Endpoint: POST /analyze (accepts JSON input and returns attention data).
3. Streamlit Frontend for Interactive Visualization
Users can enter text and visualize BERT’s attention mechanism.
BertViz is used to generate attention maps for each layer and head.
Outputs:
Tokenized Text
Attention Weights
Interactive Attention Visualization
4. Attention Visualization with BertViz
Uses bertviz.head_view() to render attention heatmaps.
Enables layer-wise and head-wise navigation for deeper insights.
5. Debugging & Optimization
Handles tensor conversions properly to prevent NumPy and PyTorch conflicts.
Error handling for API failures, missing dependencies, and improper input formatting.
Installation & Setup
1. Clone the Repository
git clone https://github.com/yourusername/enhanced-attention-visualization.git
cd enhanced-attention-visualization

2. Install Dependencies
pip install -r requirements.txt

3. Run the Flask Backend
python backend.py

Expected output:
Running on http://127.0.0.1:5000/

4. Run the Streamlit Frontend
streamlit run frontend.py
Expected output:
-> The frontend UI should open in the browser.
-> The Flask backend should be running before starting the frontend.
Usage
1. Open the Streamlit app in the browser.
2. Enter a sentence in the text box.
3. Click Analyze to generate tokenization and attention visualization.
4. Navigate through different layers and heads to explore how BERT attends to words.

# Project Structure
📂 enhanced-attention-visualization
 ┣ 📜 backend.py          # Flask API for processing text with BERT
 ┣ 📜 frontend.py         # Streamlit UI for user interaction
 ┣ 📜 requirements.txt    # List of required dependencies
 ┣ 📜 README.md           # Project documentation
 ┗ 📂 screenshots         # Placeholder for screenshots
Screenshots
1. Streamlit Interface

2. Tokenized Text Output

3. Attention Visualization

Known Issues & Troubleshooting
1. Attention Visualization Not Displaying
Ensure bertviz is installed properly:
pip install bertviz
Restart the Flask backend before running Streamlit.

2. API Connection Error
Ensure Flask is running before opening Streamlit.
netstat -an | findstr 5000
If no output appears, restart the Flask server.

3. BertViz Not Rendering in Streamlit
Convert html_output properly:
from IPython.core.display import HTML
if isinstance(html_output, HTML):
    html_output = html_output.data  # Extract raw HTML string
Ensure Streamlit supports external components.

# Future Enhancements
Fine-tune BERT for custom datasets.
Deploy the app online using Hugging Face Spaces or Streamlit Cloud.
Improve real-time visualization performance.

# Contributors
Vinay Joshva – Frontend & BERT Tokenizer with BertViz.
Hrishikesh Komaragiri - Backend & BERT Tokenizer with BertViz.