import streamlit as st
import requests
from bertviz import head_view
import torch
from IPython.core.display import HTML

# Flask API endpoint
API_URL = "http://127.0.0.1:5000/analyze"

# Streamlit UI
st.title("Enhanced Attention Visualization with BERT")
st.write("Enter text to analyze how BERT pays attention to different words.")

# User input
user_text = st.text_area("Enter your text here:", "Big Data Analytics is an exciting field of research.")

if st.button("Analyze"):
    if user_text.strip():
        with st.spinner("Processing..."):
            response = requests.post(API_URL, json={"text": user_text})

            if response.status_code == 200:
                data = response.json()
                tokens = data["tokens"]
                
                # Ensure attention data remains PyTorch tensors
                attention = [torch.tensor(layer) for layer in data["attention"]]

                st.write("### Tokenized Text:")
                st.write(tokens)

                st.write("### Attention Visualization:")
                st.write("Attention Shape:", [attn.shape for attn in attention])

                html_output = head_view(attention, tokens, html_action="return")

                # Convert IPython HTML object to raw HTML string
                if isinstance(html_output, HTML):
                    html_output = html_output.data  # Extract string

                if isinstance(html_output, str):
                    st.components.v1.html(html_output, height=800, scrolling=True)

                    # Save HTML file for debugging
                    with open("attention_visualization.html", "w", encoding="utf-8") as f:
                        f.write(html_output)
                    st.markdown("[Open Attention Visualization](./attention_visualization.html)")
                else:
                    st.error("Error: Attention visualization could not be generated.")

                # Debugging Output
                st.write("HTML Output Type:", type(html_output))

            else:
                st.error("Error processing the text. Please try again.")
    else:
        st.warning("Please enter some text before analyzing.")
