from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from io import BytesIO
import pandas as pd

# Load environment variables (like your Google API key)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Check if the API key is set; if not, display an error message
if not api_key:
    st.error("API key is not set. Please check your .env file.")
else:
    # Configure the generative AI model with the provided API key
    genai.configure(api_key=api_key)

    # Function to get the response from the generative AI model for images and text
    def get_gemini_response(input_prompt, text_data=None):
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([text_data, input_prompt])
        return response.text

    # Function to process the uploaded PDF
    def input_pdf_setup(uploaded_file):
        if uploaded_file is not None:
            pdf_reader = PdfReader(BytesIO(uploaded_file.read()))  # Use PdfReader instead of PdfFileReader
            text_data = ""
            for page in pdf_reader.pages:
                text_data += page.extract_text()
            return text_data
        else:
            raise FileNotFoundError("No file uploaded")

    # Function to extract and analyze insights from the PDF based on the paper structure
    def analyze_paper(text_data):
        prompt = """
        Analyze the following research paper and extract the following information under the specified headings:

        - Problem Statement: Summarize the core problem the paper is addressing.
        - Literature Survey: Summarize the key papers referenced in the literature survey. Present them in a table with columns: Paper Title, Methodology Used, Datasets, Performance Metrics, and Limitations.
        - Methodology: Summarize the methodology used in the paper.
        - Dataset: Describe the dataset used in the paper.
        - Performance Metrics: Outline the performance metrics used to evaluate the approach.
        - Limitations: Highlight the limitations mentioned in the paper.
        """
        return get_gemini_response(prompt, text_data=text_data)

    # Initialize Streamlit app
    st.set_page_config(page_title="Research Paper Analyzer")

    st.header("Research Paper Analyzer")

    # Initialize session state for analysis results if it doesn't exist
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = ""

    # File upload functionality
    uploaded_file = st.file_uploader("Upload a research paper (PDF)...", type=["pdf"])

    if uploaded_file is not None:
        # Process PDF
        text_data = input_pdf_setup(uploaded_file)

        # Analyze the paper and extract insights
        st.session_state['analysis_results'] = analyze_paper(text_data)

        # Display the analysis results
        st.subheader("Analysis Results:")
        analysis_results = st.session_state['analysis_results']
        st.write(analysis_results)

    # Disclaimer
    st.subheader("Disclaimer")
    st.write("""
    *Disclaimer:* The analysis provided by this application is generated using a generative AI model. This analysis is for informational purposes only and should not be considered a substitute for professional academic advice or critical reading. Always review the paper thoroughly for accurate insights.
    """)

    # Copyright notice
    st.markdown("""
    ---
    &copy; 2024 Rithvik Koruturu. All rights reserved.
    """)
