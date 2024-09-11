from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai
from google.generativeai import types as generation_types
from PIL import Image
import pdfplumber

# Load environment variables (like your Google API key)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Check if the API key is set; if not, display an error message
if not api_key:
    st.error("API key is not set. Please check your .env file.")
else:
    # Configure the generative AI model with the provided API key
    genai.configure(api_key=api_key)


def analyze_report_content(report_text, gender):
    """
    Analyzes report content using Gemini Pro and provides details like observations, status, risks, remedies, and specialist suggestions.

    Args:
        report_text (str): Text extracted from the uploaded report.
        gender (str): Patient's gender.

    Returns:
        str: Analyzed report with details for each observation.
    """

    analysis_prompt = f"""
    You are an advanced AI medical assistant. Given the following report text, analyze each observation in the following format:

    Observation Name - Value with Units - Normal Ranges with Units - Status (normal/slightly over the border/slightly below the border/over the border/below the border).

    Additionally, provide:
      - Potential Risks associated with the report findings.
      - Remedies to avoid these potential risks.
      - Suggest which specialist doctor to consult if needed.

    Patient Gender: {gender}

    Report Text:
    {report_text}
    """

    try:
        model = genai.GenerativeModel("gemini-pro")
        chat = model.start_chat(history=[])

        # Summarize long reports (optional)
        if len(report_text) > 1000:
            report_text = summarize_report(report_text)  # Implement a summarizing function

        # Split report into smaller chunks for processing
        chunks = [report_text[i:i + 1000] for i in range(0, len(report_text), 1000)]
        responses = []

        for chunk in chunks:
            response = chat.send_message(analysis_prompt.replace("{report_text}", chunk))
            responses.append(response.text)

        return "\n".join(responses)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return ""  # Return an empty string on error


def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from all pages of a PDF file, including after blank pages.

    Args:
        uploaded_file (streamlit.UploadedFile): Uploaded PDF file.

    Returns:
        str: Extracted text from the PDF file.
    """

    extracted_text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page_number, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                extracted_text += f"\n\nPage {page_number + 1}:\n{page_text}"
    return extracted_text


def handle_image_uploads(uploaded_files):
    """
    Processes uploaded images and analyzes them using Gemini Flash.

    Args:
        uploaded_files (list): List of uploaded images.

    Returns:
        str: Image analysis context from the Gemini Flash model.
    """

    image_context = ""
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            image_data = input_image_setup(uploaded_file)
            image_context += get_gemini_image_response("Analyze the image content", image_data)
    return image_context


def input_image_setup(uploaded_file):
    """
    Processes the uploaded image for sending to the Gemini Flash model.

    Args:
        uploaded_file (streamlit.UploadedFile): Uploaded image file.

    Returns:
        list: List containing image data in the required format.
    """

    if
