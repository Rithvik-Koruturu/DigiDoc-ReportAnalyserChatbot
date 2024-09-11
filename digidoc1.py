from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai
from PIL import Image
from PyPDF2 import PdfReader
from io import BytesIO

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
    def get_gemini_response(input_prompt, image_data=None, text_data=None):
        model = genai.GenerativeModel('gemini-1.5-flash')
        if image_data:
            response = model.generate_content([image_data[0], input_prompt])
        elif text_data:
            response = model.generate_content([text_data, input_prompt])
        else:
            response = "No data provided."
        return response.text

    # Function to process the uploaded image
    def input_image_setup(uploaded_file):
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            mime_type = uploaded_file.type
            image_parts = [
                {
                    "mime_type": mime_type,
                    "data": bytes_data
                }
            ]
            return image_parts
        else:
            raise FileNotFoundError("No file uploaded")

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

    # Function to get analysis results for text input based on gender
    def analyze_report(text_data, gender):
        prompt = f"""
        Analyze the following report for a {gender} and extract key information including:
        - Value
        - Normal Value
        - Status of Observation
        - Potential Risks
        - Remedies
        - Specialist Recommendations
        Format the results in a structured manner.
        """
        return get_gemini_response(prompt, text_data=text_data)

    # Initialize Streamlit app
    st.set_page_config(page_title="CliniCheck")

    st.header("CliniCheck-AI Report Analyzer")

    # Initialize session state for analysis results and gender if it doesn't exist
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = ""
    if 'gender' not in st.session_state:
        st.session_state['gender'] = ""

    # Gender selection
    st.subheader("Select Your Gender")
    gender_options = ["Select", "Male", "Female", "Non-binary", "Prefer not to say"]
    selected_gender = st.selectbox("Gender", options=gender_options)
    st.session_state['gender'] = selected_gender

    # File upload functionality
    uploaded_file = st.file_uploader("Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension in ["jpg", "jpeg", "png"]:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)

            # Process image
            image_data = input_image_setup(uploaded_file)
            image_prompt = """
            You are an advanced AI agent capable of analyzing images to extract data. 
            Analyze the uploaded image and provide information including values, normal ranges, status, potential risks, and remedies.
            """
            st.session_state['analysis_results'] = get_gemini_response(image_prompt, image_data=image_data)

        elif file_extension == "pdf":
            text_data = input_pdf_setup(uploaded_file)
            # Process PDF with gender context
            if selected_gender != "Select":
                st.session_state['analysis_results'] = analyze_report(text_data, selected_gender)
            else:
                st.error("Please select a gender to proceed with the analysis.")

        # Display the analysis results
        st.subheader("Analysis Results:")
        analysis_results = st.session_state['analysis_results']
        st.write(analysis_results)



    # Disclaimer
    st.subheader("Disclaimer")
    st.write("""
    **Disclaimer:** The analysis provided by this application is generated using a generative AI model. This analysis is for informational purposes only and should not be considered a substitute for professional medical advice, diagnosis, or treatment. Neither the developer nor the company is responsible for any decisions made based on this analysis. Always consult a qualified healthcare professional for accurate results and advice.
    """)

 

    # Copyright notice
    st.markdown("""
    ---
    &copy; 2024 Rithvik Koruturu. All rights reserved.
    """)
