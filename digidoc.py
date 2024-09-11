from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai
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

    # Function to analyze the report content based on gender
    def analyze_report_content(report_text, gender):
        analysis_prompt = f"""
        You are an advanced AI medical assistant. Given the following report text and the patient's gender, analyze the values, 
        identify normal ranges specific to the gender, and display these ranges alongside the values. Also, provide any 
        differential ranges, and determine if the values are normal or abnormal based on these ranges. Suggest remedies 
        to avoid risks and recommend specialist doctors if needed.
        
        Patient Gender: {gender}
        Report Text:
        {report_text}
        """
        model = genai.GenerativeModel("gemini-pro")
        chat = model.start_chat(history=[])
        response = chat.send_message(analysis_prompt, stream=True)
        return response

    # Function to extract text from a PDF file
    def extract_text_from_pdf(uploaded_file):
        extracted_text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                extracted_text += page.extract_text() + "\n"
        return extracted_text

    # Function to handle image uploads
    def handle_image_uploads(uploaded_files):
        image_context = ""
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                image_data = input_image_setup(uploaded_file)
                image_context += get_gemini_image_response("Analyze the image content", image_data)
        return image_context

    # Function to process the uploaded image
    def input_image_setup(uploaded_file):
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            mime_type = uploaded_file.type
            image_parts = [{"mime_type": mime_type, "data": bytes_data}]
            return image_parts
        else:
            raise FileNotFoundError("No file uploaded")

    # Function to get response from the generative AI model for images
    def get_gemini_image_response(input_prompt, image_data=None):
        model = genai.GenerativeModel('gemini-1.5-flash')
        if image_data:
            response = model.generate_content([image_data[0], input_prompt])
            return response.text
        else:
            return "No image data provided."

    # Function to handle user queries with context from reports
    def get_response_with_context(question, report_text=None, image_context=None, gender=None):
        model = genai.GenerativeModel("gemini-pro")
        chat = model.start_chat(history=[])
        
        # Combine context from reports, images, and gender
        combined_context = f"Patient Gender: {gender}\n"
        if report_text:
            combined_context += f"Report Data: {report_text}\n"
        if image_context:
            combined_context += f"Image Analysis: {image_context}\n"
        
        # Formulate the final prompt for the model
        final_prompt = f"""
        You are an advanced AI medical assistant. Use the following data extracted from the reports, images, 
        and gender information to answer the user's question comprehensively. Provide relevant information, 
        possible diagnoses, and suggest specialist doctors if needed.

        {combined_context}

        Question: {question}
        """
        response = chat.send_message(final_prompt, stream=True)
        return response

    # Initialize Streamlit app
    st.set_page_config(page_title="Report Analyzer Chatbot")
    st.header("Report Analyzer Chatbot")

    # Patient gender selection
    gender = st.selectbox("Select the patient's gender:", ["Male", "Female", "Other"])

    # Allow multiple images and PDF upload
    uploaded_files = st.file_uploader("Upload images or a PDF report...", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

    report_text = ""
    image_context = ""
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                report_text += extract_text_from_pdf(uploaded_file)
            else:
                image_context += handle_image_uploads([uploaded_file])

    # Process the report content if available
    if report_text:
        response = analyze_report_content(report_text, gender)
        st.subheader("Report Analysis:")
        report_lines = []
        for chunk in response:
            for line in chunk.text.splitlines():
                st.write(line)
                report_lines.append(line)

        # Display a button to print the report
        if st.button("Print Report"):
            print_script = f"""
            <script>
                window.print();
            </script>
            """
            st.markdown(print_script, unsafe_allow_html=True)

    # Process image context if available
    if image_context:
        st.subheader("Image Analysis:")
        for line in image_context.splitlines():
            st.write(line)

    # Unified input field for additional queries
    user_input = st.text_input("Ask a question related to the report or health:")

    # Button to get a response
    if st.button("Get Response"):
        if user_input:
            response = get_response_with_context(user_input, report_text, image_context, gender)
            for chunk in response:
                for line in chunk.text.splitlines():
                    st.write(line)
