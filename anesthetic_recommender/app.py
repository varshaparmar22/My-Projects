import streamlit as st
import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import database as db
from twilio.rest import Client
from langchain_ollama import OllamaLLM

# --- Configuration and Setup ---
load_dotenv()

# for paid account of huggingface - Check for Hugging Face API Token
# if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
#     st.error("Hugging Face API token not found! Please add it to your .env file.")
#     st.stop()


# Initialize the LLM from Hugging Face Hub
try:
    llm = OllamaLLM(model="gemma3")
    #### If you have paid account of HuggingFace then use this as in free account it will not work
    # as all LLM are not accessible under free account #####
    # llm = HuggingFaceEndpoint( 
    #     repo_id="google/flan-t5-xl",
    #     temperature=0.7,
    #     max_new_tokens=1024
    # )

except Exception as e:
    st.error(f"Failed to initialize the LLM. Please check your API token. Error: {e}")
    st.stop()

# --- Notification Functions ---
def send_sms(recipient_phone, call_details):
    """Sends an availability request SMS using Twilio."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_phone = os.getenv("TWILIO_PHONE_NUMBER")

    if not all([account_sid, auth_token, twilio_phone]):
        st.warning("Twilio credentials are not fully set. Cannot send SMS.")
        return False

    body = f"Urgent anesthetic call. Details: {call_details}. Please reply YES/NO with your availability."

    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=body,
            from_=twilio_phone,
            to=recipient_phone
        )
        st.info(f"SMS sent to {recipient_phone}.")
        return True
    except Exception as e:
        st.error(f"Failed to send SMS to {recipient_phone}. Error: {e}")
        return False

def send_email(recipient_email, call_details):
    """Sends an availability request email to a substitute."""
    gmail_address = os.getenv("GMAIL_ADDRESS")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD")

    if not all([gmail_address, gmail_password]):
        st.warning("Gmail credentials are not set. Cannot send email.")
        return False

    subject = "Urgent Anesthetic Call - Availability Request"
    body = f"""
    Dear Colleague,

    We have an urgent need for an anesthetist for an upcoming call with the following details:
    {call_details}

    Could you please let us know your availability as soon as possible?

    A simple 'Yes' or 'No' reply would be greatly appreciated.

    Thank you,
    The Anesthesia Department
    """
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = gmail_address
    msg['To'] = recipient_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(gmail_address, gmail_password)
            server.send_message(msg)
        st.info(f"Email sent to {recipient_email}.")
        return True
    except Exception as e:
        st.error(f"Failed to send email to {recipient_email}. Error: {e}")
        return False

# --- Recommendation Logic ---
def get_recommendations(case_details, anesthetists):
    """Uses an LLM to get recommendations."""
    anesthetist_list_str = "\n".join([
        f"- ID {a['id']}: {a['name']}, Specialty: {a['specialty']}, Experience: {a['experience_years']} years. Notes: {a['notes']}"
        for a in anesthetists
    ])

    template = """
    You are an expert assistant for a hospital's anesthesia department.
    Your task is to recommend the best substitute anesthetists for a specific medical case.

    Here are the details of the case:
    {case_details}

    Here is the list of available anesthetists:
    {anesthetist_list}

    Based on the case details, please identify the most suitable anesthetists from the list.
    Provide your response as a comma-separated list of their IDs ONLY as text. For example: 1,3,5
    """
    
    prompt = PromptTemplate(template=template, input_variables=["case_details", "anesthetist_list"])
    # New LangChain way
    chain = prompt | llm
   
    try:
        
        response = chain.invoke({"case_details": case_details, "anesthetist_list": anesthetist_list_str})
        
        if response:
            recommended_ids_str = str(response)
            recommended_ids = [int(id_str.strip()) for id_str in recommended_ids_str.split(',') if id_str.strip().isdigit()]
            return recommended_ids
        else:
            st.error("LLM response format is unexpected. Could not parse recommendations.")
            return []
            
    except Exception as e:
        st.error(f"An error occurred while getting recommendations from the LLM: {e}")
        return []

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Anesthetic Substitute Recommender 🤖")
st.write("This tool helps you find and contact the best substitute anesthetists for a new call.")

# Initialize database
db.init_db()
all_anesthetists = db.get_all_anesthetists()
anesthetist_map = {a['id']: a for a in all_anesthetists}

# --- Session State Initialization ---
if 'recommended_anesthetists' not in st.session_state:
    st.session_state.recommended_anesthetists = []
if 'final_selection' not in st.session_state:
    st.session_state.final_selection = []

# --- Main Application Flow ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Enter Call Details")
    with st.form("call_details_form"):
        case_specialty = st.selectbox("Required Specialty", ["General", "Pediatric", "Cardiac", "Neuro", "Obstetric", "Other"])
        case_description = st.text_area("Brief Case Description", "e.g., 5-hour cardiac bypass surgery for a high-risk adult patient.")
        submitted = st.form_submit_button("Get Recommendations")

    if submitted:
        with st.spinner("Asking the AI for recommendations..."):
            full_case_details = f"Specialty: {case_specialty}. Description: {case_description}"
            recommended_ids = get_recommendations(full_case_details, all_anesthetists)
            
            if recommended_ids:
                st.session_state.recommended_anesthetists = [anesthetist_map[id] for id in recommended_ids if id in anesthetist_map]
                st.session_state.final_selection = st.session_state.recommended_anesthetists[:] # Start with all recommended
                st.success("AI has provided recommendations!")
            else:
                st.error("Could not retrieve AI recommendations. Please check the logs or try again.")
                st.session_state.recommended_anesthetists = []
                st.session_state.final_selection = []

with col2:
    st.header("2. Review and Contact Substitutes")
    if not st.session_state.recommended_anesthetists:
        st.info("Recommendations will appear here after you submit the call details.")
    else:
        st.subheader("AI Recommended Substitutes:")
        
        notification_options = st.multiselect(
            'How do you want to notify them?',
            ['Email', 'SMS'],
            default=['Email']
        )
        
        edited_selection = []
        for anesthetist in st.session_state.recommended_anesthetists:
            is_selected = st.checkbox(
                f"**{anesthetist['name']}** ({anesthetist['specialty']}, {anesthetist['experience_years']} yrs)",
                value=anesthetist in st.session_state.final_selection,
                key=f"select_{anesthetist['id']}"
            )
            if is_selected:
                edited_selection.append(anesthetist)
        
        st.session_state.final_selection = edited_selection

        if st.button("🚀 Send Notifications to Selected Substitutes", type="primary"):
            if not st.session_state.final_selection:
                st.warning("Please select at least one anesthetist to notify.")
            elif not notification_options:
                st.warning("Please select a notification method (Email/SMS).")
            else:
                full_case_details = f"Specialty: {case_specialty}. Description: {case_description}"
                with st.spinner("Sending notifications..."):
                    for anesthetist in st.session_state.final_selection:
                        if 'Email' in notification_options:
                            send_email(anesthetist['email'], full_case_details)
                        if 'SMS' in notification_options:
                            if anesthetist.get('phone_number'):
                                send_sms(anesthetist['phone_number'], full_case_details)
                            else:
                                st.warning(f"SMS not sent to {anesthetist['name']}: No phone number on file.")
                st.success("Notification process complete. Check status messages above.")

st.divider()
st.header("📋 Full List of Available Anesthetists")
st.dataframe(all_anesthetists, use_container_width=True)
