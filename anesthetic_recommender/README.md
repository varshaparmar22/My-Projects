# Anesthetic Substitute Recommender

This AI-powered application helps anesthetists find qualified and available substitutes for medical calls. It provides recommendations based on specialty and experience, and automates the initial outreach via email.

## Features

- **Intelligent Recommendations**: Uses a Large Language Model (LLM) to suggest the best-fit substitutes.
- **Dynamic List Management**: Allows the primary anesthetist to review and curate the list of recommended substitutes.
- **Automated Email Outreach**: Sends availability requests to selected candidates with a single click.
- **Status Tracking**: Provides a simple interface to track the confirmation status of substitutes.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd anesthetic_recommender
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**

    Create a `.env` file in the root of the project directory and add the following:

    ```
    # Get a free API token from https://huggingface.co/settings/tokens
    HUGGINGFACEHUB_API_TOKEN="your-huggingface-api-token"
    # for paid account of huggingface - Check for Hugging Face API Token
    # HUGGINGFACE is not allowing access to any LLM for free account
    # so if you have access to LLM then only consider it
    # check_model.py run with your api key and check if you have acccess to it.
    # Your Gmail credentials for sending emails
    # Note: You may need to create an "App Password" for this to work
    # https://support.google.com/accounts/answer/185833
    GMAIL_ADDRESS="your-email@gmail.com"
    GMAIL_APP_PASSWORD="your-gmail-app-password"\n\n    # Your Twilio credentials for sending SMS (https://www.twilio.com/try-twilio)\n    TWILIO_ACCOUNT_SID="your-twilio-account-sid"\n    TWILIO_AUTH_TOKEN="your-twilio-auth-token"\n    TWILIO_PHONE_NUMBER="your-twilio-phone-number"
    ```

## How to Run

Once the setup is complete, run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your web browser.
