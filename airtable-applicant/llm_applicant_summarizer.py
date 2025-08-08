import os
import json
import time
import logging
import requests
from dotenv import load_dotenv
import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Load .env
load_dotenv()

# Airtable setup
BASE_ID = os.getenv("BASE_ID")
AIRTABLE_TOKEN = os.getenv("AIRTABLE_API_TOKEN")
TABLE_NAME = "Applicants"
AIRTABLE_URL = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"
HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_TOKEN}",
    "Content-Type": "application/json"
}

# AWS Bedrock setup
bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))

# LLM prompt
PROMPT_TEMPLATE = """You are reviewing the following job applicant data:

{applicant_data}

Please review the data and respond in the following format:

Summary (≤75 words):
[Your summary here]

Quality Score (1–10):
[Your score here]

Missing or Contradictory Fields:
[List here, or write 'None']

Follow-up Questions:
1. [Question 1]
2. [Question 2]
3. [Question 3]
"""

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_applicant_by_id(applicant_id):
    params = {"filterByFormula": f"{{Applicant ID}} = '{applicant_id}'"}
    response = requests.get(AIRTABLE_URL, headers=HEADERS, params=params)
    response.raise_for_status()
    records = response.json().get("records", [])
    return records[0] if records else None

def call_bedrock_llm(payload, retries=3):
    #for attempt in range(1, retries + 1):
    for attempt in range(1):
        try:
            response = bedrock.invoke_model(
                modelId="amazon.titan-text-lite-v1",
                body=json.dumps({
                    "inputText": payload,
                    "textGenerationConfig": {
                        "maxTokenCount": 500,
                        "temperature": 0.5,
                        "topP": 1.0
                    }
                }),
                contentType="application/json",
                accept="application/json"
            )
            result = json.loads(response['body'].read())
            print(f"LLM: {payload}")
            print(f"LLM Response: {result}")
            return result['results'][0]['outputText']
        except (BotoCoreError, ClientError, KeyError, json.JSONDecodeError) as e:
            logger.error(f"[Attempt {attempt}] LLM call failed: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                raise

def update_applicant_fields(record_id, summary, score, follow_ups):
    fields = {
        "LLM Summary": summary.strip(),
        "LLM Score": score.strip(),
        "LLM Follow-Ups": follow_ups.strip()
    }
    resp = requests.patch(
        f"{AIRTABLE_URL}/{record_id}",
        headers=HEADERS,
        json={"fields": fields}
    )
    if resp.status_code == 200:
        logger.info(f"✅ Updated applicant {record_id} with LLM results.")
    else:
        logger.warning(f"⚠️ Failed to update applicant {record_id}: {resp.text}")

def extract_llm_fields(response_text):
    summary, score, follow_ups = "", "", ""
    lines = response_text.strip().splitlines()

    capture = None
    follow_up_list = []

    for line in lines:
        l = line.strip()
        if l.lower().startswith("summary"):
            capture = "summary"
            continue
        elif l.lower().startswith("quality score"):
            capture = "score"
            continue
        elif l.lower().startswith("missing") or "contradictory" in l.lower():
            capture = "missing"
            continue
        elif l.lower().startswith("follow-up questions"):
            capture = "follow"
            continue

        if capture == "summary":
            summary += l + " "
        elif capture == "score":
            score = ''.join(filter(str.isdigit, l)) or "N/A"
        elif capture == "follow":
            if l and any(char.isalpha() for char in l):
                follow_up_list.append(l.strip("1234567890.:- ").strip())

    return summary.strip(), score.strip(), "; ".join(follow_up_list).strip()


def process_applicant(applicant_id):
    record = fetch_applicant_by_id(applicant_id)
    if not record:
        logger.warning(f"Applicant ID {applicant_id} not found.")
        return

    fields = record.get("fields", {})
    if fields.get("LLM Summary") or fields.get("LLM Score") or fields.get("LLM Follow-Ups"):
        logger.info(f"🔁 Applicant {applicant_id} already has LLM results. Skipping.")
        return

    compressed_json = fields.get("Compressed JSON")
    if not compressed_json:
        logger.warning(f"Applicant {applicant_id} missing Compressed JSON.")
        return

    try:
        applicant_data = json.dumps(json.loads(compressed_json), indent=2)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON for applicant {applicant_id}: {e}")
        return

    prompt = PROMPT_TEMPLATE.format(applicant_data=applicant_data)

    try:
        llm_response = call_bedrock_llm(prompt)
        #llm_response = "This is a mock response for testing purposes."  # Replace with actual LLM call
        summary, score, follow_ups = extract_llm_fields(llm_response)
        update_applicant_fields(record["id"], summary, score, follow_ups)
    except Exception as e:
        logger.error(f"❌ Final failure for applicant {applicant_id}: {e}")

if __name__ == "__main__":
    # Replace with dynamic input or loop if needed
    test_applicant_id = "5"  # Replace with actual ID from Airtable
    process_applicant(test_applicant_id)
