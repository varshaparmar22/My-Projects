import requests
import json
from dotenv import load_dotenv
import os
import gzip

# Load environment variables
load_dotenv()


# Airtable API details
BASE_ID = os.getenv("BASE_ID")  # Found in Airtable API docs for your base
AIRTABLE_TOKEN = os.getenv("AIRTABLE_API_TOKEN")  # Personal Access Token

# API URLs for each table
TABLES = {
    "Applicants": f"https://api.airtable.com/v0/{BASE_ID}/Applicants",
    "Personal Details": f"https://api.airtable.com/v0/{BASE_ID}/Personal%20Details",
    "Work Experience": f"https://api.airtable.com/v0/{BASE_ID}/Work%20Experience",
    "Salary Preferences": f"https://api.airtable.com/v0/{BASE_ID}/Salary%20Preferences"
}

HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_TOKEN}",
    "Content-Type": "application/json"
}

# -------------------------
# Fetch all records
# -------------------------
def fetch_records(url):
    records = []
    offset = None
    while True:
        params = {}
        if offset:
            params["offset"] = offset
        r = requests.get(url, headers=HEADERS, params=params)
        r.raise_for_status()
        data = r.json()
        records.extend(data.get("records", []))
        offset = data.get("offset")
        if not offset:
            break
    return records

# -------------------------
# Upsert record by matching linked Applicant ID
# -------------------------
def upsert_by_applicant_link(table_url, applicant_record_id, fields):
    records = fetch_records(table_url)
    existing = None
    for rec in records:
        if applicant_record_id in rec.get("fields", {}).get("Applicants", []):
            existing = rec
            break

    if existing:
        rec_id = existing["id"]
        resp = requests.patch(f"{table_url}/{rec_id}", headers=HEADERS, json={"fields": fields})
        resp.raise_for_status()
        print(f"Updated record in {table_url}")
    else:
        resp = requests.post(table_url, headers=HEADERS, json={"fields": fields})
        resp.raise_for_status()
        print(f"Inserted record in {table_url}")

# -------------------------
# Main restore process
# -------------------------
def restore_from_applicants():
    applicants = fetch_records(TABLES["Applicants"])
    for applicant in applicants:
        applicant_id = applicant["fields"].get("Applicant ID")
        record_id = applicant["id"]

        compressed_json_str = applicant["fields"].get("Compressed JSON")
        if not compressed_json_str:
            print(f"No compressed JSON found for applicant {applicant_id}")
            continue

        # Parse stored JSON string
        try:
            merged_data = json.loads(compressed_json_str)
        except Exception as e:
            print(f"Error parsing JSON for Applicant ID {applicant_id}: {e}")
            continue

        print(f"Restoring data for Applicant ID: {applicant_id}")

        # -------------------------
        # 1. Personal Details
        # -------------------------
        personal_fields = {
            "Full Name": merged_data["personal"].get("name", ""),
            "Location": merged_data["personal"].get("location", ""),
            "Applicants": [record_id]  # Link back to applicant
        }
        upsert_by_applicant_link(TABLES["Personal Details"], record_id, personal_fields)

        # -------------------------
        # 2. Work Experience (delete old, insert new)
        # -------------------------
        existing_exp = fetch_records(TABLES["Work Experience"])
        for exp in existing_exp:
            if record_id in exp.get("fields", {}).get("Applicants", []):
                requests.patch(f"{TABLES['Work Experience']}/{exp['id']}", headers=HEADERS)
            # else:        
            #     exp_fields = {
            #         "Company": exp.get("company", ""),
            #         "Title": exp.get("title", ""),
            #         "Applicants": [record_id]
            #     }
            #     requests.post(TABLES["Work Experience"], headers=HEADERS, json={"fields": exp_fields})

        # -------------------------
        # 3. Salary Preferences
        # -------------------------
        salary_fields = {
            "Preferred Rate": merged_data["salary"].get("rate", ""),
            "Currency": merged_data["salary"].get("currency", ""),
            "Availability (hrs/wk)": merged_data["salary"].get("availability", ""),
            "Applicants": [record_id]
        }
        upsert_by_applicant_link(TABLES["Salary Preferences"], record_id, salary_fields)

        print(f"✅ Finished restoring Applicant ID: {applicant_id}")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    restore_from_applicants()
