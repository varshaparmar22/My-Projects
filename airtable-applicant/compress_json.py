import os
import requests
import json
import gzip
from dotenv import load_dotenv

load_dotenv()


# Airtable API details
BASE_ID = os.getenv("BASE_ID")  # Found in Airtable API docs for your base
AIRTABLE_TOKEN = os.getenv("AIRTABLE_API_TOKEN")  # Personal Access Token


# API URLs for each table
TABLES = {
    "Applicants": f"https://api.airtable.com/v0/{BASE_ID}/Applicants",
    "Personal Details": f"https://api.airtable.com/v0/{BASE_ID}/Personal Details",
    "Work Experience": f"https://api.airtable.com/v0/{BASE_ID}/Work Experience",
    "Salary Preferences": f"https://api.airtable.com/v0/{BASE_ID}/Salary Preferences",
    "Shortlisted Leads": f"https://api.airtable.com/v0/{BASE_ID}/Shortlisted Leads"
}

# Headers for requests
HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_TOKEN}",
    "Content-Type": "application/json"
}

# -------------------------
# Helper: Fetch all records from Airtable
# -------------------------
def fetch_records(url):
    records = []
    offset = None
    while True:
        params = {}
        if offset:
            params["offset"] = offset
        resp = requests.get(url, headers=HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
        records.extend(data.get("records", []))
        offset = data.get("offset")
        if not offset:
            break
    return records

def match_by_applicant_id(record, applicant_id):
    ids = record.get("fields", {}).get("Applicant ID (from Applicants)", [])
    
    if not ids:
        return False
    # Convert both to string for comparison
    return str(ids[0]) == str(applicant_id)

# -------------------------
# Build JSON for a specific applicant_id
# -------------------------
def build_applicant_json(applicant_id):
    personal_details = fetch_records(TABLES["Personal Details"])
    work_experience = fetch_records(TABLES["Work Experience"])
    salary_prefs = fetch_records(TABLES["Salary Preferences"])
   
    # Filter by applicant_id
    personal = next((p for p in personal_details if match_by_applicant_id(p, applicant_id)), {})
    experiences = [w for w in work_experience if match_by_applicant_id(w, applicant_id)]
    salary = next((s for s in salary_prefs if match_by_applicant_id(s, applicant_id)), {})

    # Build structured JSON
    data = {
        "personal": {
            "name": personal.get("fields", {}).get("Full Name", ""),
            "location": personal.get("fields", {}).get("Location", "")
        },
        "experience": [
            {
                "company": e.get("fields", {}).get("Company", ""),
                "title": e.get("fields", {}).get("Title", "")
            }
            for e in experiences
        ],
        "salary": {
            "rate": salary.get("fields", {}).get("Preferred Rate", 0),
            "currency": salary.get("fields", {}).get("Currency", ""),
            "availability": salary.get("fields", {}).get("Availability (hrs/wk)", 0)
        }
    }
    return data


# -------------------------
# Save compressed JSON locally
# -------------------------
def save_compressed_json(data, filename):
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    with open(filename, "wt", encoding="utf-8") as f:
        f.write(json_str)
    print(f"Compressed JSON saved to {filename}")


# -------------------------
# Update record in Applicants table
# -------------------------
def update_applicant_record(applicant_record_id, json_data):
    # Store JSON as string in Airtable text field "Compressed JSON"
    update_payload = {
        "fields": {
            "Compressed JSON": json.dumps(json_data, ensure_ascii=False)
        }
    }
    url = f"{TABLES['Applicants']}/{applicant_record_id}"
    resp = requests.patch(url, headers=HEADERS, json=update_payload)
    resp.raise_for_status()
    print(f"Updated Applicants record {applicant_record_id}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Fetch all applicants
    applicants = fetch_records(TABLES["Applicants"])
    print(f"Found {len(applicants)} applicants")
    if not applicants:
        print("No applicants found.")
        exit(0) 
    for applicant in applicants:
        applicant_id = applicant["fields"].get("Applicant ID")
        record_id = applicant["id"]

        if not applicant_id:
            continue

        print(f"Processing Applicant ID: {applicant_id}")

        # Build JSON
        merged_json = build_applicant_json(applicant_id)

        # Save compressed JSON locally
        filename = f"applicant_{applicant_id}.json"
        save_compressed_json(merged_json, filename)

        # Update Airtable Applicants table with JSON
        update_applicant_record(record_id, merged_json)
