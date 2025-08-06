import requests
import json
from datetime import datetime, timezone

# Airtable API details
BASE_ID = "appsAu9EYADvRBlNR"
AIRTABLE_TOKEN = "patldFrkz5SiZ5jCa.57590a65449d57b677f6b1d42899d77c4925dc54bf5b622a8f0c834c5d376ae5"  # Personal Access Token

TABLES = {
    "Applicants": f"https://api.airtable.com/v0/{BASE_ID}/Applicants",
    "Shortlisted Leads": f"https://api.airtable.com/v0/{BASE_ID}/Shortlisted%20Leads"
}

HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_TOKEN}",
    "Content-Type": "application/json"
}

# Tier‑1 companies list
TIER1_COMPANIES = {"Google", "Meta", "OpenAI", "Microsoft", "Apple", "Amazon"}

# Allowed locations
ALLOWED_LOCATIONS = {"US", "USA", "United States", "Canada", "UK", "United Kingdom", "Germany", "India"}

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
# Parse date string to datetime safely
# -------------------------
def parse_date(date_str):
    if not date_str:
        return None
    for fmt in ("%d-%m-%Y", "%m-%d-%Y", "%d-%m-%y", "%m-%d-%y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

# -------------------------
# Calculate total years of experience
# -------------------------
def calculate_experience_years(experiences):
    total_days = 0
    worked_tier1 = False

    for exp in experiences:
        company = exp.get("company", "").strip()
        if company in TIER1_COMPANIES:
            worked_tier1 = True

        start_str = exp.get("start") or exp.get("Start")  # Handle JSON key variations
        end_str = exp.get("end") or exp.get("End")

        start_date = parse_date(start_str)
        end_date = parse_date(end_str) or datetime.now(timezone.utc).isoformat()

        if start_date and end_date and end_date > start_date:
            total_days += (end_date - start_date).days

    total_years = total_days / 365.25
    return total_years, worked_tier1

# -------------------------
# Evaluate if applicant meets the criteria
# -------------------------
def evaluate_applicant(merged_data):
    reasons = []

    # 1. Experience
    total_years, worked_tier1 = calculate_experience_years(merged_data.get("experience", []))
    if total_years >= 4 or worked_tier1:
        reasons.append(f"Experience OK ({total_years:.1f} yrs)")
    else:
        return False, f"Insufficient experience ({total_years:.1f} yrs)"

    # 2. Compensation
    try:
        rate = float(merged_data.get("salary", {}).get("rate", 0))
    except:
        rate = 0
    try:
        availability = float(merged_data.get("salary", {}).get("availability", 0))
    except:
        availability = 0

    if rate <= 100 and availability >= 20:
        reasons.append(f"Compensation OK (${rate}/hr, {availability} hrs/wk)")
    else:
        return False, f"Compensation criteria not met (${rate}/hr, {availability} hrs/wk)"

    # 3. Location
    location = merged_data.get("personal", {}).get("location", "").strip()
    if location in ALLOWED_LOCATIONS:
        reasons.append(f"Location OK ({location})")
    else:
        return False, f"Location '{location}' not in allowed list"

    return True, ", ".join(reasons)

# -------------------------
# Main Shortlist process
# -------------------------
def shortlist_leads():
    applicants = fetch_records(TABLES["Applicants"])
    for applicant in applicants:
        applicant_id = applicant["fields"].get("Applicant ID")
        record_id = applicant["id"]
        compressed_json_str = applicant["fields"].get("Compressed JSON")

        if not compressed_json_str:
            continue

        try:
            merged_data = json.loads(compressed_json_str)
        except Exception as e:
            print(f"Error parsing JSON for Applicant ID {applicant_id}: {e}")
            continue

        qualifies, reason = evaluate_applicant(merged_data)
        if not qualifies:
            print(f"❌ Applicant {applicant_id} not shortlisted: {reason}")
            continue

        # Prepare fields for Shortlisted Leads table
        lead_fields = {
            "LeadId": f"LEAD-{applicant_id}",
            "Applicants": [record_id],
            "Compressed JSON": json.dumps(merged_data, ensure_ascii=False),
            "Score Reason": reason,
            "Created At": datetime.now(timezone.utc).isoformat()
        }

        # Insert into Shortlisted Leads
        resp = requests.post(TABLES["Shortlisted Leads"], headers=HEADERS, json={"fields": lead_fields})
        if resp.status_code == 200:
            print(f"✅ Shortlisted Applicant {applicant_id}: {reason}")
        else:
            print(f"⚠️ Failed to insert Shortlisted Lead for Applicant {applicant_id}: {resp.text}")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    shortlist_leads()
