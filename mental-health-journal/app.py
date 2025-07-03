# app.py
import streamlit as st
from langchain_ollama import OllamaLLM
from journal_prompt import dual_response_template
import streamlit as st
import pandas as pd
from datetime import datetime
import os


# File path
CSV_PATH = "journal_entries.csv"

# Create CSV if it doesn't exist
if not os.path.exists(CSV_PATH):
    pd.DataFrame(columns=["timestamp", "entry", "reflection", "summary", "sentiment"]).to_csv(CSV_PATH, index=False)


# UI
st.set_page_config(page_title="Mental Health Journal Assistant", page_icon="🧠", layout="wide")
st.title("🧠 Mental Health Journal Assistant")

journal_entry = st.text_area("Write your journal entry:")

if st.button("💡 Reflect & Save"):
    if journal_entry.strip():
        with st.spinner("Analyzing and reflecting..."):
            llm = OllamaLLM(model="gemma3")

            # Generate reflection + summary
            full_prompt = dual_response_template.format(entry=journal_entry)
            response = llm.invoke(full_prompt)

            # Parse response
            if "=== Suggestion ===" in response and "=== Summary ===" in response and "=== Sentiment ===" in response:
                parts = response.split("=== Summary ===")
                suggestion = parts[0].replace("=== Suggestion ===", "").strip()
                summary_sentiment = parts[1].split("=== Sentiment ===")
                summary = summary_sentiment[0].strip()
                sentiment = summary_sentiment[1].strip() if len(summary_sentiment) > 1 else "Sentiment not found."
            elif "=== Suggestion ===" in response and "=== Summary ===" not in response and "=== Sentiment ===" in response:    
                suggestion = response.split("=== Suggestion ===")[1].split("=== Sentiment ===")[0].strip()
                summary = "Summary not found."
                sentiment = response.split("=== Sentiment ===")[1].strip()  
            else:
                suggestion = response.strip()
                summary = "Summary not found."

            
            # Store to CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_data = pd.DataFrame([[timestamp, journal_entry, suggestion, summary, sentiment]],
                                    columns=["timestamp", "entry", "reflection", "summary", "sentiment"])
            new_data.to_csv(CSV_PATH, mode='a', header=False, index=False)

            st.success("✅ Journal entry saved with reflection and sentiment.")
            st.subheader("🧘 AI Reflection")
            st.write(suggestion)
            st.subheader("📝 Summary")
            st.success(summary)
            st.subheader("❤️ Sentiment")
            st.info(sentiment)
    else:
        st.warning("Please enter your journal text.")

# View Past Entries
if st.button("📅 View Past Summaries"):
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        st.info("No entries yet.")
    else:
        st.subheader("🗂 Past Summaries")
        for _, row in df.iterrows():
            st.markdown(f"**🕒 {row['timestamp']}**")
            st.markdown(f"- **Summary:** {row['summary']}")
            st.markdown(f"- **Sentiment:** {row['sentiment']}")
            st.markdown("---")
