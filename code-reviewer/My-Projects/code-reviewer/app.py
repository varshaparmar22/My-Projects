# app.py

import streamlit as st
from langchain_aws.chat_models import ChatBedrock
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ---------------------------
# import necessary libraries & variables
# ---------------------------
import os
from dotenv import load_dotenv

load_dotenv()
REGION = os.getenv("AWS_REGION", "us-east-1")
LLM_MODEL_ID = os.getenv("LLM_MODEL") 

st.set_page_config(page_title="AI Code Reviewer", layout="wide")
st.title("💻 AI Code Reviewer & Debugging Assistant")

# ---------------------------
# Claude (Bedrock) Setup
# ---------------------------

def get_bedrock_llm():
    return ChatBedrock(
        region_name=REGION,
        model_id=LLM_MODEL_ID
    )
# ---------------------------
# Create Prompt Template                
# ---------------------------

def get_prompt_template():
    return PromptTemplate.from_template(
        """
You are a highly skilled senior software engineer.

Your task is to **{task}** the following Python code.

Respond in markdown format with clear explanations, bullet points, or suggestions.

---

### Code:

```python
{code}
"""
)
# ---------------------------
# Call Claude for Code Review
# ---------------------------

def review_code_with_claude(code_snippet: str, task: str = "review") -> str:
    llm = get_bedrock_llm()
    prompt = get_prompt_template()
    chain = LLMChain(llm=llm, prompt=prompt)


    try:
        response = chain.run(code=code_snippet, task=task)
    except Exception as e:
        response = f"❌ Error while processing with Claude:\n{str(e)}"

    return response

# ---------------------------
# Streamlit UI
# ---------------------------

code = st.text_area("Paste your Python code below:", height=300)

task_option = st.selectbox(
"Choose a task:",
["review", "find bugs", "explain", "refactor"]
)

if st.button("Analyze Code"):
    if code.strip():
        with st.spinner("Analyzing your code with Claude..."):
            result = review_code_with_claude(code_snippet=code, task=task_option)
            st.subheader("💡 Claude's Feedback")
            st.markdown(result)
    else:
        st.warning("Please paste some Python code to analyze.")

