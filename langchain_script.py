from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import time
from secret_key import GOOGLE_API_KEY

import os
print(os.getenv("GEMINI_API_KEY"))
# Initialize Gemini via LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
)

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["user_prompt"],
    template="""
You are a script generator. Create a detailed script based on:
{user_prompt}
"""
)

# Chain setup
chain = prompt_template | llm

def generate_script(user_prompt):
    for attempt in range(4):
        try:
            result = chain.invoke({"user_prompt":  user_prompt})
            return result.content if hasattr(result, "content") else result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(3)

    return "❌ Failed to generate script after multiple attempts."
