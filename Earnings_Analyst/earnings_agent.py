# earnings_agent.py

import os
from rag_module import initialize_rag, retrieve_filtered


import google.generativeai as genai

genai.configure(api_key="AIzaSyBn-86fV6HvaFKF_KIfFL2IgbwsZgjKb6k")
model = genai.GenerativeModel("gemini-2.5-flash")


GPT_PROMPT = """
You are a CFA-level senior equity research analyst with 20+ years of experience in financial statement analysis and institutional investing.

Your task is to perform a structured earnings analysis using ONLY the provided context.

---------------------
CRITICAL RULES
---------------------
- Use ONLY the provided context
- Do NOT use external knowledge
- Do NOT hallucinate numbers
- If unsure, avoid specific numbers
- Be analytical, not descriptive
- Keep reasoning realistic (no overconfidence)

---------------------
STRUCTURED ANALYSIS FRAMEWORK
---------------------
Internally evaluate the company on these 4 dimensions:

1. Revenue Performance
   - Growth strength
   - Stability

2. Profitability
   - Margins
   - Earnings growth

3. Earnings Quality
   - Sustainability
   - One-time vs recurring

4. Management Guidance
   - Future outlook
   - Risks or confidence signals

For EACH dimension, internally assign a score (0–10).

Then combine them into ONE final score using balanced judgment:
- Strong across all → 8–9
- Mixed signals → 5–7
- Weak fundamentals → 2–4

IMPORTANT:
Do NOT output these internal scores. Use them only to decide the final score.
Only use numbers explicitly present in the context. Do NOT introduce new figures.
If performance is strong across most dimensions, prefer scores in the 8–9 range.
Each key point must be under 30 words.

---------------------
OUTPUT REQUIREMENTS
---------------------
Return ONLY valid JSON in this exact format:

{
  "agent": "Earnings Analyst",
  "company": "{company}",
  "key_points": [
    "Insight 1 (max 25 words)",
    "Insight 2 (max 25 words)",
    "Insight 3 (max 25 words)",
    "Insight 4 (max 25 words)"
  ],
  "score": <Number between 0-10 depending on the score>,
  "reasoning": "Concise explanation clearly justifying the score using the four dimensions"
}

STRICTLY:
- No markdown
- No explanation outside JSON
- No extra text

---------------------
INPUT
---------------------
Company: {company}

Context:
{context}
"""

def get_context(company):
    query = "revenue growth profitability margins earnings guidance financial performance"

    results = retrieve_filtered(
        query=query,
        company=company,
        data_type="earnings_q1"
    )

    #print("\n[DEBUG] Retrieved chunks:", len(results))   # DEBUG

    if not results:
        print("[DEBUG] No results found!")              # Debug
        return "No financial data available"

    context = "\n".join(results[:3])

    #print("\n[DEBUG] Context preview:\n", context[:300])  #Debug

    return context


import json

def safe_parse(output):
    try:
        return json.loads(output)
    except:
        try:
            start = output.find("{")
            end = output.rfind("}") + 1
            return json.loads(output[start:end])
        except:
            return {"error": "Invalid JSON", "raw": output}

def run_gemini(company, context):
    prompt = GPT_PROMPT.replace("{company}", company).replace("{context}", context)

    #print("\n[DEBUG] Prompt preview:\n", prompt[:300])   # Debug

    response = model.generate_content(prompt)

    #print("\n[DEBUG] Raw model output:\n", response.text)  #debug

    if not response or not hasattr(response, "text"):
        return '{"error": "No response"}'

    return response.text

def earnings_agent(company):
    context = get_context(company)
    output = run_gemini(company, context)

    parsed = safe_parse(output)

    #print("\n[DEBUG] Parsed output:\n", parsed)  #Debug

    return parsed
    

def test():
    print("It works")

def save_output(result, company):
    filename = f"outputs/{company.lower()}_earnings.json"
    os.makedirs("outputs", exist_ok=True)
    with open(filename, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[INFO] Output saved to {filename}")    
    

if __name__ == "__main__":
    initialize_rag()
    result = earnings_agent("Apple")
    print(result)
    save_output(result, "Apple")
    
    
    
# IF YOU GUYS NEED BOTH OUTPUTS OF BOTH THE APPLE AND TESLA, USE 
#THIS BELOW CODE
"""
if __name__ == "__main__":
    initialize_rag()
    for company in ["Apple", "Tesla"]:
        result = earnings_agent(company)
        print(result)
        save_output(result, company)
"""