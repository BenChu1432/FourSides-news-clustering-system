import json
from together import Together
from dotenv import load_dotenv
import os

load_dotenv()
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY")
client = Together(api_key=TOGETHER_AI_API_KEY)

system_prompt = """
你是一位專業的新聞分析助理，負責閱讀新聞內容並撰寫「標題」與「摘要」。請嚴格依照以下規則輸出：

【語言與風格】
- 使用繁體中文
- 採用中立、客觀的語語

【格式要求】
- 請務必輸出以下兩個欄位，缺一不可：
  - "headline"：一句話總結新聞主題（字串）
  - "summary"：用小一段歸納總結新聞報道重點
- 僅允許輸出合法 JSON 格式，不可有多餘文字或說明
- 不得省略任一欄位，即使資訊有限，也請合理補齊

{
  "headline": "一句話總結新聞主題（字串）",
  "summary": "用小一段總結內文"
}
""".strip()



def get_headline_and_summary(article_text: str) -> dict:
    prompt = f"""請分析以下新聞文章，並依 system prompt 的格式與規則輸出結構化 JSON 有關新聞的標題與摘要:

--- ARTICLE START ---
{article_text}
--- ARTICLE END ---
"""
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": prompt.strip()},
        ],
        temperature=0.6,
    )
    raw = response.choices[0].message.content
    # Attempt to parse JSON; fall back to best-effort cleanup
    try:
        return json.loads(raw)
    except Exception:
        # Optional: strip code fences or extra text
        s = raw.strip()
        if s.startswith("```"):
            s = s.strip("`").strip()
            if s.startswith("json"):
                s = s[4:].strip()
        return json.loads(s)