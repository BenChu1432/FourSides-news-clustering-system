import json
import re
from together import Together
from dotenv import load_dotenv
import os

load_dotenv()
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY")
client = Together(api_key=TOGETHER_AI_API_KEY)

def clean_and_fix_json(raw: str) -> dict:
    s = raw.strip()

    # 移除 markdown code fence
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s)
        s = re.sub(r"```$", "", s).strip()

    # 嘗試強制補上缺失的大括號
    if not s.endswith("}"):
        s += "}"

    # 移除多餘逗號
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)

    # 修正 headline / summary / question 若沒用雙引號包起來
    def fix_quotes(field: str, text: str) -> str:
        # 修復未用雙引號包裹的欄位值
        pattern = rf'"{field}":\s*([^\n",\[\]{{}}]+)'
        return re.sub(pattern, lambda m: f'"{field}": "{m.group(1).strip()}"', text)

    for field in ["headline", "summary", "question"]:
        s = fix_quotes(field, s)

    # 嘗試解析
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        print("❌ 最終仍無法解析 JSON：", e)
        print("🔎 修正後原始內容：", s)
        raise e

system_prompt = """
你是一位專業的新聞分析助理，負責閱讀新聞內容並撰寫「標題」、「摘要」與「引發討論的問題」。請嚴格依照以下規則輸出：

【語言與風格】
- 使用繁體中文
- 採用中立、客觀的語氣

【格式要求】
- 僅允許輸出 JSON 格式，包含以下欄位：
  - "headline"：一句話總結新聞主題（字串）
  - "summary"：用小一段歸納新聞重點（字串）
  - "question"：設計一個具爭議性、可明確表態支持/反對的問題（字串）

📌【問題設計規則】
- 問題必須能讓讀者以「支持 / 中立 / 反對」作答
- 禁止開放式問題（如「您認為政府應該怎麼做？」）
- 禁止模糊問題（如「您如何看待此議題？」）
- 問題應設計為：是否應該...？是否支持...？是否贊成...？
- 問題應針對一項具體政策、行為、立場或決策
- 問題應可反映出社會上不同意見

✅【範例格式】（請嚴格照以下格式輸出）：

{
  "headline": "民眾對政府防毒政策意見分歧",
  "summary": "在一項社群投票中，針對政府應如何更有效防止毒品流入市面以保護民眾健康與安全，45% 表示支持、20% 表示中立、35% 表示反對，顯示社會對該議題尚存爭議。",
  "question": "您是否支持政府加強防毒措施以維護公共健康與安全？"
}

⚠️ 僅允許輸出合法 JSON 格式，不可有多餘文字或說明
⚠️ 不得省略任一欄位，即使資訊有限，也請合理補齊
""".strip()



def get_headline_summary_and_question(article_text: str) -> dict:
    prompt = f"""請分析以下新聞文章，並依 system prompt 的格式與規則輸出結構化 JSON 有關新聞的標題與摘要:

--- ARTICLE START ---
{article_text}
--- ARTICLE END ---
"""
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": prompt.strip()},
        ],
        temperature=0.6,
    )
    raw = response.choices[0].message.content
    print("raw:",raw)
    # Attempt to parse JSON; fall back to best-effort cleanup
    try:
        return json.loads(raw)
    except Exception:
        # Optional: strip code fences or extra text
        return clean_and_fix_json(raw)