import os
from dotenv import load_dotenv
from together import Together
import json
from openai import OpenAI
import together
load_dotenv()
TOGETHER_AI_API_KEY=os.getenv("TOGETHER_AI_API_KEY")
client = Together(api_key=TOGETHER_AI_API_KEY)

# LLAMA:meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
# Alibaba: Qwen/Qwen2.5-7B-Instruct-Turbo

def safe_extract_json(raw_output: str) -> dict:
    # 嘗試找出最外層的 JSON 區塊
    match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    if not match:
        print("⚠️ 無法擷取 JSON 區塊")
        return None

    json_str = match.group(0)

    try:
        result = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        print("⚠️ JSON Decode Error:", e)
        print("原始 JSON 字串:")
        print(json_str)
        return None


system_prompt = f"""
你是一位專業的新聞分析助理，負責閱讀新聞內容並撰寫「標題」與「摘要」。請嚴格依照以下規則輸出：

【語言與風格】
- 使用**繁體中文**
- 採用**中立、客觀**的語氣
- 摘要使用**條列式（使用「‧」符號）**

【格式要求】
- 請務必輸出以下兩個欄位，**缺一不可**：
  - `"headline"`：一句話總結新聞主題
  - `"summary"`：條列1～3點新聞要點
- 僅允許輸出**合法 JSON 格式**，不可有多餘文字或說明
- 不得省略任一欄位，即使資訊有限，也請合理補齊

【JSON 結構範例】
{{
  "headline": "高雄前金區公共開發案啟動，打造亞洲資產中心",
  "summary": "‧ 高雄市政府啟動前金區公共開發案，對外招商\n‧ 計畫建造30層商業大樓，引進企業資源\n‧ 預期打造亞洲資產中心，對標NASDAQ"
}}
"""



import re

def write_headline_and_summary(article_text: str) -> dict:
    prompt = f"""請閱讀以下新聞文章，並依照 system prompt 規則，輸出包含 `headline` 與 `summary` 的 JSON 結構，請確保兩個欄位都存在：

--- ARTICLE START ---
{article_text}
--- ARTICLE END ---
"""

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": prompt.strip()}
        ],
        temperature=0.6
    )

    raw_output = response.choices[0].message.content.strip()
    print("Raw output:", raw_output)

    return safe_extract_json(raw_output)