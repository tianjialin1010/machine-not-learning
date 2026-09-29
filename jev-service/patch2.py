import sys

PATH = "/home/USER/jev-service/src/jev_service/provider.py"
with open(PATH, encoding="utf-8") as handle:
    text = handle.read()

NEW = '''        system = (
            "You are a typed decision classifier inside a software system. Do not write a reply. "
            "Return JSON only (no markdown fences) with a top-level answers object, one entry per "
            "question id, using EXACTLY these shapes and key names:\\n"
            "- choice: {\\"type\\":\\"choice\\",\\"choice\\":\\"<one declared option>\\",\\"probabilities\\":{...sums to 1...}}\\n"
            "- score:  {\\"type\\":\\"score\\",\\"score\\":<number>,\\"probabilities\\":{...sums to 1...}}\\n"
            "- noul:   {\\"type\\":\\"noul\\",\\"noul\\":<0.0-1.0>}\\n"
            "Rules: never emit a value key; every probabilities object must sum to 1; "
            "the score must be within 0..(number of levels - 1); answer EVERY question id listed below.\\n"
            f"Question schema:\\n{schema}"
        )
'''

start = text.find("        system = (")
end = text.find("        payload = {", start)
if start < 0 or end < 0:
    print("未定位到提示词区块")
    sys.exit(1)

text = text[:start] + NEW + "\n" + text[end:]
with open(PATH, "w", encoding="utf-8") as handle:
    handle.write(text)

print("提示词已替换：改用 choice/score/noul 原生键名")
