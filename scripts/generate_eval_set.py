"""
train.csvのパターンを参考に、評価用データ1002件を生成する。
数値系カテゴリ（physics, unit_conversion, numeral_system）は
プログラムで正確に答えを計算する。
text_encryption, bit_manipulation, symbol_transformationはDeepSeek APIで生成。

使い方:
  pip install openai
  python generate_eval_set.py --api-key YOUR_DEEPSEEK_API_KEY --output eval_set.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import string
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# 数値系カテゴリ: プログラムで生成（APIは使わない）
# ---------------------------------------------------------------------------

def generate_physics(rng: random.Random) -> dict:
    """d = 0.5 * g * t^2 の問題を生成。gをランダムに設定。"""
    g = round(rng.uniform(5.0, 20.0), 2)
    num_examples = rng.randint(3, 6)
    examples = []
    for _ in range(num_examples):
        t = round(rng.uniform(1.0, 5.0), 2)
        d = round(0.5 * g * t ** 2, 2)
        examples.append((t, d))

    t_q = round(rng.uniform(1.0, 5.0), 2)
    answer = round(0.5 * g * t_q ** 2, 2)

    lines = ["In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations:"]
    for t, d in examples:
        lines.append(f"For t = {t}s, distance = {d} m")
    lines.append(f"Now, determine the falling distance for t = {t_q}s given d = 0.5*g*t^2.")
    prompt = "\n".join(lines)
    return {"prompt": prompt, "answer": str(answer), "category": "physics"}


def generate_unit_conversion(rng: random.Random) -> dict:
    """secret unit conversion: output = input * factor"""
    factor = round(rng.uniform(0.3, 3.5), 4)
    num_examples = rng.randint(3, 6)
    examples = []
    for _ in range(num_examples):
        val = round(rng.uniform(5.0, 60.0), 2)
        converted = round(val * factor, 2)
        examples.append((val, converted))

    val_q = round(rng.uniform(5.0, 60.0), 2)
    answer = round(val_q * factor, 2)

    lines = ["In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:"]
    for val, conv in examples:
        lines.append(f"{val} m becomes {conv}")
    lines.append(f"Now, convert the following measurement: {val_q} m")
    prompt = "\n".join(lines)
    return {"prompt": prompt, "answer": str(answer), "category": "unit_conversion"}


def int_to_roman(num: int) -> str:
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    result = ""
    for i in range(len(val)):
        while num >= val[i]:
            result += syms[i]
            num -= val[i]
    return result


def generate_numeral_system(rng: random.Random) -> dict:
    """ローマ数字変換問題を生成。"""
    num_examples = rng.randint(4, 6)
    used = set()
    examples = []
    for _ in range(num_examples):
        n = rng.randint(1, 100)
        while n in used:
            n = rng.randint(1, 100)
        used.add(n)
        examples.append((n, int_to_roman(n)))

    n_q = rng.randint(1, 100)
    while n_q in used:
        n_q = rng.randint(1, 100)
    answer = int_to_roman(n_q)

    lines = ["In Alice's Wonderland, numbers are secretly converted into a different numeral system. Some examples are given below:"]
    for n, r in examples:
        lines.append(f"{n} -> {r}")
    lines.append(f"Now, write the number {n_q} in the Wonderland numeral system.")
    prompt = "\n".join(lines)
    return {"prompt": prompt, "answer": answer, "category": "numeral_system"}


# ---------------------------------------------------------------------------
# LLM系カテゴリ: DeepSeek APIで生成
# ---------------------------------------------------------------------------

BIT_MANIP_PROMPT = """Create one 8-bit binary transformation puzzle in EXACTLY this format (no extra text):

PROMPT: In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:
[5-8 examples, each on its own line: XXXXXXXX -> XXXXXXXX]

Now, determine the output for: XXXXXXXX
ANSWER: XXXXXXXX

Rules:
- Choose ONE consistent rule (e.g. right rotate by 1, XOR with 10101010, etc.)
- All examples AND the answer must follow the SAME rule
- Every input/output must be exactly 8 binary digits
- Do NOT explain the rule
- Do NOT include any text outside the PROMPT:/ANSWER: format
"""

TEXT_ENC_PROMPT = """Create one text decryption puzzle in EXACTLY this format (no extra text):

PROMPT: In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
[encrypted phrase] -> [original phrase]
[encrypted phrase] -> [original phrase]
[encrypted phrase] -> [original phrase]
Now, decrypt the following text: [encrypted phrase]
ANSWER: [decrypted phrase]

Rules:
- Use a consistent substitution or Caesar cipher
- Use simple 2-4 word phrases (Alice's Wonderland vocabulary: alice, rabbit, queen, king, wizard, dragon, mouse, hatter, turtle, cat, bird, princess, student, teacher, knight)
- All examples must follow the SAME cipher
- Do NOT explain the cipher
- Do NOT include any text outside the PROMPT:/ANSWER: format
"""

SYMBOL_PROMPT = """Create one symbol transformation puzzle in EXACTLY this format (no extra text):

PROMPT: In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
[expression] = [result]
[expression] = [result]
[expression] = [result]
[expression] = [result]
Now, determine the result for: [expression]
ANSWER: [result]

Rules:
- Use a consistent symbol substitution (e.g. replace specific chars with others)
- Use special characters: ! @ # $ % ^ & * [ ] { } \\ / | < > + - = ' ` ~ , .
- Expressions should be 4-8 characters long
- All examples must follow the SAME rule
- Do NOT explain the rule
- Do NOT include any text outside the PROMPT:/ANSWER: format
"""


def call_deepseek(client, system_prompt: str) -> str | None:
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": system_prompt}],
            max_tokens=800,
            temperature=0.9,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  API error: {e}")
        return None


def parse_prompt_answer(text: str) -> tuple[str, str] | None:
    if not text or "PROMPT:" not in text or "ANSWER:" not in text:
        return None
    try:
        prompt_part = text.split("PROMPT:")[1].split("ANSWER:")[0].strip()
        answer_part = text.split("ANSWER:")[1].strip().split("\n")[0].strip()
        if prompt_part and answer_part:
            return prompt_part, answer_part
    except Exception:
        pass
    return None


def generate_llm_category(client, category: str, count: int) -> list[dict]:
    prompt_map = {
        "bit_manipulation": BIT_MANIP_PROMPT,
        "text_encryption": TEXT_ENC_PROMPT,
        "symbol_transformation": SYMBOL_PROMPT,
    }
    template = prompt_map[category]
    rows = []
    attempts = 0
    max_attempts = count * 3

    while len(rows) < count and attempts < max_attempts:
        attempts += 1
        result = call_deepseek(client, template)
        parsed = parse_prompt_answer(result) if result else None
        if parsed:
            prompt_text, answer_text = parsed
            rows.append({"prompt": prompt_text, "answer": answer_text, "category": category})
            if len(rows) % 10 == 0:
                print(f"    {len(rows)}/{count} 件完了")

    return rows


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="DeepSeek API key")
    parser.add_argument("--output", default="eval_set.csv")
    parser.add_argument("--per-category", type=int, default=167)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=args.api_key, base_url="https://api.deepseek.com")

    rng = random.Random(args.seed)
    all_rows = []

    # 数値系カテゴリ（API不要）
    for cat, fn in [
        ("physics", generate_physics),
        ("unit_conversion", generate_unit_conversion),
        ("numeral_system", generate_numeral_system),
    ]:
        print(f"\n=== {cat} 生成中（プログラム計算）===")
        rows = [fn(rng) for _ in range(args.per_category)]
        all_rows.extend(rows)
        print(f"  {len(rows)}件完了")

    # LLM系カテゴリ（DeepSeek API）
    for cat in ["bit_manipulation", "text_encryption", "symbol_transformation"]:
        print(f"\n=== {cat} 生成中（DeepSeek API）===")
        rows = generate_llm_category(client, cat, args.per_category)
        all_rows.extend(rows)
        print(f"  {len(rows)}件完了")

    # ID付与してシャッフル
    rng.shuffle(all_rows)
    for i, row in enumerate(all_rows):
        row["id"] = f"eval_{i:05d}"

    # 保存
    out = Path(args.output)
    fieldnames = ["id", "prompt", "answer", "category"]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n保存完了: {len(all_rows)}件 → {out.resolve()}")

    # カテゴリ別集計
    from collections import Counter
    counts = Counter(r["category"] for r in all_rows)
    print("\nカテゴリ別件数:")
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}件")


if __name__ == "__main__":
    main()
