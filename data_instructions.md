# データ担当者向け指示書

## このコンペでやること

NVIDIAのAIモデル（Nemotron-30B）に「Alice's Wonderland」という推論パズルを解かせるためのデータを作る。

モデルの精度を上げるために、**学習データを増やす・改善する**のがデータ担当の仕事。

---

## データの中身を理解する

`train.csv`を開くと、こんな問題が9500件入っている：

### カテゴリ一覧（各約1600件ずつ、合計6カテゴリ）

#### 1. ビット操作（16.9%）
```
In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.
Here are some examples:
01010001 -> 11011101
00001001 -> 0110110
（問題）→ ?
Answer: 10010111
```
→ 2進数の変換ルールを推測して適用する

#### 2. 物理（重力定数）（16.8%）
```
In Alice's Wonderland, the gravitational constant has been secretly changed.
For t = 1.37s, distance = 14.92 m
For t = 4.27s, distance = 144.96 m
Now, determine ...
Answer: 154.62
```
→ d = 0.5 × g × t² のgが変わっている。例から新しいgを推測して計算する

#### 3. 単位変換（16.8%）
```
In Alice's Wonderland, a secret unit conversion is applied.
10.08 m becomes 6.69
17.83 m becomes 11.83
Now, convert: 25.09 m
Answer: 16.65
```
→ 変換係数を例から推測して適用する（例：1m = 0.664...）

#### 4. テキスト暗号（16.6%）
```
In Alice's Wonderland, secret encryption rules are used on text.
ucoov pwgtfyoqg -> queen discovers
pqrsfv pqorzg -> dragon dreams
bxo sfjpov pqrsfv dfjjfi → ?
Answer: cat imagines book
```
→ 暗号化されたテキストを例から解読する（シーザー暗号や置換暗号）

#### 5. 数値変換（ローマ数字等）（16.6%）
```
In Alice's Wonderland, numbers are secretly converted into a different numeral system.
11 -> XI
15 -> XV
94 -> XCIV
Now, write the number 38 in the Wonderland numeral system.
Answer: XXXVIII
```
→ 数値変換のルールを推測する（ローマ数字、2進数、16進数など）

#### 6. 記号変換（16.4%）
```
In Alice's Wonderland, a secret set of transformation rules is applied to equations.
`!*[{ = '"[`
\'*'> = ![@
Now, determine the result for: [[-!'
Answer: @&
```
→ 記号の置換ルールを推測して適用する

---

## データ担当がやること

### Step 1: 環境準備（30分）

1. Pythonをインストール（すでにある場合はスキップ）
2. 必要なライブラリをインストール：
```bash
pip install pandas openai anthropic tqdm
```

3. `train.csv`をダウンロードしてローカルに置く

### Step 2: データ分析（1時間）

まずカテゴリ別に問題を理解する。以下のスクリプトを実行：

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
print(f"総件数: {len(df)}")

# 単語数分布
df['word_count'] = df['prompt'].str.split().str.len()
print(f"\n単語数 統計:")
print(df['word_count'].describe())

# 文字数分布
df['char_count'] = df['prompt'].str.len()
print(f"\n文字数 統計:")
print(df['char_count'].describe())

# トークン数の簡易推定（文字数 / 4 が目安）
df['token_estimate'] = df['char_count'] / 4
print(f"\nトークン数推定 統計:")
print(df['token_estimate'].describe())

# 分布をグラフで見る
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df['word_count'].hist(bins=50, ax=axes[0])
axes[0].set_title('単語数分布')
df['char_count'].hist(bins=50, ax=axes[1])
axes[1].set_title('文字数分布')
df['token_estimate'].hist(bins=50, ax=axes[2])
axes[2].set_title('トークン数分布（推定）')
plt.tight_layout()
plt.savefig('data_distribution.png')
print("\nグラフを data_distribution.png に保存しました")
```

**この分析で何がわかるか：**
- トークン数が多い問題 → 学習時に切り捨てられる可能性がある（max_seq_len=2048が上限）
- トークン数が2048を超える問題は学習データから除外するか短くする必要がある
- 合成データを作る際の長さの目安になる

各カテゴリの問題を10件ずつ手で読んで、パターンを理解する。

### Step 3: 合成データ生成（メイン作業）

**目標：各カテゴリ1000件追加 = 合計6000件の新データを作る**

#### 方法：DeepSeek APIを使う（安くて数学・論理系に強い）

**DeepSeek APIキーの取得：**
1. https://platform.deepseek.com/ にアクセス
2. アカウント作成 → API Keys → Create API Key
3. キーをコピーして保存

**インストール：**
```bash
pip install openai  # DeepSeekはOpenAI互換APIなのでopenaiライブラリを使う
```

以下のスクリプトを参考に、各カテゴリの問題を自動生成する：

```python
from openai import OpenAI
import csv
import random

# DeepSeek APIの設定（OpenAI互換）
client = OpenAI(
    api_key="YOUR_DEEPSEEK_API_KEY",  # ここにDeepSeekのAPIキーを入れる
    base_url="https://api.deepseek.com"
)

def generate_problem(category_prompt):
    """問題を生成する汎用関数"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": category_prompt}],
        max_tokens=1000
    )
    return response.choices[0].message.content

# カテゴリ別のプロンプトテンプレート
PROMPTS = {
    "bit_manipulation": """
Create one 8-bit binary transformation puzzle in this exact format:

PROMPT: In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.
The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:
[3-5 examples of input -> output]

Now apply the same rule to: [new input]
ANSWER: [correct output]

Rules:
- Choose ONE consistent transformation rule (e.g., XOR with 10101010, left rotate by 2, etc.)
- All examples and answer must follow the SAME rule
- Input and output must be valid 8-bit binary strings
- Do not explain the rule
""",
    "text_encryption": """
Create one text encryption puzzle in this exact format:

PROMPT: In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
[encrypted phrase] -> [original phrase]
[encrypted phrase] -> [original phrase]
[encrypted phrase] -> [original phrase]
[encrypted phrase to decode]
ANSWER: [decoded phrase]

Rules:
- Use a consistent cipher (Caesar cipher, substitution cipher, etc.)
- Each example must follow the SAME rule
- Use simple 2-4 word phrases
- Do not explain the rule
""",
    "unit_conversion": """
Create one unit conversion puzzle in this exact format:

PROMPT: In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:
[value] m becomes [converted value]
[value] m becomes [converted value]
[value] m becomes [converted value]
[value] m becomes [converted value]
Now, convert the following measurement: [new value] m
ANSWER: [converted value]

Rules:
- Use ONE consistent conversion factor (e.g., multiply by 0.664)
- All values must follow the SAME rule
- Round answers to 2 decimal places
- Do not explain the conversion factor
""",
    "numeral_system": """
Create one numeral system conversion puzzle in this exact format:

PROMPT: In Alice's Wonderland, numbers are secretly converted into a different numeral system. Some examples are given below:
[number] -> [converted]
[number] -> [converted]
[number] -> [converted]
[number] -> [converted]
Now, write the number [new number] in the Wonderland numeral system.
ANSWER: [converted number]

Rules:
- Use a real numeral system (Roman numerals, binary, hexadecimal, octal, base-5, etc.)
- All examples must follow the SAME system
- Do not explain the system
""",
    "symbol_transformation": """
Create one symbol transformation puzzle in this exact format:

PROMPT: In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
[symbol expression] = [result]
[symbol expression] = [result]
[symbol expression] = [result]
[symbol expression] = [result]
Now, determine the result for: [new expression]
ANSWER: [result]

Rules:
- Use consistent symbol substitution rules
- Use special characters like !, @, #, $, %, ^, &, *, [, ], {, }, \, /, |, <, >, +, -, =
- All examples must follow the SAME rules
- Do not explain the rules
""",
    "physics": """
Create one physics puzzle where gravitational constant is changed in this exact format:

PROMPT: In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations:
For t = [time]s, distance = [distance] m
For t = [time]s, distance = [distance] m
For t = [time]s, distance = [distance] m
For t = [time]s, distance = [distance] m
For t = [time]s, distance = [distance] m
Now, determine the distance for t = [new time]s
ANSWER: [distance]

Rules:
- Use formula: distance = 0.5 * g * t^2 where g is a secret value between 5 and 20
- All examples must use the SAME g value
- Round answers to 2 decimal places
- Do not reveal the value of g
"""
}

def generate_all_categories(num_per_category=100):
    """全カテゴリの問題を生成"""
    all_rows = []
    
    for category, prompt_template in PROMPTS.items():
        print(f"\n=== {category} を生成中 ===")
        count = 0
        attempts = 0
        
        while count < num_per_category and attempts < num_per_category * 2:
            attempts += 1
            try:
                result = generate_problem(prompt_template)
                
                # PROMPT:とANSWER:を抽出
                if "PROMPT:" in result and "ANSWER:" in result:
                    prompt_part = result.split("PROMPT:")[1].split("ANSWER:")[0].strip()
                    answer_part = result.split("ANSWER:")[1].strip().split("\n")[0].strip()
                    
                    if prompt_part and answer_part:
                        row_id = f"syn_{category[:3]}_{count:05d}"
                        all_rows.append({
                            "id": row_id,
                            "prompt": prompt_part,
                            "answer": answer_part
                        })
                        count += 1
                        print(f"  {count}/{num_per_category} 完了")
                        
            except Exception as e:
                print(f"  エラー（スキップ）: {e}")
        
        print(f"{category}: {count}件生成完了")
    
    return all_rows

# 実行（まず各カテゴリ10件でテスト）
print("テスト生成開始（各カテゴリ10件）...")
rows = generate_all_categories(num_per_category=10)

# CSVに保存
with open('synthetic_data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer'])
    writer.writeheader()
    writer.writerows(rows)

print(f"\n保存完了: {len(rows)}件 → synthetic_data.csv")
print("問題なければ num_per_category=1000 に変えて本番生成してください")
```

### Step 4: 品質チェック

生成したデータを確認する：

```python
import pandas as pd

df = pd.read_csv('synthetic_bit_manipulation.csv')

# 1. 空のデータがないか確認
print(df.isnull().sum())

# 2. 答えが空でないか確認
empty_answers = df[df['answer'].str.strip() == '']
print(f"答えが空: {len(empty_answers)}件")

# 3. サンプルを目視確認
for _, row in df.head(5).iterrows():
    print("=== プロンプト ===")
    print(row['prompt'][:200])
    print(f"答え: {row['answer']}")
    print()
```

問題があるデータは手で修正または削除する。

### Step 5: 最終データをまとめる

```python
import pandas as pd

# 元データ読み込み
original = pd.read_csv('train.csv')

# 合成データを全部読み込んで結合
import glob
synthetic_files = glob.glob('synthetic_*.csv')
synthetic_dfs = [pd.read_csv(f) for f in synthetic_files]
synthetic = pd.concat(synthetic_dfs, ignore_index=True)

# IDが重複しないように確認
print(f"元データ: {len(original)}件")
print(f"合成データ: {len(synthetic)}件")

# 結合
combined = pd.concat([original, synthetic], ignore_index=True)
combined.to_csv('train_augmented.csv', index=False)
print(f"結合後: {len(combined)}件")
```

---

## 優先順位

1. **まずビット操作と記号変換**から始める（パターンが明確で生成しやすい）
2. 次に**テキスト暗号と数値変換**
3. 最後に**物理と単位変換**（数値の正確さが必要で難しい）

---

## 注意事項

- **答えが正しいことを必ず確認する**（LLMが間違えることがある）
- 特に数値問題（物理・単位変換）は手計算で検証する
- 生成したデータは必ずサンプルを目視確認する
- API料金に注意（Claude APIは1000件で約$1〜2程度）

---

## 完成したら渡すもの

- `train_augmented.csv`（元データ＋合成データ）
- 何件生成したか、どんな品質チェックをしたかのメモ

わからないことがあれば気軽に聞いてください！
