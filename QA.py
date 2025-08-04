
#!/usr/bin/env python
"""
交互式 / 命令行 QA 示例
-------------------------------------------------
用法 1：交互模式
    python qa_demo.py

用法 2：命令行一次性查询
    python qa_demo.py "How to learn Python?"

环境要求：已微调模型、准备好的 qa_pairs.csv
目录默认与 BertModel.py 相同，可通过 --model / --data 参数修改
"""

import argparse
import pathlib
import sys

import pandas as pd

# 复用已实现的类
from BertModel import BERTQuestionEncoder, QADatabase


def build_system(model_dir: str, data_csv: str) -> QADatabase:
    print("🔧 Loading encoder & building index ...")
    encoder = BERTQuestionEncoder(model_dir)
    db = QADatabase(encoder)
    db.build(pd.read_csv(data_csv))
    return db


def interactive_loop(db: QADatabase):
    print("\n=== 进入交互模式，输入空行或 exit 退出 ===")
    while True:
        q = input("你问> ").strip()
        if q.lower() in {"", "exit", "quit", "q"}:
            break
        top = db.search(q, top_k=5)
        print("-" * 60)
        for hit in top:
            print(f"[{hit['rank']}] {hit['score']:.3f}  A: {hit['answer']}")
            print(f"     Q: {hit['question']}\n")
        print("-" * 60)


def main():
    default_model = "./models/finetuned-all-MiniLM-L6-v2"
    default_data = "./processed_data/qa_pairs.csv"

    parser = argparse.ArgumentParser(description="QA Retrieval Demo")
    parser.add_argument("query", nargs="?", help="直接查询的文本（留空进入交互模式）")
    parser.add_argument("--model", default=default_model, help="微调模型目录")
    parser.add_argument("--data", default=default_data, help="包含 question/answer 的 csv 路径")
    args = parser.parse_args()

    # 路径检查
    for p in [args.model, args.data]:
        if not pathlib.Path(p).exists():
            sys.exit(f"❌ 路径不存在：{p}")

    qa_db = build_system(args.model, args.data)

    if args.query:
        ans = qa_db.search(args.query, top_k=3)[0]
        print(f"\nQ: {args.query}\nA: {ans['answer']}\n(score={ans['score']:.3f})")
    else:
        interactive_loop(qa_db)


if __name__ == "__main__":
    main()
