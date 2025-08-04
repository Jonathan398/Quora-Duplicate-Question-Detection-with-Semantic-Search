
import os
import sys
import argparse
import random
import warnings
from typing import List, Dict

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import InputExample

warnings.filterwarnings("ignore")

# ======================================================================================
# Core Functional Classes
# ======================================================================================


class BERTQuestionEncoder:
    """Handles loading the model and encoding text into vectors."""

    def __init__(self, model_name_or_path, max_length=128, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name_or_path = model_name_or_path

        print(f"🤖 Loading Model: '{self.model_name_or_path}'")
        print(f"💻 Using device: {self.device}")

        self.model = SentenceTransformer(model_name_or_path, device=self.device)
        self.model.max_seq_length = max_length
        print("✅ Model loaded successfully.")

    def encode(
            self,
            texts: List[str],
            batch_size: int = 32,
            show_progress_bar: bool = True,
    ) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )

    def analyze_embedding_similarity(self, embeddings: np.ndarray) -> Dict:
        """Diagnose embedding anisotropy via cosine-similarity stats."""
        sample_size = min(1000, len(embeddings))
        idx = np.random.choice(len(embeddings), sample_size, replace=False)
        emb = embeddings[idx]
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        sims = cosine_similarity(emb)
        pairwise = sims[np.triu_indices(sample_size, k=1)]
        return {"mean": float(np.mean(pairwise)), "std": float(np.std(pairwise))}


class QADatabase:
    """Stores QA data, builds and queries a Faiss index."""

    def __init__(self, encoder: BERTQuestionEncoder):
        self.encoder = encoder
        self.questions, self.answers, self.metadata = [], [], []
        self.embeddings = None
        self.index = None
        self.embedding_analysis = {}

    def build(
            self,
            df: pd.DataFrame,
            pair_id_col="pair_id",
            question_col="question",
            answer_col="answer",
    ):
        print(f"\n📚 Building database from DataFrame with {len(df)} rows...")
        self.questions = df[question_col].astype(str).tolist()
        self.answers = df[answer_col].astype(str).tolist()
        self.metadata = df.to_dict("records")

        print("\n🧠 Encoding questions to vectors...")
        self.embeddings = self.encoder.encode(self.questions)

        print("\n🔍 Analyzing embedding space...")
        self.embedding_analysis = self.encoder.analyze_embedding_similarity(
            self.embeddings
        )

        print("\n⚡ Building FAISS index for fast search...")
        self._build_faiss_index()

    def _build_faiss_index(self):
        dim = self.embeddings.shape[1]
        emb_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb_norm.astype(np.float32))
        print(f"✅ FAISS index built with {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        q_emb = self.encoder.encode([query], show_progress_bar=False)
        q_emb = q_emb / np.linalg.norm(q_emb)
        scores, idxs = self.index.search(q_emb.astype(np.float32), top_k)
        return [
            {
                "rank": i + 1,
                "score": float(s),
                "question": self.questions[j],
                "answer": self.answers[j],
                "metadata": self.metadata[j],
            }
            for i, (s, j) in enumerate(zip(scores[0], idxs[0]))
        ]


class QASystemEvaluator:
    """Evaluates the QA system and prints a report."""

    def __init__(self, db: QADatabase):
        self.db = db
        self.results = {}
        self.detailed_results = []
        self.pair_id_col = "pair_id"

    def run(self, test_set: List[Dict], k_values: List[int] = [1, 3, 5, 10]):
        print(f"\n🚀 Running evaluation on {len(test_set)} test queries...")
        max_k = max(k_values)
        for item in tqdm(test_set, desc="Evaluating"):
            query = item["question"]
            expected_id = item[self.pair_id_col]
            retrieved = self.db.search(query, top_k=max_k)
            relevance = [
                1 if r["metadata"][self.pair_id_col] == expected_id else 0 for r in retrieved
            ]
            self.detailed_results.append(
                {
                    "query": query,
                    "expected_pair_id": expected_id,
                    "retrieved": retrieved,
                    "relevance": relevance,
                }
            )
        self._calc_metrics(k_values)
        self._report(k_values)

    def _calc_metrics(self, k_values):
        metrics = {f"{m}@{k}": [] for k in k_values for m in ["Recall", "Precision", "AP", "NDCG"]}
        metrics["MRR"] = []
        for res in self.detailed_results:
            rel = res["relevance"]
            # MRR
            metrics["MRR"].append(1 / (rel.index(1) + 1) if 1 in rel else 0.0)
            # Top-K metrics
            for k in k_values:
                rel_k = rel[:k]
                hits = sum(rel_k)
                metrics[f"Recall@{k}"].append(1.0 if hits > 0 else 0.0)
                metrics[f"Precision@{k}"].append(hits / k)
                # MAP
                running_hits, ap = 0, 0
                for i, r in enumerate(rel_k):
                    if r:
                        running_hits += 1
                        ap += running_hits / (i + 1)
                metrics[f"AP@{k}"].append(ap / max(1, running_hits))
                # NDCG
                dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel_k))
                idcg = sum(sorted(rel_k, reverse=True)[i] / np.log2(i + 2) for i in range(len(rel_k)))
                metrics[f"NDCG@{k}"].append(dcg / idcg if idcg else 0.0)
        self.results = {k: float(np.mean(v)) for k, v in metrics.items()}

    def _report(self, k_values):
        print("\n" + "=" * 80)
        print(" " * 20 + "QA Retrieval System Evaluation Report")
        print("=" * 80)
        print(f"\nOverall MRR: {self.results['MRR']:.4f}\n")
        df = pd.DataFrame(
            {
                "k": k_values,
                "Recall": [self.results[f"Recall@{k}"] for k in k_values],
                "Precision": [self.results[f"Precision@{k}"] for k in k_values],
                "MAP": [self.results[f"AP@{k}"] for k in k_values],
                "NDCG": [self.results[f"NDCG@{k}"] for k in k_values],
            }
        )
        print(df.to_string(index=False, float_format="%.4f"))
        self._plot(df)

    def _plot(self, df):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Retrieval Performance vs. Top-K", fontsize=16)
        metrics = ["Recall", "Precision", "MAP", "NDCG"]
        for ax, m in zip(axes.flat, metrics):
            df.plot(x="k", y=m, marker="o", ax=ax, title=f"{m} @ K")
            ax.set_xlabel("K")
            ax.grid(True, linestyle="--")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("retrieval_performance.png")
        print("✅ Performance plot saved as 'retrieval_performance.png'")


# ======================================================================================
# Model Fine-Tuning
# ======================================================================================


class ModelFinetuner:
    def __init__(self, base_model_path: str, save_path: str):
        self.base_model_path = base_model_path
        self.save_path = save_path

    @staticmethod
    def _make_triplets(df, q_col="question", id_col="pair_id"):
        grouped = df.groupby(id_col)[q_col].apply(list).to_dict()
        triplets, pids = [], list(grouped.keys())
        for gid, positives in tqdm(grouped.items(), desc="Generating Triplets"):
            if len(positives) < 2:
                continue
            for i in range(len(positives)):
                for j in range(i + 1, len(positives)):
                    neg_gid = random.choice([pid for pid in pids if pid != gid])
                    triplets.append(
                        InputExample(
                            texts=[positives[i], positives[j], random.choice(grouped[neg_gid])]
                        )
                    )
        print(f"✅ Created {len(triplets)} triplets for training.")
        return triplets

    def run(self, train_df: pd.DataFrame, epochs=1, batch_size=32):
        print("\n" + "=" * 80 + "\n" + " " * 28 + "Starting Model Fine-Tuning\n" + "=" * 80)
        model = SentenceTransformer(self.base_model_path)
        triplets = self._make_triplets(train_df)
        if not triplets:
            print("❌ No triplets generated; skipping fine-tuning.")
            os.makedirs(self.save_path, exist_ok=True)
            model.save(self.save_path)
            return

        loader = DataLoader(
            triplets,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=model.smart_batching_collate,
        )
        criterion = losses.TripletLoss(model=model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

        steps_per_epoch = len(loader)
        total_steps = steps_per_epoch * epochs
        warmup_steps = max(1, int(0.1 * total_steps))
        scheduler = None
        if warmup_steps:
            from transformers import get_linear_schedule_with_warmup

            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )

        device = model.device
        print(f"\n🔥 Training for {epochs} epoch(s), {steps_per_epoch} steps per epoch\n")

        for epoch in range(epochs):
            epoch_loss = 0.0
            progress = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
            for features, labels in progress:
                # move features to correct device
                for feat in features:
                    for key in feat:
                        if isinstance(feat[key], torch.Tensor):
                            feat[key] = feat[key].to(device)

                optimizer.zero_grad()
                loss = criterion(features, labels)
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

                epoch_loss += loss.item()
                progress.set_postfix({"loss": f"{loss.item():.4f}"})

            print(f"Epoch {epoch+1} average loss: {epoch_loss / steps_per_epoch:.4f}")

        os.makedirs(self.save_path, exist_ok=True)
        model.save(self.save_path)
        print(f"\n✅ Fine-tuning complete. Model saved to '{self.save_path}'")


# ======================================================================================
# Main
# ======================================================================================


def check_files_exist(paths: List[str]) -> bool:
    for p in paths:
        if not os.path.exists(p):
            print(f"❌ Missing required file: {p}")
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="训练 epoch 数")
    args = parser.parse_args()

    BASE_MODEL = "./models/all-MiniLM-L6-v2"
    FINETUNED_MODEL = "./models/finetuned-all-MiniLM-L6-v2"
    DATA_DIR = "./processed_data"

    train_file = os.path.join(DATA_DIR, "train_qa.csv")
    test_file = os.path.join(DATA_DIR, "test_qa.csv")
    all_pairs_file = os.path.join(DATA_DIR, "qa_pairs.csv")

    if not check_files_exist([train_file, test_file, all_pairs_file, BASE_MODEL]):
        sys.exit(1)

    # 1. Fine-tune
    train_df = pd.read_csv(train_file)
    ModelFinetuner(BASE_MODEL, FINETUNED_MODEL).run(train_df, epochs=args.epochs, batch_size=32)

    # 2. Evaluate
    encoder = BERTQuestionEncoder(FINETUNED_MODEL)
    qa_db = QADatabase(encoder)
    qa_db.build(pd.read_csv(all_pairs_file))

    evaluator = QASystemEvaluator(qa_db)
    evaluator.run(pd.read_csv(test_file).to_dict("records"))


if __name__ == "__main__":
    main()
