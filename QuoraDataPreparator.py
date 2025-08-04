"""
Quora Question Pairs Dataset Preparation Script with Progress Indicators
Enhanced version with detailed progress feedback
"""

import pandas as pd
import numpy as np
import os
import re
import string
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings('ignore')


class QuoraDataPreparator:
    def __init__(self, data_dir='data'):
        """
        Initialize Quora Data Preparator

        Args:
            data_dir: Directory containing train.csv and test.csv
        """
        self.data_dir = data_dir
        self.train_file = os.path.join(data_dir, 'train.csv')
        self.test_file = os.path.join(data_dir, 'test.csv')

        self.raw_train_data = None
        self.raw_test_data = None
        self.cleaned_train_data = None
        self.cleaned_test_data = None
        self.qa_pairs = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def get_file_size(self, file_path):
        """Get file size in MB"""
        if os.path.exists(file_path):
            size_bytes = os.path.getsize(file_path)
            size_mb = size_bytes / (1024 * 1024)
            return size_mb
        return 0

    def load_data(self, sample_size=None, chunk_size=50000):
        """
        Load train and test data files with progress indicators

        Args:
            sample_size: If specified, load only specified number of samples
            chunk_size: Size of chunks for progress reporting
        """
        print("=" * 60)
        print("STEP 1: LOADING DATA FILES")
        print("=" * 60)

        # Check if files exist and show sizes
        if not os.path.exists(self.train_file):
            raise FileNotFoundError(f"Training file not found: {self.train_file}")

        train_size_mb = self.get_file_size(self.train_file)
        print(f"📁 Training file: {self.train_file}")
        print(f"📊 File size: {train_size_mb:.1f} MB")

        if os.path.exists(self.test_file):
            test_size_mb = self.get_file_size(self.test_file)
            print(f"📁 Test file: {self.test_file}")
            print(f"📊 Test file size: {test_size_mb:.1f} MB")
        else:
            print(f"⚠️  Test file not found: {self.test_file}")
            self.raw_test_data = None

        try:
            # Load training data with progress
            print(f"\n🔄 Loading training data...")
            start_time = time.time()

            if sample_size:
                print(f"   📝 Loading sample of {sample_size:,} rows...")
                self.raw_train_data = pd.read_csv(self.train_file, nrows=sample_size)
                print(f"   ✅ Loaded {len(self.raw_train_data):,} rows in {time.time() - start_time:.1f}s")
            else:
                # For large files, show progress by estimating rows
                if train_size_mb > 100:  # If file is larger than 100MB
                    print(f"   📝 Large file detected. This might take a while...")
                    print(f"   🔄 Reading CSV file... (Please wait)")

                self.raw_train_data = pd.read_csv(self.train_file)
                print(
                    f"   ✅ Loaded complete dataset: {len(self.raw_train_data):,} rows in {time.time() - start_time:.1f}s")

            # Load test data if available
            if os.path.exists(self.test_file):
                print(f"\n🔄 Loading test data...")
                start_time = time.time()
                self.raw_test_data = pd.read_csv(self.test_file)
                print(f"   ✅ Loaded test data: {len(self.raw_test_data):,} rows in {time.time() - start_time:.1f}s")

            # Display basic information
            self.display_basic_info()

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

    def display_basic_info(self):
        """Display basic dataset information with better formatting"""
        print(f"\n" + "=" * 60)
        print("DATASET INFORMATION")
        print("=" * 60)

        print(f"\n📊 TRAINING DATA:")
        print(f"   Shape: {self.raw_train_data.shape}")
        print(f"   Columns: {list(self.raw_train_data.columns)}")
        print(f"   Memory usage: {self.raw_train_data.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")

        print(f"\n📈 DATA TYPES:")
        for col, dtype in self.raw_train_data.dtypes.items():
            print(f"   {col}: {dtype}")

        print(f"\n🔍 MISSING VALUES:")
        missing = self.raw_train_data.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                print(f"   {col}: {count:,} ({count / len(self.raw_train_data) * 100:.1f}%)")
            else:
                print(f"   {col}: No missing values ✅")

        if 'is_duplicate' in self.raw_train_data.columns:
            print(f"\n🏷️  DUPLICATE LABELS:")
            label_counts = self.raw_train_data['is_duplicate'].value_counts()
            total = len(self.raw_train_data)
            print(f"   Non-duplicate (0): {label_counts[0]:,} ({label_counts[0] / total * 100:.1f}%)")
            print(f"   Duplicate (1): {label_counts[1]:,} ({label_counts[1] / total * 100:.1f}%)")

        print(f"\n📝 SAMPLE DATA:")
        print(self.raw_train_data.head(3).to_string())

        if self.raw_test_data is not None:
            print(f"\n📊 TEST DATA:")
            print(f"   Shape: {self.raw_test_data.shape}")
            print(f"   Columns: {list(self.raw_test_data.columns)}")

    def clean_text(self, text):
        """Clean individual text with progress-friendly processing"""
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\?\.\!\,\-\'\"]', '', text)
        text = text.strip()

        return text

    def clean_data(self):
        """Clean training and test data with detailed progress"""
        print(f"\n" + "=" * 60)
        print("STEP 2: DATA CLEANING")
        print("=" * 60)

        # Clean training data
        print(f"🧹 Cleaning training data...")
        self.cleaned_train_data = self.raw_train_data.copy()

        # Remove rows with missing values in questions
        initial_shape = self.cleaned_train_data.shape[0]
        print(f"   🔍 Checking for missing values in questions...")
        self.cleaned_train_data = self.cleaned_train_data.dropna(subset=['question1', 'question2'])
        removed_missing = initial_shape - self.cleaned_train_data.shape[0]
        print(f"   ✅ Removed {removed_missing:,} rows with missing values")
        print(f"   📊 Remaining: {self.cleaned_train_data.shape[0]:,} rows")

        # Clean text in training data
        print(f"\n   🔤 Cleaning question text...")
        total_questions = len(self.cleaned_train_data) * 2  # 2 questions per row

        print(f"   🔄 Processing question1 ({len(self.cleaned_train_data):,} questions)...")
        tqdm.pandas(desc="   Cleaning Q1", ncols=80)
        self.cleaned_train_data['question1_clean'] = self.cleaned_train_data['question1'].progress_apply(
            self.clean_text)

        print(f"   🔄 Processing question2 ({len(self.cleaned_train_data):,} questions)...")
        tqdm.pandas(desc="   Cleaning Q2", ncols=80)
        self.cleaned_train_data['question2_clean'] = self.cleaned_train_data['question2'].progress_apply(
            self.clean_text)

        # Remove rows with empty text after cleaning
        before_empty_removal = self.cleaned_train_data.shape[0]
        print(f"\n   🔍 Removing rows with empty text after cleaning...")
        self.cleaned_train_data = self.cleaned_train_data[
            (self.cleaned_train_data['question1_clean'].str.len() > 0) &
            (self.cleaned_train_data['question2_clean'].str.len() > 0)
            ]
        removed_empty = before_empty_removal - self.cleaned_train_data.shape[0]
        print(f"   ✅ Removed {removed_empty:,} rows with empty text")
        print(f"   📊 Remaining: {self.cleaned_train_data.shape[0]:,} rows")

        # Remove completely duplicate rows
        before_duplicate_removal = self.cleaned_train_data.shape[0]
        print(f"\n   🔍 Removing duplicate question pairs...")
        self.cleaned_train_data = self.cleaned_train_data.drop_duplicates(
            subset=['question1_clean', 'question2_clean']
        )
        removed_dupes = before_duplicate_removal - self.cleaned_train_data.shape[0]
        print(f"   ✅ Removed {removed_dupes:,} duplicate pairs")
        print(f"   📊 Final training data: {self.cleaned_train_data.shape[0]:,} rows")

        # Add text length statistics
        print(f"\n   📏 Calculating text lengths...")
        self.cleaned_train_data['q1_len'] = self.cleaned_train_data['question1_clean'].str.len()
        self.cleaned_train_data['q2_len'] = self.cleaned_train_data['question2_clean'].str.len()

        # Clean test data if available
        if self.raw_test_data is not None:
            print(f"\n🧹 Cleaning test data...")
            self.cleaned_test_data = self.raw_test_data.copy()

            print(f"   🔄 Processing test question1...")
            tqdm.pandas(desc="   Test Q1", ncols=80)
            self.cleaned_test_data['question1_clean'] = self.cleaned_test_data['question1'].progress_apply(
                self.clean_text)

            print(f"   🔄 Processing test question2...")
            tqdm.pandas(desc="   Test Q2", ncols=80)
            self.cleaned_test_data['question2_clean'] = self.cleaned_test_data['question2'].progress_apply(
                self.clean_text)

            # Add length statistics
            self.cleaned_test_data['q1_len'] = self.cleaned_test_data['question1_clean'].str.len()
            self.cleaned_test_data['q2_len'] = self.cleaned_test_data['question2_clean'].str.len()

            print(f"   ✅ Test data cleaned: {len(self.cleaned_test_data):,} pairs")

        print(f"\n✅ Data cleaning completed!")
        self.analyze_cleaned_data()

    def analyze_cleaned_data(self):
        """Analyze cleaned data with detailed statistics"""
        print(f"\n" + "=" * 60)
        print("STEP 3: DATA ANALYSIS")
        print("=" * 60)

        # Training data analysis
        print(f"📊 TRAINING DATA STATISTICS:")
        q1_avg = self.cleaned_train_data['q1_len'].mean()
        q1_med = self.cleaned_train_data['q1_len'].median()
        q2_avg = self.cleaned_train_data['q2_len'].mean()
        q2_med = self.cleaned_train_data['q2_len'].median()

        print(f"   Question1 - Average: {q1_avg:.1f} chars, Median: {q1_med:.1f} chars")
        print(f"   Question2 - Average: {q2_avg:.1f} chars, Median: {q2_med:.1f} chars")

        if 'is_duplicate' in self.cleaned_train_data.columns:
            label_counts = self.cleaned_train_data['is_duplicate'].value_counts()
            total = len(self.cleaned_train_data)
            print(f"\n🏷️  LABEL DISTRIBUTION:")
            print(f"   Non-duplicate pairs: {label_counts[0]:,} ({label_counts[0] / total * 100:.1f}%)")
            print(f"   Duplicate pairs: {label_counts[1]:,} ({label_counts[1] / total * 100:.1f}%)")

        # Test data analysis
        if self.cleaned_test_data is not None:
            print(f"\n📊 TEST DATA STATISTICS:")
            tq1_avg = self.cleaned_test_data['q1_len'].mean()
            tq2_avg = self.cleaned_test_data['q2_len'].mean()
            print(f"   Question1 - Average: {tq1_avg:.1f} chars")
            print(f"   Question2 - Average: {tq2_avg:.1f} chars")

        # Create visualizations
        print(f"\n🎨 Creating data visualizations...")
        self.plot_data_analysis()
        print(f"   ✅ Charts saved as 'data_analysis.png'")

    def plot_data_analysis(self):
        """Create data analysis plots"""
        plt.figure(figsize=(15, 10))

        # Training data plots
        plt.subplot(2, 3, 1)
        plt.hist(self.cleaned_train_data['q1_len'], bins=50, alpha=0.7, color='blue', label='Question1')
        plt.hist(self.cleaned_train_data['q2_len'], bins=50, alpha=0.7, color='red', label='Question2')
        plt.title('Training: Question Length Distribution')
        plt.xlabel('Length (characters)')
        plt.ylabel('Frequency')
        plt.legend()

        if 'is_duplicate' in self.cleaned_train_data.columns:
            plt.subplot(2, 3, 2)
            counts = self.cleaned_train_data['is_duplicate'].value_counts()
            plt.bar(['Non-Duplicate', 'Duplicate'], counts.values, color=['red', 'blue'])
            plt.title('Training: Label Distribution')
            plt.ylabel('Count')

            # Add percentage labels
            total = counts.sum()
            for i, v in enumerate(counts.values):
                plt.text(i, v + total * 0.01, f'{v:,}\n({v / total * 100:.1f}%)', ha='center')

        # Test data plots
        if self.cleaned_test_data is not None:
            plt.subplot(2, 3, 4)
            plt.hist(self.cleaned_test_data['q1_len'], bins=50, alpha=0.7, color='green', label='Question1')
            plt.hist(self.cleaned_test_data['q2_len'], bins=50, alpha=0.7, color='orange', label='Question2')
            plt.title('Test: Question Length Distribution')
            plt.xlabel('Length (characters)')
            plt.ylabel('Frequency')
            plt.legend()

        # Length comparison
        plt.subplot(2, 3, 3)
        plt.scatter(self.cleaned_train_data['q1_len'], self.cleaned_train_data['q2_len'],
                    alpha=0.5, s=1, c='blue')
        plt.title('Training: Q1 vs Q2 Length')
        plt.xlabel('Question1 Length')
        plt.ylabel('Question2 Length')

        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close to save memory

    def create_qa_pairs(self):
        """Convert question pairs to QA pairs with progress tracking"""
        print(f"\n" + "=" * 60)
        print("STEP 4: CREATING QA PAIRS")
        print("=" * 60)

        if 'is_duplicate' not in self.cleaned_train_data.columns:
            print("⚠️  No 'is_duplicate' column found. Using all question pairs.")
            duplicate_pairs = self.cleaned_train_data
        else:
            # Use only duplicate question pairs
            duplicate_pairs = self.cleaned_train_data[self.cleaned_train_data['is_duplicate'] == 1]

        print(f"🔄 Processing {len(duplicate_pairs):,} duplicate question pairs...")
        print(f"   Each pair will create 2 QA pairs (bidirectional)")
        print(f"   Expected total: {len(duplicate_pairs) * 2:,} QA pairs")

        qa_list = []

        # Process with progress bar
        for _, row in tqdm(duplicate_pairs.iterrows(),
                           desc="Creating QA pairs",
                           total=len(duplicate_pairs),
                           ncols=80):
            # Create bidirectional QA pairs: q1->q2 and q2->q1
            qa_list.append({
                'question': row['question1_clean'],
                'answer': row['question2_clean'],
                'original_q1': row['question1'],
                'original_q2': row['question2'],
                'q_len': row['q1_len'],
                'a_len': row['q2_len'],
                'pair_id': row['id'] if 'id' in row else len(qa_list)
            })

            qa_list.append({
                'question': row['question2_clean'],
                'answer': row['question1_clean'],
                'original_q1': row['question2'],
                'original_q2': row['question1'],
                'q_len': row['q2_len'],
                'a_len': row['q1_len'],
                'pair_id': row['id'] if 'id' in row else len(qa_list)
            })

        self.qa_pairs = pd.DataFrame(qa_list)
        print(f"✅ Created {len(self.qa_pairs):,} QA pairs")

        # Display QA pair samples
        print(f"\n📝 QA PAIR SAMPLES:")
        for i in range(min(3, len(self.qa_pairs))):
            print(f"\n   Sample {i + 1}:")
            print(f"   Q: {self.qa_pairs.iloc[i]['question']}")
            print(f"   A: {self.qa_pairs.iloc[i]['answer']}")

        return self.qa_pairs

    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """Split dataset with progress indicators"""
        print(f"\n" + "=" * 60)
        print("STEP 5: SPLITTING DATASET")
        print("=" * 60)

        print(f"📊 Split configuration:")
        print(f"   Training: {train_ratio * 100:.0f}%")
        print(f"   Validation: {val_ratio * 100:.0f}%")
        print(f"   Test: {test_ratio * 100:.0f}%")

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1")

        if self.qa_pairs is None:
            raise ValueError("Please create QA pairs first")

        total_pairs = len(self.qa_pairs)
        print(f"\n🔄 Splitting {total_pairs:,} QA pairs...")

        # First split: separate test set
        print(f"   Step 1: Separating test set...")
        train_val_data, test_data = train_test_split(
            self.qa_pairs,
            test_size=test_ratio,
            random_state=random_state,
            stratify=None
        )

        # Second split: separate validation set from train-val set
        print(f"   Step 2: Separating validation set...")
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_size_adjusted,
            random_state=random_state
        )

        self.train_data = train_data.reset_index(drop=True)
        self.val_data = val_data.reset_index(drop=True)
        self.test_data = test_data.reset_index(drop=True)

        print(f"\n✅ Dataset split completed:")
        print(f"   Training set: {len(self.train_data):,} samples ({len(self.train_data) / total_pairs * 100:.1f}%)")
        print(f"   Validation set: {len(self.val_data):,} samples ({len(self.val_data) / total_pairs * 100:.1f}%)")
        print(f"   Test set: {len(self.test_data):,} samples ({len(self.test_data) / total_pairs * 100:.1f}%)")

        # Statistics
        self.analyze_splits()

    def analyze_splits(self):
        """Analyze dataset split results"""
        print(f"\n📊 SPLIT ANALYSIS:")

        datasets = {
            'Training': self.train_data,
            'Validation': self.val_data,
            'Test': self.test_data
        }

        for name, data in datasets.items():
            avg_q_len = data['q_len'].mean()
            avg_a_len = data['a_len'].mean()
            max_q_len = data['q_len'].max()
            min_q_len = data['q_len'].min()

            print(f"\n   {name} Set:")
            print(f"     Samples: {len(data):,}")
            print(f"     Avg question length: {avg_q_len:.1f} chars")
            print(f"     Avg answer length: {avg_a_len:.1f} chars")
            print(f"     Question length range: {min_q_len}-{max_q_len} chars")

    def save_processed_data(self, output_dir='processed_data'):
        """Save all processed data with progress tracking"""
        print(f"\n" + "=" * 60)
        print("STEP 6: SAVING PROCESSED DATA")
        print("=" * 60)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")

        saved_files = []

        # Save cleaned training data
        if self.cleaned_train_data is not None:
            train_file = os.path.join(output_dir, 'cleaned_train_data.csv')
            print(f"💾 Saving cleaned training data...")
            self.cleaned_train_data.to_csv(train_file, index=False)
            size_mb = self.get_file_size(train_file)
            print(f"   ✅ {train_file} ({size_mb:.1f} MB)")
            saved_files.append(train_file)

        # Save cleaned test data
        if self.cleaned_test_data is not None:
            test_file = os.path.join(output_dir, 'cleaned_test_data.csv')
            print(f"💾 Saving cleaned test data...")
            self.cleaned_test_data.to_csv(test_file, index=False)
            size_mb = self.get_file_size(test_file)
            print(f"   ✅ {test_file} ({size_mb:.1f} MB)")
            saved_files.append(test_file)

        # Save QA pairs
        if self.qa_pairs is not None:
            qa_file = os.path.join(output_dir, 'qa_pairs.csv')
            print(f"💾 Saving QA pairs...")
            self.qa_pairs.to_csv(qa_file, index=False)
            size_mb = self.get_file_size(qa_file)
            print(f"   ✅ {qa_file} ({size_mb:.1f} MB)")
            saved_files.append(qa_file)

        # Save split datasets
        if all([self.train_data is not None, self.val_data is not None, self.test_data is not None]):
            print(f"💾 Saving split datasets...")

            train_file = os.path.join(output_dir, 'train_qa.csv')
            self.train_data.to_csv(train_file, index=False)
            print(f"   ✅ {train_file} ({self.get_file_size(train_file):.1f} MB)")

            val_file = os.path.join(output_dir, 'val_qa.csv')
            self.val_data.to_csv(val_file, index=False)
            print(f"   ✅ {val_file} ({self.get_file_size(val_file):.1f} MB)")

            test_file = os.path.join(output_dir, 'test_qa.csv')
            self.test_data.to_csv(test_file, index=False)
            print(f"   ✅ {test_file} ({self.get_file_size(test_file):.1f} MB)")

            saved_files.extend([train_file, val_file, test_file])

        # Save test data for inference
        inference_data = self.prepare_test_data_for_inference()
        if inference_data is not None:
            inference_file = os.path.join(output_dir, 'test_inference.csv')
            print(f"💾 Saving test inference data...")
            inference_data.to_csv(inference_file, index=False)
            print(f"   ✅ {inference_file} ({self.get_file_size(inference_file):.1f} MB)")
            saved_files.append(inference_file)

        # Save data statistics
        stats_file = os.path.join(output_dir, 'data_statistics.txt')
        print(f"💾 Saving data statistics...")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("Quora Dataset Processing Statistics\n")
            f.write("=" * 50 + "\n\n")

            if self.raw_train_data is not None:
                f.write(f"Original training dataset size: {len(self.raw_train_data):,}\n")

            if self.cleaned_train_data is not None:
                f.write(f"Cleaned training dataset size: {len(self.cleaned_train_data):,}\n")

            if self.raw_test_data is not None:
                f.write(f"Test dataset size: {len(self.raw_test_data):,}\n")

            if self.qa_pairs is not None:
                f.write(f"QA pairs count: {len(self.qa_pairs):,}\n")

            if self.train_data is not None:
                f.write(f"Training set size: {len(self.train_data):,}\n")
                f.write(f"Validation set size: {len(self.val_data):,}\n")
                f.write(f"Test set size: {len(self.test_data):,}\n")

        print(f"   ✅ {stats_file}")
        saved_files.append(stats_file)

        print(f"\n✅ All data saved successfully!")
        print(f"📁 Total files saved: {len(saved_files)}")

        return saved_files

    def prepare_test_data_for_inference(self):
        """Prepare test data for model inference"""
        if self.cleaned_test_data is None:
            return None

        inference_data = []
        for _, row in self.cleaned_test_data.iterrows():
            inference_data.append({
                'test_id': row['test_id'] if 'test_id' in row else len(inference_data),
                'question1': row['question1_clean'],
                'question2': row['question2_clean'],
                'original_q1': row['question1'],
                'original_q2': row['question2'],
                'q1_len': row['q1_len'],
                'q2_len': row['q2_len']
            })

        return pd.DataFrame(inference_data)

    def create_sample_queries(self, num_samples=50):
        """Create sample queries for testing"""
        if self.test_data is None:
            print("Please split the dataset first")
            return None

        print(f"\n🔄 Creating {num_samples} sample queries...")

        # Sample from test set
        sample_data = self.test_data.sample(n=min(num_samples, len(self.test_data)), random_state=42)

        sample_queries = []
        for _, row in sample_data.iterrows():
            # Extract keywords from question as query
            question_words = row['question'].split()
            # Select 3-5 words as keyword query
            num_keywords = min(5, max(3, len(question_words) // 2))
            keywords = ' '.join(question_words[:num_keywords])

            sample_queries.append({
                'query': keywords,
                'full_question': row['question'],
                'expected_answer': row['answer'],
                'original_pair_id': row['pair_id']
            })

        # Save sample queries
        os.makedirs('processed_data', exist_ok=True)
        sample_df = pd.DataFrame(sample_queries)
        sample_df.to_csv('processed_data/sample_queries.csv', index=False)
        print(f"✅ Sample queries saved: processed_data/sample_queries.csv")

        return sample_queries


def main():
    """Main function with enhanced progress tracking"""
    print("🚀 QUORA DATASET PREPARATION TOOL")
    print("=" * 60)

    # Check environment
    print("🔍 Environment Check:")
    if not os.path.exists('data'):
        print("❌ 'data' directory not found!")
        print("   Please create a 'data' directory and place your files there.")
        return

    if not os.path.exists('data/train.csv'):
        print("❌ 'data/train.csv' not found!")
        print("   Please place your train.csv file in the 'data' directory.")
        return

    print("✅ Environment check passed")

    # Initialize preparator
    preparator = QuoraDataPreparator(data_dir='data')

    try:
        start_time = time.time()

        # Execute workflow
        preparator.load_data()  # Remove sample_size to load all data
        preparator.clean_data()
        preparator.create_qa_pairs()
        preparator.split_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        saved_files = preparator.save_processed_data()
        preparator.create_sample_queries(num_samples=100)

        total_time = time.time() - start_time

        # Final summary
        print(f"\n" + "🎉" * 60)
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print("🎉" * 60)

        print(f"\n⏱️  Total processing time: {total_time / 60:.1f} minutes")

        print(f"\n📁 FILES CREATED:")
        output_files = [
            "processed_data/cleaned_train_data.csv",
            "processed_data/cleaned_test_data.csv",
            "processed_data/qa_pairs.csv",
            "processed_data/train_qa.csv",
            "processed_data/val_qa.csv",
            "processed_data/test_qa.csv",
            "processed_data/test_inference.csv",
            "processed_data/sample_queries.csv",
            "processed_data/data_statistics.txt",
            "data_analysis.png"
        ]

        for file in output_files:
            if os.path.exists(file):
                size_mb = preparator.get_file_size(file)
                print(f"   ✅ {file} ({size_mb:.1f} MB)")

        print(f"\n📊 DATASET SUMMARY:")
        if preparator.cleaned_train_data is not None:
            print(f"   Training data: {len(preparator.cleaned_train_data):,} question pairs")
        if preparator.cleaned_test_data is not None:
            print(f"   Test data: {len(preparator.cleaned_test_data):,} question pairs")
        if preparator.qa_pairs is not None:
            print(f"   QA pairs created: {len(preparator.qa_pairs):,}")
        if preparator.train_data is not None:
            print(f"   Training set: {len(preparator.train_data):,} QA pairs")
            print(f"   Validation set: {len(preparator.val_data):,} QA pairs")
            print(f"   Test set: {len(preparator.test_data):,} QA pairs")

        print(f"\n🚀 Ready for model development!")

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
