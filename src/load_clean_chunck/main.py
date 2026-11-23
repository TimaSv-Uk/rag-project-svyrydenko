from .clean_dataset import clean_dataset
from .load_missing import load_missing
from .load_data import load_data
from .chunk_data import chunk_data
from .dataset_stats import print_dataset_stats

# NOTE: Flow of loading the data.
# saved in data/chunks

if __name__ == "__main__":
    load_data()
    load_missing()
    print(f"Number of removed short articles: {clean_dataset("data/raw_american_novelists.jsonl", "data/clean_american_novelists.jsonl")}")
    chunk_data()
    print_dataset_stats()

