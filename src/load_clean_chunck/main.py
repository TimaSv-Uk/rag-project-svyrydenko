from .clean_dataset import clean_dataset
from .download_missing_data import download_missing_data
from .download_data import download_data
from .chunk_data import chunk_data
from .dataset_stats import print_dataset_stats

# NOTE: Flow of loading the data.
# saved in data/chunks

if __name__ == "__main__":
    download_data()
    download_missing_data()
    print(f"Number of removed short articles: {clean_dataset("data/raw_american_novelists.jsonl", "data/clean_american_novelists.jsonl")}")
    chunk_data()
    print_dataset_stats()

