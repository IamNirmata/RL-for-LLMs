# prepare_toy_parquet.py
import os
import pandas as pd

def main():
    os.makedirs("data", exist_ok=True)

    rows = [
        {"prompt": "Return JSON with answer and explanation. What is 2+2?", "ground_truth": "4", "data_source": "toy"},
        {"prompt": "Return JSON with answer and explanation. What is 3+5?", "ground_truth": "8", "data_source": "toy"},
        {"prompt": "Return JSON with answer and explanation. What is 10-7?", "ground_truth": "3", "data_source": "toy"},
    ]

    df = pd.DataFrame(rows)
    df.to_parquet("data/toy_train.parquet", index=False)
    df.to_parquet("data/toy_val.parquet", index=False)

    print("Parquet files written to ./data")

if __name__ == "__main__":
    main()
