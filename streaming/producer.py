"""
streaming/producer.py — MODULE 5: Kafka Producer
Replays Elliptic transactions in time-step order (1 -> 49), publishing each
as a JSON message to the 'transactions' topic — simulating transactions
arriving live, in the order they actually occurred.
Run:  python -m streaming.producer
Stop: Ctrl+C
"""

import json
import time
import pandas as pd
from kafka import KafkaProducer

TOPIC = "transactions"
SECONDS_PER_STEP = 2  # pause between time steps, so the consumer can keep up visibly

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)


def main():
    print("Loading transaction data...")
    df = pd.read_csv("data/processed/elliptic_merged.csv")
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    steps = sorted(df["time_step"].unique())
    print(f"Loaded {len(df)} transactions across {len(steps)} time steps.\n")

    print(f"Producer started. Publishing to topic '{TOPIC}'.")
    print("Press Ctrl+C to stop.\n")

    try:
        for step in steps:
            batch = df[df["time_step"] == step]
            for _, row in batch.iterrows():
                msg = {
                    "txId": int(row["txId"]),
                    "time_step": int(step),
                    "features": [float(row[c]) for c in feature_cols],
                }
                producer.send(TOPIC, value=msg)
            producer.flush()
            print(f"  Step {step:>2}/{steps[-1]}: sent {len(batch)} transactions")
            time.sleep(SECONDS_PER_STEP)
    except KeyboardInterrupt:
        print("\nProducer stopped.")
    finally:
        producer.close()


if __name__ == "__main__":
    main()
