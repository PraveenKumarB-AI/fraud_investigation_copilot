"""
streaming/consumer.py — MODULE 5: Kafka Consumer + live scoring
Listens on the 'transactions' topic and scores each transaction with the
saved XGBoost model (Module 4) the instant it arrives — a live simulation
of a real-time fraud-scoring pipeline. Runs independently of the producer.
Run:  python -m streaming.consumer
Stop: Ctrl+C
"""

import json
import pickle
import numpy as np
from kafka import KafkaConsumer

TOPIC = "transactions"
THRESHOLD = 0.5

with open("models/checkpoints/xgb_baseline.pkl", "rb") as f:
    bundle = pickle.load(f)
model = bundle["model"]

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    group_id="fraud-scorer",
)


def main():
    print(f"Consumer started. Scoring transactions from '{TOPIC}' with XGBoost.")
    print("Press Ctrl+C to stop.\n")

    flagged, total = 0, 0
    try:
        for message in consumer:
            data = message.value
            x = np.array(data["features"]).reshape(1, -1)
            prob = model.predict_proba(x)[0, 1]
            total += 1

            if prob > THRESHOLD:
                flagged += 1
                print(f"  FLAGGED  txId={data['txId']}  step={data['time_step']}  "
                      f"fraud_prob={prob:.3f}")
            elif total % 500 == 0:
                print(f"  ... {total} scored so far, {flagged} flagged")
    except KeyboardInterrupt:
        print(f"\nConsumer stopped. Scored {total} transactions, flagged {flagged} "
              f"({flagged/total*100:.1f}%).")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
