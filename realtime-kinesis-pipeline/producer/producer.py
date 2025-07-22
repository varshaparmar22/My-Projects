import boto3
import json
import time
import random

kinesis = boto3.client('kinesis', region_name='us-east-1')

def generate_event():
    return {
        "user_id": random.randint(1, 1000),
        "action": random.choice(["click", "scroll", "hover"]),
        "timestamp": time.time()
    }

while True:
    event = generate_event()
    print("Sending:", event)
    kinesis.put_record(
        StreamName="clickstream",
        Data=json.dumps(event),
        PartitionKey=str(event['user_id'])
    )
    time.sleep(2)
