# Real-Time Data Pipeline using AWS (Kinesis + Lambda + DynamoDB)

## Architecture

Python Producer → Kinesis Stream → Lambda Function → DynamoDB

## Setup

### 1. Create Resources
Run:
```bash
cd setup
bash deploy.sh
```

### 2. Create Lambda Function
- Runtime: Python 3.11
- Upload `lambda_function.py`
- Add Kinesis trigger: `clickstream`

### 3. Run Producer
```bash
cd producer
python producer.py
```

## Cleanup
```bash
aws kinesis delete-stream --stream-name clickstream --enforce-consumer-deletion
aws dynamodb delete-table --table-name ClickEvents
aws lambda delete-function --function-name kinesis-lambda
```
