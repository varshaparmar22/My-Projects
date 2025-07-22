#!/bin/bash

echo "Creating Kinesis Stream..."
aws kinesis create-stream --stream-name clickstream --shard-count 1

echo "Creating DynamoDB Table..."
aws dynamodb create-table \
    --table-name ClickEvents \
    --attribute-definitions AttributeName=event_id,AttributeType=S \
    --key-schema AttributeName=event_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST

echo "Reminder: Manually create a Lambda function and upload lambda_function.py"
