import base64
import json
import boto3
from decimal import Decimal
import uuid  

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('ClickEvents')

def lambda_handler(event, context):
    print("Lambda triggered!")

    for record in event['Records']:
        try:
            # Decode base64
            payload = base64.b64decode(record['kinesis']['data']).decode('utf-8')
            print("Decoded payload:", payload)

            # Parse JSON with Decimal support
            data = json.loads(payload, parse_float=Decimal)

            # ✅ Add required partition key
            data['event_id'] = str(uuid.uuid4())

            # Put into DynamoDB
            response = table.put_item(Item=data)
            print("PutItem succeeded:", response)

        except Exception as e:
            print("Error processing record:", e)
