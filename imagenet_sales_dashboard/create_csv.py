import pandas as pd
import random
from datetime import datetime, timedelta

categories = ['n02958343 car', 'n03770679 minivan', 'n04285008 sports_car', 'n04461696 tow_truck']
dates = [datetime.today() - timedelta(days=i) for i in range(30)]

data = []
for date in dates:
    for cat in categories:
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'category': cat,
            'sales': random.randint(20, 200)
        })

df = pd.DataFrame(data)
df.to_csv("imagenet_sales.csv", index=False)
