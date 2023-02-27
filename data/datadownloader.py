from polygon import RESTClient
from local_settings import polygon as settings

#from datetime import date
from datetime import date, datetime
from typing import Any, Optional
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from RestClient import MyRESTClient 

start = datetime(2022,1,1)
client = MyRESTClient(settings['api_key'])
df = client.get_bars(market='crypto', ticker='X:BTCUSD', from_=start)
print(df)

file = open('btcusdt.txt', 'w')
file.write(df)
file.close()