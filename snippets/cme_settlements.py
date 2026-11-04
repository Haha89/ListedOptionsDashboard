""" Snippet kept for later: CME futures settlements for one product and trade date, as JSON.
Not wired into the app and not tested. Check CME's terms of use before relying on this endpoint. """
import json

import requests as r

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36"
header = {'User-Agent': user_agent}
link = 'https://www.cmegroup.com/CmeWS/mvc/Settlements/Futures/Settlements/425/FUT?tradeDate=11/05/2021'
page = r.get(link,headers=header)
raw_json = json.loads(page.text)
