import quandl
import datetime as dt

quandl.ApiConfig.api_key='sn3psDvBzGao3UpM6QE5'

#gets closing price of a given stock
def indiv_stock(symbol,start,end):
	stocks=[]
	mydata = quandl.get('EOD/'+str(symbol), start_date=str(start), end_date=str(end),column_index='4')
	stocks.append(mydata)
	return stocks

end_date=dt.date.today()-dt.timedelta(days=1)
before=dt.timedelta(days=1825)
start_date=end_date-before
stock_price=indiv_stock("AAPL",start_date,end_date)
print(stock_price)
n=len(stock_price)
print(stock_price[n-3:n-1])
