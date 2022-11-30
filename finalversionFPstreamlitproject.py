# -*- coding: utf-8 -*-
"Created on Tue Nov 29 19:28:25 2022@author: ngantayat1"

#==============================================================================#
#importing the libraries required
#==============================================================================#


import streamlit as st                                         #web development platform
import yfinance as yf                                          #module to collect financial data from yahoo finance service
import pandas as pd                                            #data manipulation library
from datetime import datetime, timedelta                       #manipulating date and time
import yahoo_fin.stock_info as si                              #module to retrieve some extra information
import numpy as np                                             #data manipulation library
import plotly.express as px                                    # library for visualization
import plotly.graph_objects as go                              # library for visualization
from plotly.subplots import make_subplots                      # library for visualization
import matplotlib.pyplot as plt                                # library for visualization
import cufflinks as cf

#==============================================================================#
#designing Tab1
#==============================================================================#
# --- Table to show data ---

# Add table to show stock data
# This function gets the stock data and save it to cache to resuse

def Tab1():
        
    st.title("Summary")
    st.write("Select ticker on the left to start:")
    st.write(ticker)

    def getsummary(ticker):
        table = si.get_quote_table(ticker, dict_result = False)
        return table 
           
    c1, c2 = st.columns((1,1))
    with c1:        
        summary = getsummary(ticker)
        summary['value'] = summary['value'].astype(str)
        showsummary = summary.iloc[[14, 12, 5, 2, 6, 1, 16, 3],]
        showsummary.set_index('attribute', inplace=True)
        st.dataframe(showsummary)
            
            
    with c2:        
        summary = getsummary(ticker)
        summary['value'] = summary['value'].astype(str)
        showsummary = summary.iloc[[11, 4, 13, 7, 8, 10, 9, 0],]
        showsummary.set_index('attribute', inplace=True)
        st.dataframe(showsummary)
            
    @st.cache 
    def getstockdata(ticker):
        stockdata = yf.download(ticker, period = 'MAX')
        return stockdata
        
    if ticker != '-':
        chartdata = getstockdata(ticker) 
                   
        fig = px.area(chartdata, chartdata.index, chartdata['Close'])
        
                 

        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(label = "MAX", step="all")
                ])
            )
        )
        st.plotly_chart(fig)
        
#==============================================================================#
#designing Tab2
#==============================================================================#
                
def Tab2():
    st.title("Chart")
    st.write(ticker)
    
    st.write("Set duration to '-' to select date range")
    
    c1, c2, c3, c4,c5 = st.columns((1,1,1,1,1))
    
    with c1:
        
        start_date = st.date_input("Start date", datetime.today().date() - timedelta(days=30),key='start_date')
        
    with c2:
        
        end_date = st.date_input("End date", datetime.today().date(),key='end_date')        
        
    with c3:
        
        duration = st.selectbox("Select duration", ['-', '1Mo', '3Mo', '6Mo', 'YTD','1Y', '3Y','5Y', 'MAX'])          
        
    with c4: 
        
        inter = st.selectbox("Select interval", ['1d', '1mo'])
        
    with c5:
        
        plot = st.selectbox("Select Plot", ['Line', 'Candle'])
           
             
    @st.cache             
    def getchartdata(ticker):
        MA = yf.download(ticker, period = 'MAX')
        MA['MA'] = MA['Close'].rolling(50).mean()
        MA = MA.reset_index()
        MA = MA[['Date', 'MA']]
        
        if duration != '-':        
            chartdata1 = yf.download(ticker, period = duration, interval = inter)
            chartdata1 = chartdata1.reset_index()
            chartdata1 = chartdata1.merge(MA, on='Date', how='left')
            return chartdata1
        else:
            chartdata2 = yf.download(ticker, start_date, end_date, interval = inter)
            chartdata2 = chartdata2.reset_index()
            chartdata2 = chartdata2.merge(MA, on='Date', how='left')                             
            return chartdata2
         
        
    if ticker != '-':
            chartdata = getchartdata(ticker) 
            
                       
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            if plot == 'Line':
                fig.add_trace(go.Scatter(x=chartdata['Date'], y=chartdata['Close'], mode='lines', 
                                         name = 'Close'), secondary_y = False)
            else:
                fig.add_trace(go.Candlestick(x = chartdata['Date'], open = chartdata['Open'], 
                                             high = chartdata['High'], low = chartdata['Low'], close = chartdata['Close'], name = 'Candle'))
              
                    
            fig.add_trace(go.Scatter(x=chartdata['Date'], y=chartdata['MA'], mode='lines', name = '50-day MA'), secondary_y = False)
            
            fig.add_trace(go.Bar(x = chartdata['Date'], y = chartdata['Volume'], name = 'Volume'), secondary_y = True)

            fig.update_yaxes(range=[0, chartdata['Volume'].max()*3], showticklabels=False, secondary_y=True)
        
      
            st.plotly_chart(fig)
            
#==============================================================================#
#designing Tab3
#==============================================================================#

def Tab3():
      st.title("Financials")
      st.write(ticker)
      
      statement = st.selectbox("Show", ['Income Statement', 'Balance Sheet', 'Cash Flow'])
      period = st.selectbox("Period", ['Yearly', 'Quarterly'])
      
      @st.cache
      def getyearlyincomestatement(ticker):
            return yf.Ticker(ticker).financials
      
      @st.cache
      def getquarterlyincomestatement(ticker):
            return yf.Ticker(ticker).quarterly_financials
      
      @st.cache
      def getyearlybalancesheet(ticker):
            return yf.Ticker(ticker).balance_sheet
      
      @st.cache
      def getquarterlybalancesheet(ticker):
            return  yf.Ticker(ticker).quarterly_balance_sheet
      @st.cache
      def getyearlycashflow(ticker):
            return yf.Ticker(ticker).cashflow
      
      @st.cache
      def getquarterlycashflow(ticker):
            return  yf.Ticker(ticker).quarterly_cashflow
        
          
      if ticker != '-' and statement == 'Income Statement' and period == 'Yearly':
                data = getyearlyincomestatement(ticker)
                st.table(data)
            
      if ticker != '-' and statement == 'Income Statement' and period == 'Quarterly':
                data = getquarterlyincomestatement(ticker)
                st.table(data)            

      if ticker != '-' and statement == 'Balance Sheet' and period == 'Yearly':
                data = getyearlybalancesheet(ticker)
                st.table(data)            
      
      if ticker != '-' and statement == 'Balance Sheet' and period == 'Quarterly':
                data = getquarterlybalancesheet(ticker)
                st.table(data)        
      
      if ticker != '-' and statement == 'Cash Flow' and period == 'Yearly':
                data = getyearlycashflow(ticker)
                st.table(data)        
      
        
      if ticker != '-' and statement == 'Cash Flow' and period == 'Quarterly':
                data = getquarterlycashflow(ticker)
                st.table(data)      



#==============================================================================#
#designing Tab4
#==============================================================================#
def Tab4():
    
    st.subheader('Tab4 - Monte Carlo Simulation')
    
   
    col1,col2 = st.columns(2)
    simulations = col1.selectbox("Select number of simulations (n)", [200, 500, 1000])
    time_horizon = col2.selectbox("Select a time horizon (t)", [30, 60, 90])
    
    @st.cache
    def montecarlosimulation(ticker, time_horizon, simulations):
        
        end_date = datetime.today().date()
        start_date = end_date - timedelta(days=30)
        stockprice = yf.Ticker(ticker).history(start=start_date, end=end_date)
        close_price = stockprice['Close']
        
        daily_return = stockprice['Close'].pct_change()
        daily_volatility = np.std(daily_return)

        # Run the simulation
        simulation_df = pd.DataFrame()

        for i in range(simulations):
    
            # The list to store the next stock price
            next_price = []
    
            # Create the next stock price
            last_price = stockprice['Close'].iloc[-1]
    
            for j in range(time_horizon):
                # Generate the random percentage change around the mean (0) and std (daily_volatility)
                future_return = np.random.normal(0, daily_volatility)

                # Generate the random future price
                future_price = last_price * (1 + future_return)

                # Save the price and go next
                next_price.append(future_price)
                last_price = future_price
    
            # Store the result of the simulation
            next_price_df = pd.Series(next_price).rename('sim' + str(i))
            simulation_df = pd.concat([simulation_df, next_price_df], axis=1)
        
        return simulation_df
    
    mc = montecarlosimulation(ticker, time_horizon, simulations)
        
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=30)
    stockprice = yf.Ticker(ticker).history(start=start_date, end=end_date)
    close_price = stockprice['Close']
    
    # Plot the simulation stock price in the future
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10, forward=True)
        
    ax.plot(mc)
    plt.title('Monte Carlo simulation for ' + str(ticker) + ' stock price in next ' + str(time_horizon) + ' days')
    plt.xlabel('Day')
    plt.ylabel('Price')
    
    plt.axhline(y= close_price[-1], color ='red')
    plt.legend(['Current stock price is: ' + str(np.round(close_price[-1], 2))])
    ax.get_legend().legendHandles[0].set_color('red')
        
    st.pyplot(fig)

    # Get the ending price of the 200th day
    ending_price = mc.iloc[-1:, :].values[0, ]
    
    fig1, ax = plt.subplots()
    fig1.set_size_inches(15, 10, forward=True)
        
    ax.hist(ending_price, bins=50)
    plt.axvline(np.percentile(ending_price, 5), color='red', linestyle='--', linewidth=1)
    plt.legend(['Current Stock Price is: ' + str(np.round(np.percentile(ending_price, 5), 2))+ ' USD '])
    plt.title('Distribution of the Ending Price')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    st.pyplot(fig1)
        
    # Price at 95% confidence interval
    future_price_95ci = np.percentile(ending_price, 5)
        
    # Value at Risk
    VaR = close_price[-1] - future_price_95ci
    st.write('VaR at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD')
    



#==============================================================================#
#designing Tab5
#==============================================================================#           

def Tab5():
     st.title("Statistics")
     st.write(ticker)
     c1, c2 = st.columns(2)
     
     with c1:
         st.header("Valuation Measures")
         #@st.cache
         def getvaluation(ticker):
                 return si.get_stats_valuation(ticker)
    
         if ticker != '-':
                valuation = getvaluation(ticker)
                valuation[1] = valuation[1].astype(str)
                valuation = valuation.rename(columns = {0: 'Attribute', 1: ''})
                valuation.set_index('Attribute', inplace=True)
                st.table(valuation)
                
        
         st.header("Financial Highlights")
         st.subheader("Fiscal Year")
         
         #@st.cache
         def getstats(ticker):
                 return si.get_stats(ticker)
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[29:31,])
                
        
         st.subheader("Profitability")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[31:33,])
                
                
                
         st.subheader("Management Effectiveness")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[33:35,])
         
         
                
         st.subheader("Income Statement")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[35:43,])  
            
         
         st.subheader("Balance Sheet")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[43:49,])
         
         st.subheader("Cash Flow Statement")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[49:,])
         
        
                           
     with c2:
         st.header("Trading Information")
         
         
         st.subheader("Stock Price History")
                  
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[:7,])
         
         st.subheader("Share Statistics")
                  
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[7:19,])
         
         st.subheader("Dividends & Splits")
                  
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[19:29,])
                
#==============================================================================#
#designing Tab6
#==============================================================================#                
def Tab6():
      st.title("Portfolio's Trend")
      alltickers = si.tickers_sp500()
      selected_tickers = st.multiselect("Select tickers in your portfolio", options = alltickers, default = ['AAPL'])
      
      
      df = pd.DataFrame(columns=selected_tickers)
      for ticker in selected_tickers:
          df[ticker] = yf.download(ticker, period = '5Y')['Close']
                
               
      fig = px.line(df)
      st.plotly_chart(fig) 

# App title
def Tab7():
    st.markdown('''''')
    st.write('---')
    
    
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    ticker_list = np.array(sp500[0]['Symbol'])
    
    # Retrieving tickers data
    #ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
    tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list) # Select ticker symbol
    tickerData = yf.Ticker(tickerSymbol) # Get ticker data
    tickerDf = tickerData.history(period='1d', start=start_date, end=end_date) #get the historical prices for this ticker
    
    # Ticker information
    string_logo = '<img src=%s>' % tickerData.info['logo_url']
    st.markdown(string_logo, unsafe_allow_html=True)
    
    string_name = tickerData.info['longName']
    st.header('**%s**' % string_name)
    
    string_summary = tickerData.info['longBusinessSummary']
    st.info(string_summary)
    
    # Ticker data
    st.header('**Ticker data**')
    st.write(tickerDf)
    
    # Bollinger bands
    st.header('**Bollinger Bands**')
    qf=cf.QuantFig(tickerDf,title='First Quant Figure',legend='top',name='GS')
    qf.add_bollinger_bands()
    fig = qf.iplot(asFigure=True)
    st.plotly_chart(fig)
    
    ####
    #st.write('---')
    #st.write(tickerData.info)
      
def Tab0():
    st.title("Financial dashboard")
    st.write("Data source: Yahoo Finance")

    # --- Multiple choices box ---

    # Get the list of stock tickers from S&P500
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']

    # Add multiple choices box
    global ticker
    ticker = st.sidebar.selectbox("Ticker(s)", ticker_list)

    # --- Select date time ---

    # Add select begin-end date
    global start_date, end_date 
    col1, col2 = st.sidebar.columns(2)  # Create 2 columns
    start_date = col1.date_input("Start date", datetime.today().date() - timedelta(days=30))
    end_date = col2.date_input("End date", datetime.today().date())
    
    tab01, tab02, tab03, tab04, tab05, tab06, tab07 = st.tabs(['Summary', 'Chart','financials' , 'Monte Carlo Simulation','Statistics','Portfolio','Description'])
    # --- Add a button ---
    get = st.sidebar.button("Get data", key="get")
    
    # Show the selected tab
    with tab01:
        # Run tab 1
        Tab1()
    with tab02:
        # Run tab 2
        Tab2()
    with tab03:
        # Run tab 3
        Tab3()
    with tab04:
        # Run tab 4
        Tab4()
    with tab05:
        Tab5()
    with tab06:
        Tab6()
    with tab07:
        Tab7()
        
if __name__ == "__main__":
    Tab0()               
        
                 