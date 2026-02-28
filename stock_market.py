from dotenv import load_dotenv
load_dotenv()
import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import torch
import datetime
import torch.nn as nn
import torch.optim as optim
from bs4 import BeautifulSoup
from prophet import Prophet
from google import genai
from google.genai import types
from abc import ABC,abstractmethod
plt.style.use("seaborn-v0_8-darkgrid")
start_date=os.getenv("START_DAY")
end_date=os.getenv("END_DAY")
api_key=os.getenv("GOOGLE_GEMİNİ")
class AbstractClass(ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def analyze(self,sembol):
        pass
    @abstractmethod
    def show_graph(self,sembol,sembol2,sembol3):
        pass
    @abstractmethod
    def gemini_integration(self):
        pass
    @abstractmethod
    def your_model_result(self,sembol,num):
        pass
class StockMarket(AbstractClass):
    def analyze(self,sembol):
        try:
            df=yf.download(tickers=sembol,start=start_date,end=end_date,interval="1d")
            print(f"The average closing price over the last 365 days : {df['Close'].mean()}")
            print(f"The average maximum price over the last 365 days : {df['Close'].max()}")
            print(f"The average minimum price over the last 365 days : {df['Close'].min()}")
            return [df['Close'].mean(),df['Close'].max(),df['Close'].min()]
        except ConnectionError as connect_error:
            print(f"Connection Error : {connect_error}")
        except TimeoutError as time_error:
            print(f"Timeout Error : {time_error}")
        except Exception as except_error:
            print(f"Exception value : {except_error}")
    def show_graph(self,sembol1,sembol2,sembol3):
        try:
            df = yf.download(sembol1, interval="1d", start=start_date, end=datetime.datetime.now())
            df2 = yf.download(sembol2, interval="1d", start=start_date, end=datetime.datetime.now())
            df3 = yf.download(sembol3, interval="1d", start=start_date, end=datetime.datetime.now())
            plt.plot(df.index, df['Close'], linewidth=3, color="Green", label=f"{sembol1}")
            plt.plot(df2.index, df2['Close'], linewidth=3, color="Silver", label=f"{sembol2}")
            plt.plot(df3.index, df3['Close'], linewidth=3, color="Blue", label=f"{sembol3}")
            plt.xlabel("Date", fontsize=15, color="Black")
            plt.ylabel("Price", color="Black", fontsize=15)
            plt.title(f"{sembol1} PRICE vs {sembol2} PRICE vs {sembol3} PRICE", fontsize=20, color="Black")
            plt.legend(fontsize=10)
            plt.grid(True)
            plt.show()
            return plt
        except ConnectionError as connect_error:
            print(f"Connection Error : {connect_error}")
        except TimeoutError as time_error:
            print(f"Timeout Error : {time_error}")
        except Exception as except_error:
            print(f"Exception value : {except_error}")
    def gemini_integration(self):
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                config=types.GenerateContentConfig(
                system_instruction="You are economy professor and you want to explain basically what users sad."),
                contents="hi"
            )
            print(response.text)
        except ConnectionError as connect_error:
            print(f"Connection Error : {connect_error}")
        except TimeoutError as time_error:
            print(f"Timeout Error : {time_error}")
        except Exception as except_error:
            print(f"Exception value : {except_error}")
    def your_model_result(self,sembol,num):
        try:
            df=yf.download(sembol,interval="1d",start=start_date,end=end_date)
            values=df['Close'].values.flatten()
            def create_sequences(values,seq_length=3):
                x=[]
                y=[]
                for i in range(len(values)-seq_length):
                    prices=values[i:i+seq_length]
                    target=values[i+seq_length]
                    x.append(prices)
                    y.append(target)
                return np.array(x),np.array(y)

            x, y = create_sequences(values, seq_length=3)
            total = len(x)
            train_size = int(0.6 * total)  # %60 train
            val_size = int(0.2 * total)  # %20 validation
            test_size = total - train_size - val_size  # %20 test

            x_train = x[:train_size]
            y_train = y[:train_size]

            x_val = x[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]

            x_test = x[train_size + val_size:]
            y_test = y[train_size + val_size:]

            x_train_tensor = torch.FloatTensor(x_train).unsqueeze(-1)  # (N, 3, 1)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)  # (N, 1)

            def normalizition(tensor,maxs,mins):
                return (tensor-mins)/(maxs-mins)
            maxs_one=x_train_tensor.max()
            mins_one=x_train_tensor.min()
            maxs_two=y_train_tensor.max()
            mins_two=y_train_tensor.min()

            x_train_tensor=normalizition(x_train_tensor,maxs_one,mins_one)
            y_train_tensor=normalizition(y_train_tensor,maxs_two,mins_two)

            x_val_tensor = torch.FloatTensor(x_val).unsqueeze(-1)  # (N, 3, 1)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(-1)

            x_test_tensor = torch.FloatTensor(x_test).unsqueeze(-1)
            y_test_tensor = torch.FloatTensor(y_test).unsqueeze(-1)

            class LstmModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm=nn.LSTM(1,10,batch_first=True)
                    self.shell1=nn.Linear(10,1)
                def forward(self,a):
                    a,_=self.lstm(a)
                    a=a[:,-1,:]
                    a=self.shell1(a)
                    return a
            model=LstmModel()
            optimizer=optim.Adam(model.parameters(),lr=0.01)
            criterion=nn.MSELoss()
            for epoch in range(1000):
                prediction=model(x_train_tensor)
                loss=criterion(prediction,y_train_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch%100==0:
                    print(f"The loss value is {loss.item()}")
            model.eval()
            with torch.no_grad():
                prediction=model(x_test_tensor)
                loss=criterion(prediction,y_test_tensor)
                print(f"The test loss value is {loss.item()}")
            print("The model was educated successfully !")
            son_3_gun = values[-3:]
            son_3_tensor = torch.FloatTensor(son_3_gun).unsqueeze(0).unsqueeze(-1)
            son_3_normalized = (son_3_tensor - mins_one) / (maxs_one - mins_one)
            tahmin_normalized = model(son_3_normalized)
            tahmin = tahmin_normalized * (maxs_two - mins_two) + mins_two
            return tahmin.item()
        except ConnectionError as internet_error:
            print(f"YFinance API Error : {internet_error}")
        except ValueError as v_error:
            print(f"Shape of value is not comfort for this project , check again  : {v_error}")
        except Exception as except_error:
            print(f"Exception value : {except_error}")
class AdvancedPrediction:
    def make_a_prediction(self,sembol):
        try:
            df = yf.download(sembol, interval="1d", start="2026-2-1", end=datetime.datetime.now())
            df = df[['Close']].reset_index()
            df.columns = ['ds', 'y']
            df = df.dropna()
            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=360)
            predict = model.predict(future)
            tahmin_degeri = predict.iloc[-5]['yhat']
            return [tahmin_degeri, predict, model]
        except ConnectionError as internet_error:
            print(f"YFinance API Error : {internet_error}")
        except ValueError as v_error:
            print(f"Shape of value is not comfort for this project , check again  : {v_error}")
        except Exception as except_error:
            print(f"Exception value : {except_error}")

