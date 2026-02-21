import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import requests
import os
import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()
gold_path=os.getenv("GOLD_PATH")
df=pd.read_csv(gold_path)
silver=os.getenv("SİLVER_PATH")
silver_df=pd.read_csv(silver)
platform_gold=os.getenv("PLATFORM_GOLD_FOR_INVESTING")
from abc import ABC,abstractmethod
class AbstractGold(ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def show_price(self):
        pass
    @abstractmethod
    def draw_price(self):
        pass
    @abstractmethod
    def machine_learning_model(self):
        pass
class Gold(AbstractGold):
    def show_price(self):
        try:
            print(f"The average closing price over the last 30 days {df.tail(30)['Close'].mean()} dollars! ")
            print(f"The average maximum price over the last 30 days : {df.tail(30)['Close'].max()} dollars!")
            print(f"The average minimum price over the last 30 days : {df.tail(30)['Close'].min()} dollars!")
        except ValueError as v_error:
            print(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            print(f"Fine was not found try again , Error Code :  {file_error}!")
        except Exception as except_error:
            print(f"There was an generally error try again , Error Code : {except_error}!")
    def draw_price(self):
        try:
            plt.plot(df.tail(30)['Date'], df.tail(30)['Close'], label="Gold-Price", color="Yellow", linewidth=5)
            plt.xlabel("Time", color="Black", fontsize=15)
            plt.ylabel("Price", color="Black", fontsize=15)
            plt.title("GOLD PRICE", color="Black", fontsize=20)
            plt.legend()
            plt.grid()
            plt.show()
        except ValueError as v_error:
            print(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            print(f"Fine was not found try again , Error Code :  {file_error}!")
        except Exception as except_error:
            print(f"There was an generally error try again , Error Code : {except_error}!")
    def calculate_rsi(self):
        try:
            values = df['Close'].values
            income = []
            loss = []
            for i in range(1, len(values)):
                price = values[i] - values[i - 1]
                if price > 0:
                    income.append(price)
                if price < 0:
                    loss.append(price)
            income_list = np.array(income).astype(float).tolist()
            income_sum = 0
            for i in income_list:
                income_sum += i
            loss_list = np.array(loss).astype(float).tolist()
            loss_sum = 0
            for i in loss_list:
                loss_sum += i

            average_income = income_sum / 14
            average_loss = abs(loss_sum) / 14

            rs = average_income / average_loss

            rsi = 100 - (100 / (1 + rs))

            print(f"The RSI value is {rsi}")
        except ValueError as v_error:
            print(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            print(f"Fine was not found try again , Error Code :  {file_error}!")
        except Exception as except_error:
            print(f"There was an generally error try again , Error Code : {except_error}!")
    def compare_silver(self):
        try:
            plt.plot(silver_df.tail(14)['Date'], silver_df.tail(14)['Close'], color="Silver", linewidth=4,label="Silver Price")
            plt.plot(df.tail(14)['Date'], df.tail(14)['Close'], color="Yellow", linewidth=4, label="Gold Price")
            plt.xlabel("Date", color="Black", fontsize=15)
            plt.ylabel("Prices", color="Black", fontsize=15)
            plt.title("SILVER-GOLD PRICES", color="Black", fontsize=20)
            plt.legend()
            plt.grid()
            plt.show()
        except ValueError as v_error:
            print(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            print(f"Fine was not found try again , Error Code :  {file_error}!")
        except Exception as except_error:
            print(f"There was an generally error try again , Error Code : {except_error}!")
    def now_price(self):
        try:
            response=requests.get(url=platform_gold,timeout=10)
            html_sayfa=BeautifulSoup(response.content,"html.parser")
            price_one=html_sayfa.find("div",class_="text-5xl/9 font-bold text-[#232526] md:text-[42px] md:leading-[60px]")
            price_two=html_sayfa.find("div",class_="text-5xl/9 font-bold text-[#232526] md:text-[42px] md:leading-[60px] bg-negative-light")
            price_three=html_sayfa.find("div",class_="text-5xl/9 font-bold text-[#232526] md:text-[42px] md:leading-[60px] bg-positive-light")
            print(f"The current price of gold is {price_one.text}")
            print(f"The current price of gold is {price_two.text}")
            print(f"The current price of gold is {price_three.text}")
        except ConnectionError as connect_error:
            print(f"There was an connect error , Error Code : {connect_error}")
        except Exception as ex:
            print(f"{ex}")
    def machine_learning_model(self):
        try:
            values = df['Close'].values
            def create_sequences(values, seq_length=60):
                x = []
                y = []
                for i in range(len(values) - seq_length):
                    prices = values[i:i + seq_length]
                    target = values[i + seq_length]
                    x.append(prices)
                    y.append(target)
                return np.array(x), np.array(y)

            x, y = create_sequences(values, seq_length=60)
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

            def normalizition(tensor, maxs, mins):
                return (tensor - mins) / (maxs - mins)

            maxs_one = x_train_tensor.max()
            mins_one = x_train_tensor.min()
            maxs_two = y_train_tensor.max()
            mins_two = y_train_tensor.min()

            x_train_tensor = normalizition(x_train_tensor, maxs_one, mins_one)
            y_train_tensor = normalizition(y_train_tensor, maxs_two, mins_two)


            x_val_tensor = torch.FloatTensor(x_val).unsqueeze(-1)  # (N, 3, 1)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(-1)

            x_test_tensor = torch.FloatTensor(x_test).unsqueeze(-1)
            y_test_tensor = torch.FloatTensor(y_test).unsqueeze(-1)

            x_test_tensor=normalizition(x_test_tensor,maxs_one,mins_one)
            y_test_tensor=normalizition(y_test_tensor,maxs_two,mins_two)
            class PredictGold(nn.Module):
             def __init__(self):
                super().__init__()
                self.shell1 = nn.Linear(1, 10)
                self.shell2 = nn.Linear(10, 1)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
             def forward(self, x):
                x = self.shell1(x)
                x = self.relu(x)
                x = self.shell2(x)
                return x
            model=PredictGold()
            optimizer=optim.Adam(model.parameters(),lr=0.01)
            criterion=nn.MSELoss()
            loss_values=[]
            for epoch in range(1000):
                prediction = model(x_train_tensor)
                loss = criterion(prediction, y_train_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch % 100 == 0:
                    print(f"The loss value is {loss.item()}")
                    loss_values.append(loss.item())
            plt.scatter(loss_values, color="Black")
            model.eval()
            loss_two_list=[]
            with torch.no_grad():
                prediction1=model(x_test_tensor)
                loss2=criterion(prediction1,y_test_tensor)
                print(f"The loss value is {loss2.item()}")
                loss_two_list.append(loss2.item())
            plt.scatter(loss_two_list,color="Black")
            plt.show()
        except FileNotFoundError as file_error:
            print(f"The file was found , Error Code : {file_error}")
        except ValueError as v_error:
            print(f"Shape of value is not comfort for this project , check again  : {v_error}")
        except Exception as except_error:
            print(f"Exception value : {except_error}")





