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
from dotenv import load_dotenv
load_dotenv()
from abc import ABC,abstractmethod
silver_path=os.getenv("SILVER_PATH")
df=pd.read_csv(silver_path)
"""
specific_date = '2026-01-13'
filtered_data = df[df['Date'] == specific_date]['Close']
print(filtered_data)
"""
class AbstractGold(ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def price_silver(self):
        pass
    @abstractmethod
    def show_graph(self):
        pass
    def model_silver(self):
        pass
class Silver(AbstractGold):
    def price_silver(self):
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
    def show_graph(self):
        try:
            plt.plot(df.tail(30)['Date'],df.tail(30)['Close'],color="Black",label="Silver Price")
            plt.xlabel("Date",color="Black",fontsize=15)
            plt.ylabel("Price",color="Black",fontsize=15)
            plt.title("SILVER PRICE GRAPH for LAST 30 DAYS",color="Black",fontsize=20)
            plt.grid()
            plt.legend()
            plt.show()
        except ValueError as v_error:
            print(f"There was an value error : Error Code : {v_error}!")
        except TypeError as type_error:
            print(f"Datas are not fit for drawing graph , Error Code : {type_error}")
        except FileNotFoundError as file_error:
            print(f"Fine was not found try again , Error Code :  {file_error}!")
        except Exception as except_error:
            print(f"There was an generally error try again , Error Code : {except_error}!")
    def calculate_rsi(self):
        try:
            prices = df['Close'].values
            income = []
            loss = []
            for i in range(1, len(prices)):
                price = prices[i] - prices[i - 1]
                if price > 0:
                    income.append(price)
                if price < 0:
                    loss.append(price)
            income_list = np.array(income).astype(float).tolist()
            income_total = 0

            loss_list = np.array(loss).astype(float).tolist()
            loss_total = 0

            for i in income_list:
                income_total += i
            for i in loss_list:
                loss_total += i

            average_income = income_total / 14
            average_loss = abs(loss_total) / 14

            rs = average_income / average_loss

            rsi_value = 100 - (100 / (1 + rs))

            print(f"The RSI value of silver is {rsi_value}")
        except ValueError as v_error:
            print(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            print(f"Fine was not found try again , Error Code :  {file_error}!")
        except Exception as except_error:
            print(f"There was an generally error try again , Error Code : {except_error}!")

qwe=Silver()
qwe.calculate_rsi()

















































