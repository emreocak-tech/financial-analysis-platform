import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader,TensorDataset
from dotenv import load_dotenv
load_dotenv()
from abc import ABC,abstractmethod
silver_path=os.getenv("SILVER_PATH")
df=pd.read_csv(silver_path)
oil_path=os.getenv("CRUDE_OIL")
oil_df=pd.read_csv(oil_path)
plt.style.use("seaborn-v0_8-darkgrid")
"""
specific_date = '2026-01-13'
filtered_data = df[df['Date'] == specific_date]['Close']
print(filtered_data)
"""

class AbstractSilver(ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def price_silver(self):
        pass
    @abstractmethod
    def show_graph(self):
        pass
    @abstractmethod
    def model_silver(self):
        pass
class Silver(AbstractSilver):
    def price_silver(self):
        try:
            print(f"The average closing price over the last 30 days {df.tail(30)['Close'].mean()} dollars! ")
            print(f"The average maximum price over the last 30 days : {df.tail(30)['Close'].max()} dollars!")
            print(f"The average minimum price over the last 30 days : {df.tail(30)['Close'].min()} dollars!")
            return [df.tail(30)['Close'].mean(),df.tail(30)['Close'].max(),df.tail(30)['Close'].min()]
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
            return plt
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

            return rsi_value
        except ValueError as v_error:
            print(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            print(f"Fine was not found try again , Error Code :  {file_error}!")
        except Exception as except_error:
            print(f"There was an generally error try again , Error Code : {except_error}!")
    def compare_oil(self):
        try:
            plt.plot(df.tail(30)['Date'], df.tail(30)['Close'], color="Silver", linewidth=3, label="Silver Price")
            plt.plot(oil_df.tail(30)['date'], oil_df.tail(30)['price'], color="Black", linewidth=3,label="Crude Oil Price")
            plt.xlabel("Date", color="Black", fontsize=18)
            plt.ylabel("Price", color="Black", fontsize=18)
            plt.title("SILVER-CRUDE OIL", color="Black", fontsize=20)
            plt.legend()
            plt.grid()
            plt.show()
            return plt
        except ValueError as v_error:
            print(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            print(f"Fine was not found try again , Error Code :  {file_error}!")
        except Exception as except_error:
            print(f"There was an generally error try again , Error Code : {except_error}!")
    def model_silver(self):
        try:
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
            class Lstm_Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(1, 10, batch_first=True)
                    self.shell1 = nn.Linear(10, 1)
                def forward(self, x):
                    x,_ = self.lstm(x)
                    x = x[:, -1, :]
                    x = self.shell1(x)
                    return x
            model = Lstm_Model()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.MSELoss()
            for epoch in range(1000):
                prediction = model(x_train_tensor)
                loss = criterion(prediction, y_train_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch % 100 == 0:
                    print(f"The loss value is {loss.item()}")
            model.eval()
            with torch.no_grad():
                prediction = model(x_test_tensor)
                loss = criterion(prediction, y_test_tensor)
                print(f"The test loss value is {loss.item()}")
                print("The model was educated successfully !")
            son_3_gun = values[-3:]
            son_3_tensor = torch.FloatTensor(son_3_gun).unsqueeze(0).unsqueeze(-1)
            son_3_normalized = (son_3_tensor - mins_one) / (maxs_one - mins_one)
            with torch.no_grad():
                tahmin_normalized = model(son_3_normalized)
                tahmin = tahmin_normalized * (maxs_two - mins_two) + mins_two
            print(f"Predicted price: {tahmin.item()}")
            return tahmin.item()

        except ValueError as v_error:
            print(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            print(f"Fine was not found try again , Error Code :  {file_error}!")
        except Exception as except_error:
            print(f"There was an generally error try again , Error Code : {except_error}!")






















































