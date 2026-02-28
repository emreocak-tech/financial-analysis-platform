from dotenv import load_dotenv
load_dotenv()
from prophet import Prophet
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
crude_oil_path=os.getenv("CRUDE_OIL")
df=pd.read_csv(crude_oil_path)
plt.style.use("seaborn-v0_8-darkgrid")
from abc import ABC,abstractmethod
class AbstractOil(ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def show_information(self):
        pass
    @abstractmethod
    def show_graph(self):
        pass
    @abstractmethod
    def machine_learning_model(self):
        pass
class CrudeOil(AbstractOil):
    def show_information(self):
        try:
            print(f"The average closing price over the last 30 days {df.tail(30)['price'].mean()} dollars! ")
            print(f"The average maximum price over the last 30 days : {df.tail(30)['price'].max()} dollars!")
            print(f"The average minimum price over the last 30 days : {df.tail(30)['price'].min()} dollars!")
            return [df.tail(30)['price'].mean(), df.tail(30)['price'].max(), df.tail(30)['price'].min()]
        except ValueError as v_error:
            print(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            print(f"Fine was not found try again , Error Code :  {file_error}!")
        except Exception as except_error:
            print(f"There was an generally error try again , Error Code : {except_error}!")
    def show_graph(self):
        try:
            plt.plot(df.tail(30)['date'],df.tail(30)['price'],color="Black",linewidth=3,label="Crude Oil Price")
            plt.xlabel("Date",color="Black",fontsize=18)
            plt.ylabel("Price",color="Black",fontsize=18)
            plt.title("CRUDE OIL PRICE",color="Black",fontsize=20)
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
    def machine_learning_model(self):
        try:
            af = df[['date', 'price']].copy()
            af.columns = ['ds', 'y']
            af = af.dropna()
            af = af[af['y'] > 0]
            af['ds'] = pd.to_datetime(af['ds'])  # 1. ÖNCE çevir
            if af['ds'].dt.tz is not None:  # 2. SONRA kontrol et
                af['ds'] = af['ds'].dt.tz_convert(None)
            af['y'] = np.log(af['y'])
            model = Prophet(changepoint_prior_scale=0.1, yearly_seasonality=True)
            model.fit(af)
            future = model.make_future_dataframe(periods=15)
            predict = model.predict(future)
            tahmin_degeri = predict.iloc[-1]['yhat']
            tahmin_degeri = np.exp(tahmin_degeri)
            return [tahmin_degeri, predict, model]
        except ValueError as v_error:
            print(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            print(f"Fine was not found try again , Error Code :  {file_error}!")
        except Exception as except_error:
            print(f"There was an generally error try again , Error Code : {except_error}!")


