import numpy as np
import requests
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from google import genai
from google.genai import types
load_dotenv()
import os
from abc import ABC,abstractmethod
api_key=os.getenv("API_KEY")
google_gemini_key=os.getenv("GOOGLE_GEMİNİ")
turkish_lira=[16.58,23.73,32.51,42.0,44.0]
statics=[0.29,0.34,0.48,0.72,0.66]
statics_tensor=torch.tensor(statics,dtype=torch.float32).view(-1,1)
turkish_lira_tensor=torch.tensor(turkish_lira,dtype=torch.float32).view(-1,1)
data_loader=TensorDataset(turkish_lira_tensor,statics_tensor)
data=DataLoader(data_loader,shuffle=True,batch_size=2)
years=[2022,2023,2024,2025,2026]
url=f"https://v6.exchangerate-api.com/v6/{api_key}/latest/USD"
class MyForexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.shell1=nn.Linear(1,5)
        self.shell2=nn.Linear(5,1)
    def forward(self,x):
        x=self.shell1(x)
        x=self.relu(x)
        x=self.shell2(x)
        x=self.sigmoid(x)
        return x
class AbstractForex(ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def print_price(self):
        pass
    @abstractmethod
    def analyze_price(self):
        pass
    @abstractmethod
    def lineer_model(self):
        pass
class Forex(AbstractForex):
    def print_price(self):
        try:
            response=requests.get(url=url)
            data=response.json()
            print(f"1 dolar = {data['conversion_rates']['TRY']} turkish lira !")
        except ConnectionError as ce_error:
            print(f"Bağlantı hatası: {ce_error}")
        except requests.exceptions.Timeout as time_error:
            print(f"Zaman aşımı: {time_error}")
        except Exception as ex_error:
            print(f"Genel hata: {ex_error}")
    def analyze_price(self):
        try:
            plt.scatter(years,turkish_lira,color="Black",linewidth=3,label="USD/TRY Values")
            plt.title("USD-TRY GRAPH",fontsize=20,color="Black")
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as except_er:
            print(f"It is a comman error value : {except_er}")
    def lineer_model(self):
        model = MyForexModel()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(1000):
            for x_input, y_input in data:
                prediction = model(x_input)
                loss = criterion(prediction, y_input)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch % 100 == 0:
                    print(f"The loss value is {loss.item()}")
        prediction_data = torch.tensor([[47]], dtype=torch.float32)
        result = model.forward(prediction_data)
        result=result.view(-1,1)
        print(f"The probability of the dollar being 47 TL is %{result*100}%.")
class HelpfulUtils:
    def genai_gemini(self):
        try:
            client = genai.Client(api_key=google_gemini_key)
            text = input("Send a message : ")
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                config=types.GenerateContentConfig(
                system_instruction="You are economy professor and you want to explain basically what users sad."),
                contents=text
            )
            print(response.text)
        except ConnectionError as internet_error:
            print(f"YFinance API Error : {internet_error}")
        except ValueError as v_error:
            print(f"Shape of value is not comfort for this project , check again  : {v_error}")
        except Exception as except_error:
            print(f"Exception value : {except_error}")
forex=Forex()
forex.print_price()
forex.analyze_price()
forex.lineer_model()
utils=HelpfulUtils()
utils.genai_gemini()