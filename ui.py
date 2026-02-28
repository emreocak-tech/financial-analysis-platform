import requests
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
load_dotenv()
from exchange_rate import HelpfulUtils
from exchange_rate import Forex
from stock_market import StockMarket
from stock_market import AdvancedPrediction
from gold import Gold
from silver import Silver
from crude_oil import CrudeOil
st.title("**Kullanıcı Sözleşmesi**")
st.write("UYARI: **Lütfen Okuyunuz !** Bu uygulama yalnızca bilgilendirme amaçlıdır ve yatırım tavsiyesi içermez.- Finansal piyasalarda işlem yapmak yüksek risk içerir ve sermaye kaybına yol açabilir.- Sağlanan tahminler ve analizler kesin sonuç vermez, yanılabilir.- Alacağınız tüm yatırım kararlarının sorumluluğu tamamen size aittir.- Uygulama geliştiricisi, kullanımdan kaynaklanan herhangi bir zarardan sorumlu değildir.- Yatırım kararı vermeden önce profesyonel bir finansal danışmana başvurmanız önerilir.")
check_box=st.checkbox("Okudum,Kabul Ediyorum")
if check_box:
    asd1,asd2,asd3=st.sidebar.tabs(["Hakkında","Bilgiler","Gemini 2.5-flash"])
    with asd1:
        st.write("FinanScribe; hisse senedi, emtia (altın, gümüş, petrol) ve döviz verilerini işleyen, nesne yönelimli (OOP) mimariyle tasarlanmış modüler bir finansal analiz ve tahmin iskeletidir. Proje, ham veriyi anlamlı içgörülere dönüştürmek için istatistiksel modelleri ve derin öğrenme algoritmalarını birleştirir.\nÖne Çıkan Teknik Özellikler\nGelişmiş Mimari: Abstract Base Class (ABC) kullanımıyla SOLID prensiplerine uygun, ölçeklenebilir ve genişletilebilir sınıf yapıları.\nHibrit Tahminleme: * Derin Öğrenme: PyTorch ile özelleştirilmiş LSTM (Long Short-Term Memory) ağları.Zaman Serisi Analizi: Trend ve mevsimsellik odaklı Prophet modelleri.\nAkıllı Analiz: Teknik göstergelerin (RSI vb.) Google Gemini AI entegrasyonu ile doğal dil işleme (NLP) üzerinden yorumlanması.\nVeri Mühendisliği:Normalizasyon ve yfinance, BeautifulSoup gibi araçlarla çok kaynaklı veri yönetimi.\nTeknolojik Stack\nDil: PythonML/AI: PyTorch, Prophet, Scikit-learn, Gemini APIVeri/Görselleştirme: Pandas, NumPy, Matplotlib, Streamlit")
    with asd2:
        st.write("Teknik Stack & Özellikler:\nMimari: Sürdürülebilir ve genişletilebilir OOP iskeleti (Abstract Base Classes & Inheritance).\nTahminleme: PyTorch ile LSTM derin öğrenme modelleri ve Facebook Prophet ile zaman serisi analizi.\nVeri Yönetimi: yfinance API, BeautifulSoup web scraping ve yerel CSV entegrasyonu.\nAnalitik: RSI indikatörü hesaplama ve trend projeksiyonları.\nYZ Entegrasyonu: Google Gemini API üzerinden teknik çıktıların finansal yorumlanması (NLP).")
    with asd3:
        try:
            utils=HelpfulUtils()
            text=st.text_input("Gemini'ya ekonomi hakkında soru sor")
            button=st.button("Gemini",use_container_width=True)
            if button:
                response=utils.genai_gemini(text)
                st.write(response)
        except ConnectionError as c_error:
            st.error(f"Connection Error : {c_error}")
        except TimeoutError as time_error:
            st.error(f"Timeout error : {time_error}")
        except Exception as except_value:
            st.error(f"General Error : {except_value} , Try Again!")

    st.title("Welcome My Website!")
    #tab1,tab2,tab3,tab4,tab5=st.sidebar.tabs(["Foreign Currency💸","Stock Market📈","Gold🥇","Silver🥈","Crude Oil⛽"])
    #with tab1:
     #   st.header("Welcome to Currency page")
    tab1,tab2,tab3,tab4,tab5=st.tabs(["Foreign Currency💸","Stock Market📈","Gold🥇","Silver🥈","Crude Oil⛽"])
    with tab1:
        st.title("Welcome to Currency Page")
        st.header("Show Price of Foreign Currency")
        forex=Forex()
        st.info("USD/TRY Price")
        try:
            button=st.button("USD/TRY",use_container_width=True)
            currency=forex.print_price()
            if button:
                st.write(f"1 USD = {currency} TRY")
        except ConnectionError as c_error:
            st.error(f"Connection Error : {c_error}")
        except TimeoutError as time_error:
            st.error(f"Time out error : {time_error}")
        except Exception as except_error:
            st.error(f"General Error , Please try agin : {except_error}")
        st.info("Graph of Turkish Lira")
        try:
            buton=st.button("Show Graph",use_container_width=True)
            if buton:
                graph = forex.analyze_price()
                st.pyplot(graph)
        except Exception as except_error:
            st.error(f"General Error , Please Try Again : {except_error}")
        st.info("Machine-learning concept")
        try:
            buton=st.button("Modeli Çalıştır🤖",use_container_width=True)
            if buton:
                with st.spinner("The model is being educated at the moment!") as spin:
                    st.write(spin)
                    model = forex.lineer_model()
                    model = str(model * 100)
                    st.write(f"The probability of the dollar being 47 TL is %{model[0:4]}")
        except Exception as except_error:
            st.error(f"General Error , Please Try Again : {except_error}")
    with tab2:
        st.header("Welcome Stock Market Page")
        stock_market=StockMarket()
        companies=["AAPL","NVDA","ASELS.IS","THYAO.IS","IBM","META"]
        decision=st.selectbox("Select companies",options=companies,index=0)
        st.info(f"Show information about price of {decision}")
        try:
            button=st.button("Show information",use_container_width=True)
            if button:
                info=stock_market.analyze(decision)
                st.write(f"The average closing price over the last 365 days : {info[0]}")
                st.write(f"The average maximum price over the last 365 days : {info[1]}")
                st.write(f"The average minimum price over the last 365 days : {info[2]}")
        except ConnectionError as c_error:
            st.error(f"Connection Error : {c_error}")
        except TimeoutError as time_error:
            st.error(f"Time out error : {time_error}")
        except Exception as except_error:
            st.error(f"General Error : {except_error} , Please Try Again Later!")
        st.info("Show graph")
        try:
            multi_select = st.multiselect("Select Companies", options=companies)
            button=st.button("Show graph",use_container_width=True)
            if button:
                graph=stock_market.show_graph(multi_select[0],multi_select[1],multi_select[2])
                st.pyplot(graph)
        except ConnectionError as c_error:
            st.error(f"Connection Error : {c_error}")
        except TimeoutError as time_error:
            st.error(f"Time out error : {time_error}")
        except Exception as except_error:
            st.error(f"General Error : {except_error} , Please Try Again Later!")
        st.info("Prophet Model")
        try:
            decision2 = st.selectbox("Select company", options=companies)
            buton=st.button("Prophet Model",use_container_width=True)
            if buton:
                advanced_model=AdvancedPrediction()
                result=advanced_model.make_a_prediction(decision2)
                result=str(result[0])
                st.write(f"Our predict is  {result[0:5]}")
        except ConnectionError as c_error:
            st.error(f"Connection Error : {c_error}")
        except TimeoutError as time_error:
            st.error(f"Time out error : {time_error}")
        except ValueError as v_error:
            st.error(f"The data is not comfort for our model , Check Again : {v_error}")
        except Exception as except_error:
            st.error(f"General Error : {except_error} , Please Try Again Later!")
        try:
            st.info("LSTM-Model")
            sembol=st.selectbox("Select Company",options=companies,index=0)
            slider=st.slider("Select a value",min_value=1,max_value=10,value=3)
            buton=st.button("Modeli Çalıştır",use_container_width=True)
            if buton:
                with st.spinner(f"LSTM model training for {sembol}... This may take a minute (1000 epochs)...") as spin:
                    st.write(spin)
                    result = stock_market.your_model_result(sembol, slider)
                    result=str(result)
                    st.write(f"Our predict is {result[0:4]} for  {sembol}!")
        except ConnectionError as c_error:
            st.error(f"Connection Error : {c_error}")
        except TimeoutError as time_error:
            st.error(f"Time out error : {time_error}")
        except ValueError as v_error:
            st.error(f"The data is not comfort for our model , Check Again : {v_error}")
        except Exception as except_error:
            st.error(f"General Error : {except_error} , Please Try Again Later!")

    with tab3:
        gold=Gold()
        st.title("Welcome to Gold Page!")
        st.info("İnformation about price of gold")
        try:
            buton=st.button("Gold",use_container_width=True)
            if buton:
                result=gold.show_price()
                st.write(f"The average closing price over the last 30 days : {result[0]}")
                st.write(f"The average maximum price over the last 30 days : {result[1]}")
                st.write(f"The average minimum price over the last 30 days : {result[2]}")
        except ValueError as v_error:
            st.error(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            st.error(f"File was not found try again , Error Code :  {file_error}!")
        except Exception as except_error:
            st.error(f"General Error : {except_error} , Please try again later!")
        st.info("Gold Graph")
        try:
            buton=st.button("Gold Graph",use_container_width=True)
            if  buton:
                result=gold.draw_price()
                st.pyplot(result)
        except ValueError as v_error:
            st.error(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            st.error(f"File was not found try again , Error Code :  {file_error}!")
        except Exception as except_error:
            st.error(f"General Error : {except_error} , Please try again later!")
        st.info("RSI value")
        try:
            buton=st.button("Calculate RSI index for Gold Price",use_container_width=True)
            if buton:
                result=gold.calculate_rsi()
                result=str(result)
                st.write(f"RSI value is {result[0:4]} for gold")
        except ValueError as v_error:
            st.error(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            st.error(f"File was not found try again , Error Code :  {file_error}!")
        except Exception as except_error:
            st.error(f"General Error : {except_error} , Please try again later!")
        st.info("GOLD VS SILVER")
        try:
            buton=st.button("Compare Gold and Silver",use_container_width=True)
            if buton:
                graph=gold.compare_silver()
                st.pyplot(graph)
        except ValueError as v_error:
            st.error(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            st.error(f"File was not found try again , Error Code :  {file_error}!")
        except Exception as except_error:
            st.error(f"General Error : {except_error} , Please try again later!")
        try:
            st.info("Lineer Model")
            buton=st.button("Lineer Model Predict",use_container_width=True)
            if buton:
                with st.spinner("The model is being trained at the moment") as spin:
                    st.write(spin)
                    result=gold.machine_learning_model()
                    result=str(result)
                    st.write(f"Our predict is {result[0:4]} for Gold!")
        except ValueError as v_error:
            st.error(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            st.error(f"File was not found try again , Error Code :  {file_error}!")
        except Exception as except_error:
            st.error(f"General Error : {except_error} , Please try again later!")
    with tab4:
        st.title("Welcome to Silver Page")
        silver=Silver()
        try:
            st.info("İnformation about price of silver")
            buton=st.button("Silver",use_container_width=True)
            if buton:
                result=silver.price_silver()
                st.write(f"The average closing price over the last 30 days : {result[0]}")
                st.write(f"The average maximum price over the last 30 days : {result[1]}")
                st.write(f"The average minimum price over the last 30 days : {result[2]}")
        except ValueError as v_error:
            st.error(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            st.error(f"File was not found try again , Error Code :  {file_error}!")
        except Exception as except_error:
            st.error(f"General Error : {except_error} , Please try again later!")
        try:
            st.info("Silver Graph")
            buton=st.button("Siver Graph",use_container_width=True)
            if buton:
                result=silver.show_graph()
                st.pyplot(result)
        except ValueError as v_error:
            st.error(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            st.error(f"File was not found try again , Error Code :  {file_error}!")
        except TypeError as type_error:
            st.error(f"Data is not comfort for Matplotlib : {type_error}")
        except Exception as except_error:
            st.error(f"General Error : {except_error} , Please try again later!")
        try:
            st.info("RSI index for silver")
            buton=st.button("Calculate RSI",use_container_width=True)
            if buton:
                rsi=silver.calculate_rsi()
                st.write(f"RSI index is {rsi} for silver's price")
        except ValueError as v_error:
            st.error(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            st.error(f"File was not found try again , Error Code :  {file_error}!")
        except TypeError as type_error:
            st.error(f"Data is not comfort for Matplotlib : {type_error}")
        except Exception as except_error:
            st.error(f"General Error : {except_error} , Please try again later!")
        try:
            st.info("Compare Silver and Crude Oil")
            buton=st.button("Compare Silver and Crude Oil",use_container_width=True)
            if buton:
                plt=silver.compare_oil()
                st.pyplot(plt)
        except ValueError as v_error:
            st.error(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            st.error(f"File was not found try again , Error Code :  {file_error}!")
        except TypeError as type_error:
            st.error(f"Data is not comfort for Matplotlib : {type_error}")
        except Exception as except_error:
            st.error(f"General Error : {except_error} , Please try again later!")
        try:
            st.info("LSTM model for Silver")
            buton=st.button("LSTM Model",use_container_width=True)
            if buton:
                with st.spinner("The model is being trained at the moment") as spin:
                    st.write(spin)
                    result=silver.model_silver()
                    result=str(result)
                    st.write(f"Our predict is {result[0:5]} for Silver")
        except ValueError as v_error:
            st.error(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            st.error(f"File was not found try again , Error Code :  {file_error}!")
        except TypeError as type_error:
            st.error(f"Data is not comfort for Matplotlib : {type_error}")
        except Exception as except_error:
            st.error(f"General Error : {except_error} , Please try again later!")
    with tab5:
        st.title("Welcome to Crude Oil Page")
        crude_oil=CrudeOil()
        try:
            st.info("İnformation About Crude Oil's Price")
            buton=st.button("Crude Oil",use_container_width=True)
            if buton:
                result=crude_oil.show_information()
                st.write(f"The average closing price over the last 30 days : {result[0]}")
                st.write(f"The average maximum price over the last 30 days : {result[1]}")
                st.write(f"The average minimum price over the last 30 days : {result[2]}")
        except ValueError as v_error:
            st.error(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            st.error(f"File was not found try again , Error Code :  {file_error}!")
        except TypeError as type_error:
            st.error(f"Data is not comfort for Matplotlib : {type_error}")
        except Exception as except_error:
            st.error(f"General Error : {except_error} , Please try again later!")
        try:
            st.info("Graph of Crude Oil")
            buton=st.button("Crude Oil Graph",use_container_width=True)
            if buton:
                plt=crude_oil.show_graph()
                st.pyplot(plt)
        except ValueError as v_error:
            st.error(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            st.error(f"File was not found try again , Error Code :  {file_error}!")
        except TypeError as type_error:
            st.error(f"Data is not comfort for Matplotlib : {type_error}")
        except Exception as except_error:
            st.error(f"General Error : {except_error} , Please try again later!")
        try:
            st.info("Prophet Model")
            buton=st.button("Prediction with Prophet",use_container_width=True)
            if buton:
                result=crude_oil.machine_learning_model()
                st.write(f"Our prediction is {result[0]}")
        except ValueError as v_error:
            st.error(f"There was an value error : Error Code : {v_error}!")
        except FileNotFoundError as file_error:
            st.error(f"File was not found try again , Error Code :  {file_error}!")
        except TypeError as type_error:
            st.error(f"Data is not comfort for Matplotlib : {type_error}")
        except Exception as except_error:
            st.error(f"General Error : {except_error} , Please try again later!")














else:
    st.info("You have to accept contract!")