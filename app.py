import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Trader AI", layout="wide")
st.title("🤖 Trader Sentinel AI")

# --- Sidebar ---
daftar_aset = {
    "XAUUSD (Emas)": "GC=F",
    "BTCUSD (Bitcoin)": "BTC-USD",
    "ETHUSD (Ethereum)": "ETH-USD",
    "USDIDR (Dolar ke Rupiah)": "IDR=X"
}

# --- 2. Tampilkan di Sidebar ---
# User milih nama di layar, tapi variabel 'pilihan' akan berisi simbol aslinya
pilihan_nama = st.sidebar.selectbox("Pilih Aset:", list(daftar_aset.keys()))
symbol = daftar_aset[pilihan_nama]

period = st.sidebar.slider("Data Historis (Tahun):", 1, 5, 2)

# --- 3. Ambil Data (Gunakan variabel 'symbol') ---
data = yf.download(symbol, period=f"{period}y", interval='1d')

# Perbaikan Multi-index
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# --- 2. Logika AI ---
data['Besok'] = data['Close'].shift(-1)
data['Target'] = (data['Besok'] > data['Close']).astype(int)
data = data.dropna()

features = ['Close', 'Open', 'High', 'Low', 'Volume']
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
model.fit(data[features], data['Target'])

# --- 3. Tampilan ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Grafik Harga {symbol}")
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'])])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Hasil Analisis AI")
    data_terakhir = data[features].iloc[-1:]
    prediksi = model.predict(data_terakhir)
    probabilitas = model.predict_proba(data_terakhir)
    
    if prediksi[0] == 1:
        st.success("PREDIKSI BESOK: NAIK 🚀")
    else:
        st.error("PREDIKSI BESOK: TURUN 📉")
        
    st.write(f"Keyakinan Model: {max(probabilitas[0])*100:.2f}%")
    st.info("Catatan: Ini prediksi teknikal, bukan saran finansial.")
