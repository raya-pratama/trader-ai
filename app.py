import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Trader AI", layout="wide")
st.title("🤖 Trader Sentinel AI")

# --- 1. Sidebar & Pilihan Aset ---
daftar_aset = {
    "XAUUSD (Emas)": "GC=F",
    "BTCUSD (Bitcoin)": "BTC-USD",
    "ETHUSD (Ethereum)": "ETH-USD",
    "USDIDR (Dolar ke Rupiah)": "IDR=X"
}

pilihan_nama = st.sidebar.selectbox("Pilih Aset:", list(daftar_aset.keys()))
symbol = daftar_aset[pilihan_nama]
period = st.sidebar.slider("Data Historis (Tahun):", 1, 5, 2)

# --- 2. Ambil Data ---
data = yf.download(symbol, period=f"{period}y", interval='1d')

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# --- 3. Logika AI ---
data['Besok'] = data['Close'].shift(-1)
data['Target'] = (data['Besok'] > data['Close']).astype(int)
data = data.dropna()

features = ['Close', 'Open', 'High', 'Low', 'Volume']
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
model.fit(data[features], data['Target'])

# --- 4. Tampilan Dashboard ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Grafik Harga {pilihan_nama}") 
    
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'], 
        high=data['High'],
        low=data['Low'], 
        close=data['Close']
    )])
    
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Hasil Analisis AI")
    
    # Harga Terakhir
    harga_skrg = data['Close'].iloc[-1]
    st.metric(label="Harga Terakhir", value=f"{harga_skrg:,.2f}")
    
    st.divider()
    
    # Prediksi
    data_terakhir = data[features].iloc[-1:]
    prediksi = model.predict(data_terakhir)
    prob = model.predict_proba(data_terakhir)
    
    if prediksi[0] == 1:
        st.success("PREDIKSI BESOK: NAIK 🚀")
    else:
        st.error("PREDIKSI BESOK: TURUN 📉")
        
    st.write(f"Keyakinan Model: {max(prob[0])*100:.2f}%")
    st.info("Berdasarkan pola teknikal harian.")
