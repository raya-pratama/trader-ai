import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# 1. Konfigurasi Awal
st.set_page_config(page_title="Trader AI", layout="wide")

# 2. Kamus Aset (Mapping)
# Pastikan nama di kiri adalah yang ingin kamu tampilkan di teks
daftar_aset = {
    "XAUUSD (Emas)": "GC=F",
    "BTCUSD (Bitcoin)": "BTC-USD",
    "ETHUSD (Ethereum)": "ETH-USD",
    "USDIDR (Dolar ke Rupiah)": "IDR=X"
}

# 3. Sidebar
pilihan_user = st.sidebar.selectbox("Pilih Aset:", list(daftar_aset.keys()))
simbol_yahoo = daftar_aset[pilihan_user]
tahun = st.sidebar.slider("Data Historis (Tahun):", 1, 5, 2)

# 4. Ambil Data
data = yf.download(simbol_yahoo, period=f"{tahun}y", interval='1d')
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# 5. Logika AI
data['Besok'] = data['Close'].shift(-1)
data['Target'] = (data['Besok'] > data['Close']).astype(int)
data = data.dropna()
features = ['Close', 'Open', 'High', 'Low', 'Volume']
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
model.fit(data[features], data['Target'])

# 6. Tampilan Utama
st.title(f"🤖 Analisis AI: {pilihan_user}") # Judul besar di atas

col1, col2 = st.columns([2, 1])

with col1:
    # Bagian ini HARUS menampilkan pilihan_user
    st.subheader(f"Pergerakan Harga {pilihan_user}")
    
    fig = go.Figure(data=[go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close']
    )])
    fig.update_layout(xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Sinyal AI")
    
    # Menampilkan Harga Terakhir dengan Nama Dinamis
    harga_akhir = data['Close'].iloc[-1]
    st.metric(label=f"Harga {pilihan_user} Saat Ini", value=f"{harga_akhir:,.2f}")
    
    st.divider()
    
    # Prediksi
    df_akhir = data[features].iloc[-1:]
    pred = model.predict(df_akhir)
    if pred[0] == 1:
        st.success(f"HASIL: {pilihan_user} DIPREDIKSI NAIK 🚀")
    else:
        st.error(f"HASIL: {pilihan_user} DIPREDIKSI TURUN 📉")
