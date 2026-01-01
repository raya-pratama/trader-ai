import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# 1. Konfigurasi Halaman
st.set_page_config(page_title="AI Trader Sentinel", layout="wide")
st.title("🤖 AI Trader Sentinel")

# 2. Pengelompokan Aset
kategori_aset = {
    "Kripto 🪙": {
        "BTCUSD (Bitcoin)": "BTC-USD",
        "ETHUSD (Ethereum)": "ETH-USD",
        "SOLUSD (Solana)": "SOL-USD",
        "DOGEUSD (Dogecoin)": "DOGE-USD"
    },
    "Saham & Indeks 📈": {
        "AAPL (Apple)": "AAPL",
        "TSLA (Tesla)": "TSLA",
        "NVDA (NVIDIA)": "NVDA",
        "S&P 500": "^GSPC",
        "IHSG (Indonesia)": "^JKSE"
    },
    "Komoditas & Forex 🌍": {
        "XAUUSD (Emas)": "GC=F",
        "XAGUSD (Perak)": "SI=F",
        "USDIDR (Dolar-Rupiah)": "IDR=X",
        "EURUSD (Euro-Dolar)": "EURUSD=X",
        "Crude Oil (Minyak)": "CL=F"
    }
}

# 3. Sidebar dengan Pengelompokan
st.sidebar.header("Konfigurasi")

# Pilih Kategori dulu
pilih_kat = st.sidebar.selectbox("Pilih Kategori:", list(kategori_aset.keys()))

# Pilih Aset berdasarkan kategori yang dipilih
daftar_pilihan = kategori_aset[pilih_kat]
pilihan_user = st.sidebar.selectbox("Pilih Aset Spesifik:", list(daftar_pilihan.keys()))
simbol_yahoo = daftar_pilihan[pilihan_user]

tahun = st.sidebar.slider("Data Historis (Tahun):", 1, 10, 3)

# 4. Ambil Data & Tambah Indikator RSI (Agar AI lebih pintar)
data = yf.download(simbol_yahoo, period=f"{tahun}y", interval='1d')
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# Hitung RSI (Relative Strength Index)
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# 5. Logika AI
data['Besok'] = data['Close'].shift(-1)
data['Target'] = (data['Besok'] > data['Close']).astype(int)
data = data.dropna()

features = ['Close', 'Open', 'High', 'Low', 'Volume', 'RSI']
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
model.fit(data[features], data['Target'])

# 6. Tampilan Dashboard
st.markdown(f"### Menganalisis: **{pilihan_user}**")

col1, col2 = st.columns([2, 1])

with col1:
    # Grafik Candlestick
    fig = go.Figure(data=[go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'], name="Harga"
    )])
    
    # Tambah garis RSI di bawah (Opsional, tapi keren)
    fig.update_layout(xaxis_rangeslider_visible=False, height=600, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Statistik & Sinyal")
    
    harga_akhir = data['Close'].iloc[-1]
    perubahan = harga_akhir - data['Close'].iloc[-2]
    persen = (perubahan / data['Close'].iloc[-2]) * 100
    
    st.metric(label="Harga Terakhir", value=f"{harga_akhir:,.2f}", delta=f"{persen:.2f}%")
    
    st.write(f"**RSI (14):** {data['RSI'].iloc[-1]:.2f}")
    if data['RSI'].iloc[-1] > 70: st.warning("Kondisi: Overbought (Jenuh Beli)")
    elif data['RSI'].iloc[-1] < 30: st.info("Kondisi: Oversold (Jenuh Jual)")
    
    st.divider()
    
    # Prediksi AI
    df_akhir = data[features].iloc[-1:]
    pred = model.predict(df_akhir)
    prob = model.predict_proba(df_akhir)
    
    st.write("**Prediksi Arah Besok:**")
    if pred[0] == 1:
        st.success(f"NAIK 🚀 ({max(prob[0])*100:.1f}%)")
    else:
        st.error(f"TURUN 📉 ({max(prob[0])*100:.1f}%)")

st.caption("Data diperbarui otomatis dari Yahoo Finance. Prediksi dihitung secara real-time.")
