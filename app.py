import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# 1. Konfigurasi Halaman
st.set_page_config(page_title="AI Trader Sentinel", layout="wide")
st.title("AI Trader Sentinel")

# 2. Pengelompokan Aset
kategori_aset = {
    "Kripto": {
        "BTCUSD (Bitcoin)": "BTC-USD",
        "ETHUSD (Ethereum)": "ETH-USD",
        "SOLUSD (Solana)": "SOL-USD",
        "DOGEUSD (Dogecoin)": "DOGE-USD"
    },
    "Saham & Indeks": {
        "AAPL (Apple)": "AAPL",
        "TSLA (Tesla)": "TSLA",
        "NVDA (NVIDIA)": "NVDA",
        "S&P 500": "^GSPC",
        "IHSG (Indonesia)": "^JKSE"
    },
    "Komoditas & Forex": {
        "XAUUSD (Emas)": "GC=F",
        "XAGUSD (Perak)": "SI=F",
        "USDIDR (Dolar-Rupiah)": "IDR=X",
        "EURUSD (Euro-Dolar)": "EURUSD=X",
        "Crude Oil (Minyak)": "CL=F"
    }
}

# 3. Sidebar - Konfigurasi Aset & Waktu
st.sidebar.header("Pilih")

pilih_kat = st.sidebar.selectbox("Pilih Kategori:", list(kategori_aset.keys()))
pilihan_user = st.sidebar.selectbox("Pilih Aset Spesifik:", list(kategori_aset[pilih_kat].keys()))
simbol_yahoo = kategori_aset[pilih_kat][pilihan_user]

# --- TIMEFRAME ---

map_interval = {
    "1 Menit": {"int": "1m", "per": "1d"},
    "Harian (1D)": {"int": "15m", "per": "7d"},
    "Mingguan (1W)": {"int": "1h", "per": "1mo"},
    "Bulanan": {"int": "1d", "per": "2y"},
    "Tahunan": {"int": "1wk", "per": "5y"}
}

pilih_tf = st.sidebar.selectbox("Pilih Timeframe (Interval):", list(map_interval.keys()), index=3)
interval_kode = map_interval[pilih_tf]["int"]
period_kode = map_interval[pilih_tf]["per"]

# 4. Ambil Data
data = yf.download(simbol_yahoo, period=period_kode, interval=interval_kode)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# Tambah Indikator RSI
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# 5. Logika AI
data['Besok'] = data['Close'].shift(-1)
data['Target'] = (data['Besok'] > data['Close']).astype(int)
data_clean = data.dropna()

features = ['Close', 'Open', 'High', 'Low', 'Volume', 'RSI']

# Cek apakah data cukup untuk training
if len(data_clean) > 50:
    model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1)
    model.fit(data_clean[features], data_clean['Target'])
    can_predict = True
else:
    can_predict = False

# 6. Tampilan Dashboard
st.markdown(f"### Analisis: **{pilihan_user}** | Timeframe: **{pilih_tf}**")

col1, col2 = st.columns([2, 1])

with col1:
    fig = go.Figure(data=[go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'], name="Harga"
    )])
    fig.update_layout(xaxis_rangeslider_visible=False, height=600, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Statistik Terkini")
    harga_akhir = data['Close'].iloc[-1]
    st.metric(label=f"Harga ({pilih_tf})", value=f"{harga_akhir:,.2f}")
    
    st.write(f"**RSI (14):** {data['RSI'].iloc[-1]:.2f}")
    
    st.divider()
    
    st.subheader("Prediksi AI")
    if can_predict:
        df_akhir = data[features].iloc[-1:]
        pred = model.predict(df_akhir)
        prob = model.predict_proba(df_akhir)
        
        # Penyesuaian bahasa prediksi berdasarkan timeframe
        label_pred = "KANDEL BERIKUTNYA" if "Menit" in pilih_tf or "Jam" in pilih_tf else "BESOK"
        
        if pred[0] == 1:
            st.success(f"PREDIKSI {label_pred}: NAIK")
        else:
            st.error(f"PREDIKSI {label_pred}: TURUN")
        st.write(f"Keyakinan: {max(prob[0])*100:.1f}%")
    else:
        st.warning("Data tidak cukup untuk melakukan prediksi pada timeframe ini.")
