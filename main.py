from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from math import ceil
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html', prediction_table=None, error=None)

@app.route('/', methods=['POST'])
def predict():
    try:
        # Ambil input dari form
        komoditas = request.form.get('komoditas')
        hari = request.form.get('hari')

        # Validasi input
        if not komoditas or komoditas == "Pilih komoditas":
            return render_template("index.html", error="Error: Pilih komoditas yang valid.", prediction_table=None)
        
        if not hari or not hari.isdigit():
            return render_template("index.html", error="Error: Pilih jumlah hari dengan benar.", prediction_table=None)

        hari = int(hari)

        # Load data
        df = pd.read_csv('DataHargaPangan.csv')
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)

        # Pastikan komoditas ada dalam dataset
        if komoditas not in df.columns:
            return render_template("index.html", error="Error: Komoditas tidak ditemukan.", prediction_table=None)

        # Ambil data komoditas yang dipilih
        data = df[[komoditas]]
        scaler = MinMaxScaler()

        # Normalisasi data
        normalized_data = scaler.fit_transform(data)
        poly = PolynomialFeatures(degree=7)

        # Prediksi untuk masa depan
        last_date = df['Tanggal'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=hari)
        future_dates_num = np.arange(1, hari + 1).reshape(-1, 1)
        future_dates_poly = poly.fit_transform(future_dates_num)

        predictions = model.predict(future_dates_poly)

        # Invers transform hasil prediksi
        predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        # Hasil prediksi
        prediction_df = pd.DataFrame({
            "Tanggal": future_dates,
            "Komoditas": [komoditas] * hari,
            "Prediksi Harga": predictions_rescaled.round(2)
        })

        prediction_df['Tanggal'] = prediction_df['Tanggal'].dt.strftime('%d-%m-%Y')
        prediction_df['Prediksi Harga'] = prediction_df['Prediksi Harga'].astype(int)

        return render_template("index.html", prediction_table=prediction_df.to_dict(orient='records'), error=None)

    except Exception as e:
        return render_template("index.html", error=f"Error: {str(e)}", prediction_table=None)

@app.route('/data', methods=['GET'])
def data():
    komoditas = request.args.get('komoditas')

    # Load data
    df = pd.read_csv('DataHargaPangan.csv')
    df = df.dropna()
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)
    df['Tanggal'] = df['Tanggal'].dt.strftime('%d-%m-%Y')

    # Pastikan komoditas ada dalam dataset
    if komoditas and komoditas != "Pilih Komoditas":
        if komoditas not in df.columns:
            return render_template("data.html", error="Error: Komoditas tidak ditemukan.", dataset=None, page=1, total_pages=1, page_range=[])

        # Filter data berdasarkan komoditas
        df = df[['Tanggal', komoditas]]

    # Konfigurasi pagination
    ITEMS_PER_PAGE = 5  # Tentukan berapa item per halaman
    page = request.args.get('page', 1, type=int)

    total_items = len(df)
    total_pages = ceil(total_items / ITEMS_PER_PAGE)

    # Tentukan range halaman yang akan ditampilkan
    page_range = list(range(1, total_pages + 1))

    if total_pages > 5:
        if page <= 3:
            page_range = [1, 2, 3, 4, '...', total_pages]
        elif page >= total_pages - 2:
            page_range = [1, '...', total_pages - 3, total_pages - 2, total_pages - 1, total_pages]
        else:
            page_range = [1, '...', page - 1, page, page + 1, '...', total_pages]

    # Menentukan halaman yang ditampilkan
    start_idx = (page - 1) * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    page_data = df.iloc[start_idx:end_idx]

    return render_template('data.html', dataset=page_data.to_dict(orient='records'),
                           page=page, total_pages=total_pages, page_range=page_range, komoditas=komoditas)

if __name__ == '__main__':
    app.run(debug=True)
