from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from math import ceil
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
        df = df.dropna()

        # Pastikan komoditas ada dalam dataset
        if komoditas not in df.columns:
            return render_template("index.html", error="Error: Komoditas tidak ditemukan.", prediction_table=None)

        # Normalisasi hanya pada kolom komoditas yang dipilih
        scaler = MinMaxScaler()
        y = scaler.fit_transform(df[[komoditas]])  # Fit hanya pada kolom komoditas yang dipilih

        # Menyiapkan data untuk regresi
        df['Tanggal_Num'] = (df['Tanggal'] - df['Tanggal'].min()).dt.days  # Konversi tanggal ke numerik
        X = df[['Tanggal_Num']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model regresi polinomial
        degree = 7  # Tingkat polinomial
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Melatih model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # Prediksi untuk masa depan
        future_dates = pd.date_range(start=df['Tanggal'].max() + pd.Timedelta(days=1), periods=hari, freq='D')
        future_dates_num = (future_dates - df['Tanggal'].min()).days.values.reshape(-1, 1)
        future_dates_poly = poly.fit_transform(future_dates_num)
        predictions = model.predict(future_dates_poly)

        # Inverse transform hasil prediksi
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

@app.route('/model')
def model():
    try:
        # Load data
        df = pd.read_csv('DataHargaPangan.csv')
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)
        df = df.dropna()

        data = df[["Beras Premium",	"Beras Medium", "Kedelai Biji Kering (Impor)",	"Bawang Merah",	"Bawang Putih Bonggol",	"Cabai Merah Keriting",	"Cabai Rawit Merah",	"Daging Sapi Murni",	"Daging Ayam Ras",	"Telur Ayam Ras", "Gula Pasir/Konsumsi",	"Minyak Goreng Kms. Sederhana",	"Tepung Terigu (Curah)",	"Minyak Goreng Curah",	"Cabai Merah Besar",	"Jagung Pipilan Kering",	"Kentang",	"Tomat"]]

        # Inisialisasi scaler
        scaler = MinMaxScaler()

        # Normalisasi data
        normalized_data = scaler.fit_transform(data)

        # Konversi kembali ke DataFrame agar mudah dibaca
        normalized_df = pd.DataFrame(normalized_data, columns=["Beras Premium",	"Beras Medium", "Kedelai Biji Kering (Impor)",	"Bawang Merah",	"Bawang Putih Bonggol",	"Cabai Merah Keriting",	"Cabai Rawit Merah",	"Daging Sapi Murni",	"Daging Ayam Ras",	"Telur Ayam Ras", "Gula Pasir/Konsumsi",	"Minyak Goreng Kms. Sederhana",	"Tepung Terigu (Curah)",	"Minyak Goreng Curah",	"Cabai Merah Besar",	"Jagung Pipilan Kering",	"Kentang",	"Tomat"])

        # Menyiapkan data untuk regresi
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)
        df['Tanggal_Num'] = (df['Tanggal'] - df['Tanggal'].min()).dt.days  # Konversi tanggal ke numerik
        X = df[['Tanggal_Num']]
        y = normalized_df['Beras Premium']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model regresi polinomial
        degree = 7  # Tingkat polinomial
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Melatih model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # Evaluasi model pada testing set
        y_pred = model.predict(X_test_poly)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        r2_percentage = round(r2*100, 2)

        coefficients = model.coef_
        intercept = model.intercept_

        # Menyimpan koefisien dalam format string
        formatted_coefficients = [f"x^{i}: {coef}" for i, coef in enumerate(coefficients)]

        # Format koefisien menjadi persamaan
        formatted_terms = []
        for i, coef in enumerate(coefficients):
            if coef != 0:  # Abaikan koefisien 0
                formatted_coef = f"({coef:.4e})".replace('e', '×10^').replace('+', '')  # Format e-notation
                term = f"{formatted_coef}x^{i}" if i > 0 else f"{formatted_coef}"  # Tambahkan x^i jika i > 0
                formatted_terms.append(term)

        # Gabungkan semua term menjadi persamaan
        equation = f"y = {intercept:.4f} + " + " + ".join(formatted_terms)
        inter = f"{intercept:.4f}"

        return render_template("model.html", persamaan = equation, koefisien = formatted_coefficients, intercept = inter, mse = mse, r2 = r2_percentage, error=None)

    except Exception as e:
        return render_template("model.html", error=f"Error: {str(e)}", prediction_table=None)

if __name__ == '__main__':
    app.run(debug=True)
