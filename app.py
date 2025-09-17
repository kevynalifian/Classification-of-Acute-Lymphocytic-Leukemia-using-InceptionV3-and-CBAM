import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import os
import base64

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    """Memuat model Keras yang telah dilatih."""
    try:
        model_path = 'modelfix_lla_inception_cbam.keras'
        if not os.path.exists(model_path):
            st.error(f"File '{model_path}' tidak ditemukan di direktori kerja saat ini.")
            st.error(f"Isi direktori: {os.listdir()}")
            st.stop()
        model = keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

def preprocess_image(image):
    """
    Melakukan pra-pemrosesan gambar masukan ke ukuran target dan menormalisasi nilai piksel.
    Mengkonversi gambar RGBA ke RGB jika diperlukan.
    """
    target_size = (224, 224)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    img_array = img_array / 255.0 # Normalisasi nilai piksel ke [0, 1]
    return img_array

def preprocess_image(image):
    """
    Melakukan pra-pemrosesan gambar masukan ke ukuran target dan menormalisasi nilai piksel.
    Mengkonversi gambar RGBA ke RGB jika diperlukan.
    """
    target_size = (224, 224)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    img_array = img_array / 255.0 # Normalisasi nilai piksel ke [0, 1]
    return img_array

# Fungsi untuk melakukan prediksi
def predict(model, processed_image):
    """Melakukan prediksi menggunakan model yang dimuat."""
    predictions = model.predict(processed_image)
    # Asumsi klasifikasi biner di mana output adalah nilai tunggal
    return float(predictions[0][0])

# --- Konfigurasi UI Streamlit ---
st.set_page_config(
    page_title="Klasifikasi LLA",
    page_icon="🔬",
    layout="wide" # Gunakan layout lebar untuk distribusi kolom yang lebih baik
)

# --- Bagian CSS Kustom untuk Gambar Latar Belakang ---
background_image_path = "landing page.png"

# Baca file gambar dan konversi ke base64
try:
    with open(background_image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    background_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover; /* Menutupi seluruh area */
        background-position: center; /* Pusatkan gambar */
        background-repeat: no-repeat; /* Jangan ulangi gambar */
        background-attachment: fixed; /* Gambar tetap saat menggulir */
        color: #FF0084; /* Mengubah warna teks default aplikasi */
    }}
    /* Kontainer utama untuk konten Streamlit di atas background */
    .main-content-container {{
        background-color: rgba(255, 255, 255, 0.85); /* Putih dengan opasitas 85% */
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        margin-bottom: 20px;
        /* Untuk memastikan konten tidak terlalu dekat dengan tepi */
        margin-left: auto;
        margin-right: auto;
        max-width: 1200px; /* Batasi lebar kontainer */
        color: #FF0084; /* Mengubah warna teks di dalam kontainer utama */
    }}
    /* Mengubah warna judul dan subjudul */
    h1, h2, h3, h4, h5, h6 {{
        color: #FF0084;
    }}
    /* Mengubah warna teks di st.info dan st.success */
    /* Menggunakan !important untuk memastikan override gaya default Streamlit */
    .stAlert.info .stMarkdown p, .stAlert.success .stMarkdown p {{
        color: #FF0084 !important; /* Mengubah warna teks di dalam alert */
    }}
    /* Pastikan teks di dalam st.warning tetap terbaca */
    .stAlert.warning {{
        color: #000000 !important; /* Contoh: tetap hitam agar kontras */
    }}
    /* Mengatur warna teks di dalam elemen p (paragraf) yang mungkin digunakan oleh st.write */
    p {{
        color: #FF0084;
    }}
    /* Gaya untuk kotak deskripsi dengan opasitas dan border hitam */
    .description-box {{
        background-color: rgba(255, 255, 255, 0.5); /* Putih dengan opasitas 50% */
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 15px;
        border: 1px solid black; /* Menambahkan garis hitam */
    }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)
except FileNotFoundError:
    st.warning(f"Gambar latar belakang '{background_image_path}' tidak ditemukan. Latar belakang tidak akan diterapkan.")
    # Jika gambar tidak ditemukan, tetap sediakan kontainer dengan latar belakang semi-transparan
    st.markdown(
        """
        <style>
        .main-content-container {
            background-color: rgba(255, 255, 255, 0.95); /* Sedikit kurang transparan jika tanpa gambar background */
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            margin-bottom: 20px;
            margin-left: auto;
            margin-right: auto;
            max-width: 1200px;
            color: #FF0084; /* Mengubah warna teks di dalam kontainer utama */
        }
        h1, h2, h3, h4, h5, h6 {
            color: #FF0084;
        }
        .stAlert.info .stMarkdown p, .stAlert.success .stMarkdown p {
            color: #FF0084 !important; /* Mengubah warna teks di dalam alert */
        }
        .stAlert.warning {
            color: #000000 !important; /* Contoh: tetap hitam agar kontras */
        }
        p {
            color: #FF0084;
        }
        .description-box {
            background-color: rgba(255, 255, 255, 0.5); /* Putih dengan opasitas 50% */
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            margin-bottom: 15px;
            border: 1px solid black; /* Menambahkan garis hitam */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Konten Utama Dibungkus dalam Kontainer ---
# Semua konten Streamlit akan berada di dalam div ini untuk memastikan keterbacaan di atas latar belakang.
st.markdown('<div class="main-content-container">', unsafe_allow_html=True)

# Judul utama dan pengantar
st.title("Klasifikasi LLA (Leukemia Limfoblastik Akut)")
st.markdown("---") # Pemisah

# --- Tata Letak Konten Utama ---
col_upload_results, col_description = st.columns([1, 2]) # Kolom kiri 1 bagian, kolom kanan 2 bagian

with col_upload_results:
    st.header("Unggah Gambar Sel Darah")
    
    # Pengunggah file
    uploaded_file = st.file_uploader("Pilih gambar untuk klasifikasi", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Tampilkan gambar yang diunggah di sini, di bawah uploader
        st.image(image, caption='Gambar yang Diunggah', width=250) # Mengatur lebar gambar
        st.markdown("---") # Tambahkan pemisah setelah gambar

        st.write("Memproses dan Melakukan Klasifikasi...")

        # Muat model hanya saat gambar diunggah (di-cache untuk efisiensi)
        model = load_model()

        if model is None:
            st.warning("Tidak dapat melakukan klasifikasi karena model gagal dimuat.")
        else:
            processed_image = preprocess_image(image)
            prediction = predict(model, processed_image)

            # Tentukan kelas dan skor kepercayaan berdasarkan prediksi
            # Asumsi 0.5 sebagai ambang batas untuk klasifikasi biner
            if prediction >= 0.5:
                predicted_class_name = "Normal"
                confidence_score = prediction * 100
            else:
                predicted_class_name = "LLA" # Leukemia Limfoblastik Akut
                confidence_score = (1 - prediction) * 100

            # Tampilkan hasil klasifikasi langsung di bawah pengunggah file
            st.subheader("Hasil Klasifikasi")
            # Menggunakan HTML langsung untuk menerapkan warna font pada teks di dalam st.success dan st.info
            # CSS di atas kini menargetkan teks di dalam alert lebih spesifik.
            st.success(f"**Klasifikasi:** {predicted_class_name}")
            st.info(f"**Tingkat Kepercayaan:** {confidence_score:.2f}%")
    # Tidak ada lagi `else` untuk `st.info("Silakan unggah gambar sel darah untuk memulai klasifikasi.")`
    # Pesan ini hanya akan muncul jika `uploaded_file` adalah `None` secara default pada awal aplikasi.

with col_description:
    # Menggunakan satu blok st.markdown untuk seluruh konten deskripsi
    # Ini memastikan teks berada di dalam kotak dengan background dan border
    st.markdown(
        f"""
        <div class="description-box">
            <h3 style="color:#FF0084;">Tentang Website</h3>
            <p style="color:#FF0084;">
                Website ini dirancang untuk membantu dalam klasifikasi Leukemia Limfoblastik Akut (LLA),
            </p>
            <p style="color:#FF0084;">
                berdasarkan citra sel darah putih.
            </p>
            <hr style="border-top: 1px solid #FF0084;"> <!-- Garis pemisah dengan warna font -->
            <p style="color:#FF0084; font-size: 0.9em;">
                Model klasifikasi menggunakan InceptionV3 + CBAM dan pelatihan GAN.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Tutup div kontainer utama konten
st.markdown('</div>', unsafe_allow_html=True)