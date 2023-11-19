#!/usr/bin/env python
# coding: utf-8
# In[22]:


import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load_data():
    dataset = {'Model': ['Samsung Galaxy S21','Apple iPhone 12','Xiaomi Mi 11','Huawei P40','OnePlus 9 Pro','Sony Xperia 5 II','Google Pixel 5','Nokia 83 5G','LG Velvet','Motorola Edge+','ASUS ROG Phone 5','Lenovo Legion Duel','HTC U12+','Oppo Find X3 Pro','Vivo X60 Pro+','Realme GT','Redmi Note 10 Pro','Asus ZenFone 8','Apple iPhone SE','Samsung Galaxy S20','Xiaomi Redmi Note 9','Huawei Mate 40 Pro','OnePlus 8T','Sony Xperia 1 III','Google Pixel 4a','Nokia 54','LG Wing','Motorola Moto G Power','ASUS ZenFone 7 Pro','Lenovo Legion Pro 2','HTC U20 5G','Oppo Reno5 Pro+','Samsung Galaxy A52','Apple iPhone 13 Pro','Xiaomi Redmi 9','Huawei Nova 8 Pro','Samsung Galaxy M51','Realme Narzo 30','Vivo Y12s','Xiaomi Redmi 9A','Motorola Moto E7 Plus','OPPO A53','Samsung Galaxy A12','Realme C21','Xiaomi Poco M3','Vivo Y20s','OPPO A15s','Samsung Galaxy A02s','Realme C15','Xiaomi Redmi 9C'],
                'RAM': [ 8,  4,  8,  8, 12,  8,  8,  6,  6, 12, 16, 16,  6,
                    12, 12,  8,  8,  8,  3,  8,  4,  8, 12, 12,  6,  4,
                        8,  4,  8, 18,  8,  8,  6,  6,  4,  8,  8,  4,  3,
                        2,  4,  6,  4,  4,  4,  4,  4,  4,  4,  2], 
                'Penyimpanan': [128,  64, 256, 128, 256, 128, 128,  64, 128, 256, 256,
                    512, 128, 256, 256, 128, 128, 128,  64, 128, 128, 256,
                    256, 256, 128,  64, 256,  64, 256, 512, 256, 128, 128,
                    128,  64, 128, 128,  64,  32,  32,  64, 128, 128,  64,
                        64, 128,  64,  64,  64,  32], 
                'Kamera': [ 64,  12, 108,  50,  48,  12, 122,  64,  48, 108,  64,
                        64,  12,  50,  50,  64, 108,  64,  12,  64,  48,  50,
                        48,  12, 122,  48,  64,  16,  64,  64,  48,  50,  64,
                        12,  13,  64,  64,  48,  13,  13,  48,  13,  48,  13,
                        48,  13,  13,  13,  13,  13], 
                'Baterai': [4000, 2942, 5000, 4200, 4500, 4000, 4080, 4500, 4300,
                    5000, 6000, 5000, 3500, 4500, 4200, 4500, 5020, 4000,
                    1821, 4000, 5020, 4400, 4500, 4500, 3140, 4000, 4000,
                    5000, 5000, 5500, 5000, 4500, 4500, 3095, 5000, 4000,
                    7000, 6000, 5000, 5000, 5000, 5000, 5000, 5000, 6000,
                    5000, 4230, 5000, 6000, 5000], 
                'Harga': [14999000, 16999000, 12999000, 11999000, 16999000, 12999000,
                    10999000,  6999000,  7999000, 14999000, 16999000, 14999000,
                        6999000, 17999000, 14999000,  7999000,  4999000,  7999000,
                        5999000, 14999000,  3999000, 17999000, 10999000, 17999000,
                        4999000,  3999000, 14999000,  3999000, 12999000, 16999000,
                        9999000, 13999000,  6999000, 16999000,  2999000,  8999000,
                        4499000,  2199000,  1999000,  1499000,  1699000,  2999000,
                        2499000,  1899000,  1999000,  2699000,  1899000,  1999000,
                        1899000,  1599000]
                }
    data = pd.DataFrame(dataset)
    return data


def calculate_saw(data, weights):
    scaled_data = data.copy()
    scaler = MinMaxScaler()

    for feature in weights:
        scaled_data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))
        scaled_data[feature] *= weights[feature]

    data['Nilai Preferensi'] = scaled_data.sum(axis=1)
    data = data.sort_values(by='Nilai Preferensi', ascending=False).reset_index(drop=True)

    return data

def filter_hp(data, filter_values):
    filtered_data = data.copy()
    for feature, value in filter_values.items():
        if feature == 'Harga':
            filtered_data = filtered_data[filtered_data[feature].between(value[0], value[1])]

    return filtered_data

def recommend_hp(data, top_n):
    recommended_hp = data['Model'].head(top_n).values
    return recommended_hp

def visualize_graph(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(data['Model'], data['Nilai Preferensi'])
    ax.set_xlabel('Model HP')
    ax.set_ylabel('Nilai Preferensi')
    ax.set_title('Perbandingan Nilai Preferensi HP')
    plt.xticks(rotation=90)
    st.pyplot(fig)

def main():
    st.markdown("<h1 style='text-align: center; color: white;'>Sistem Pendukung Keputusan Pembelian HP</h1>", unsafe_allow_html=True)
    st.markdown("")

    # Load data spesifikasi HP
    hp_data = load_data()

    # Menampilkan data spesifikasi HP
    st.subheader('Data Spesifikasi HP')
    st.dataframe(hp_data)

    # Menentukan bobot untuk setiap kriteria
    weights = {
        'RAM': 0.3,
        'Penyimpanan': 0.3,
        'Kamera': 0.2,
        'Baterai': 0.2,
        'Harga': 0.4
    }

    # Menghitung nilai preferensi menggunakan algoritma SAW
    preferences = calculate_saw(hp_data, weights)

    # Menampilkan data spesifikasi HP dengan nilai preferensi
    st.subheader('Data Spesifikasi HP dengan Nilai Preferensi')
    st.dataframe(preferences)

    # Filter range harga HP yang dicari
    st.subheader('Filter Range Harga HP')
    min_price = st.number_input('Harga Minimum', value=hp_data['Harga'].min(), min_value=hp_data['Harga'].min(),
                                max_value=hp_data['Harga'].max())
    max_price = st.number_input('Harga Maksimum', value=hp_data['Harga'].max(), min_value=hp_data['Harga'].min(),
                                max_value=hp_data['Harga'].max())
    filter_values = {'Harga': (min_price, max_price)}

    filtered_hp = filter_hp(preferences, filter_values)

    # Menampilkan data spesifikasi HP yang memenuhi filter
    st.subheader('Data Spesifikasi HP yang Memenuhi Filter')
    st.dataframe(filtered_hp)

    # Jumlah rekomendasi HP yang ingin ditampilkan
    top_n = st.number_input('Jumlah Rekomendasi HP', value=1, min_value=1, max_value=len(filtered_hp))

    # Rekomendasi HP untuk dibeli
    st.subheader('Rekomendasi HP')
    recommended_hp = recommend_hp(filtered_hp, top_n)
    st.write(recommended_hp)

    # Visualisasi grafik
    st.subheader('Grafik Nilai Preferensi HP')
    visualize_graph(filtered_hp[0:7])
if __name__ == '__main__':
    main()




