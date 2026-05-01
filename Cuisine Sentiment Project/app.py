import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Tải bộ từ điển cảm xúc (Chỉ cần chạy 1 lần)
@st.cache_resource
def load_nltk():
    nltk.download('vader_lexicon')

load_nltk()
sia = SentimentIntensityAnalyzer()

# --- PHẦN 1: HÀM CÀO DỮ LIỆU ---
def get_full_data_flexible(city_name, dish_name):
    # Dùng đúng tham số truyền vào
    url = f"https://tabelog.com/en/{city_name}/rstLst/?sw={dish_name.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        items = soup.find_all("div", class_="list-rst")
        
        data_list = []
        for item in items:
            try:
                name_tag = item.find("a", class_="list-rst__rst-name-target")
                name = name_tag.text.strip()
                rating = item.find("span", class_="list-rst__rating-val").text.strip()

                raw_link = name_tag.get('href')
                if not raw_link.startswith('http'):
                    raw_link = "https://tabelog.com" + raw_link
                full_review_url = raw_link + "dtlrvwlst/"

                # Truy cập lấy Full Review
                rev_res = requests.get(full_review_url, headers=headers)
                rev_soup = BeautifulSoup(rev_res.content, "html.parser")
                comment_div = rev_soup.find("div", class_="rvw-item__rvw-comment")
                full_review = comment_div.p.get_text(separator=' ', strip=True) if comment_div else "N/A"

                data_list.append({
                    "Restaurant": name,
                    "Rating": float(rating) if rating != "0.00" else 0.0,
                    "Full_Review": full_review,
                    "Link": raw_link
                })
                time.sleep(1) # Nghỉ xíu cho Tabelog đỡ gắt
            except:
                continue
        return pd.DataFrame(data_list)
    except Exception as e:
        st.error(f"Lỗi truy cập: {e}")
        return pd.DataFrame()

# --- PHẦN 2: GIAO DIỆN ---
st.set_page_config(page_title="Japan Foodie Sentiment", layout="wide")
st.title("🏯 Japan Foodie Sentiment Platform")

with st.sidebar:
    st.header("Cấu hình tìm kiếm")
    city_input = st.text_input("Thành phố (vd: aichi, tokyo, osaka):", "aichi")
    dish_input = st.text_input("Món ăn:", "Miso Katsu")
    search_button = st.button("Phân tích ngay")

# --- PHẦN 3: LOGIC ---
if search_button:
    with st.spinner(f"Đang bóc tách dữ liệu {dish_input}..."):
        final_df = get_full_data_flexible(city_input, dish_input)
        
        if not final_df.empty:
            # Sentiment Analysis
            final_df['Vibe_Score'] = final_df['Full_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
            
            # Chia cột hiển thị
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Biểu đồ Rating vs Sentiment")
                fig, ax = plt.subplots()
                sns.scatterplot(data=final_df, x='Rating', y='Vibe_Score', s=100, ax=ax)
                for i in range(final_df.shape[0]):
                    ax.text(final_df.Rating[i]+0.02, final_df.Vibe_Score[i], final_df.Restaurant[i], fontsize=8)
                st.pyplot(fig)
            
            with col2:
                st.subheader("Bảng dữ liệu chi tiết")
                st.dataframe(final_df[['Restaurant', 'Rating', 'Vibe_Score']])
                
            st.write("### Review đầy đủ")
            st.table(final_df[['Restaurant', 'Full_Review']])
        else:
            st.warning("Không tìm thấy quán nào, thử kiểm tra lại tên thành phố/món ăn nha!")