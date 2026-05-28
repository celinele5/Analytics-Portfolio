import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# Tải bộ từ điển cảm xúc (Chỉ chạy 1 lần và lưu vào bộ nhớ cache)
@st.cache_resource
def load_nltk():
    nltk.download('vader_lexicon')

load_nltk()
sia = SentimentIntensityAnalyzer()

# --- VỊ TRÍ 1: TÍCH HỢP TỪ ĐIỂN MAPPING TỪ FILE JSON ĐÃ CHUẨN HÓA ---
CITY_TO_PREF = {
    "tokyo": "tokyo", "osaka": "osaka", "nagoya": "aichi", "yokohama": "kanagawa",
    "fukuoka": "fukuoka", "sapporo": "hokkaido", "kawasaki": "kanagawa", "kobe": "hyogo",
    "kyoto": "kyoto", "saitama": "saitama", "iwatsuki": "saitama", "hiroshima": "hiroshima",
    "sendai": "miyagi", "chiba": "chiba", "setagaya": "tokyo", "kitakyushu": "fukuoka",
    "sakai": "fukui", "niigata": "niigata", "hamamatsu": "shizuoka", "nerima": "tokyo",
    "kumamoto": "kumamoto", "ota": "gunma", "sagamihara": "kanagawa", "okayama": "okayama",
    "edogawa": "chiba", "shizuoka": "shizuoka", "adachi": "tokyo", "honcho": "saitama",
    "kawaguchi": "saitama", "kagoshima": "kagoshima", "itabashi": "tokyo", "suginami": "tokyo",
    "hachioji": "tokyo", "himeji": "hyogo", "koto": "tokyo", "utsunomiya": "tochigi",
    "matsuyama": "ehime", "matsudo": "chiba", "ichikawa": "chiba", "higashi": "hiroshima",
    "nishinomiya": "hyogo", "kawachicho": "osaka", "kurashiki": "okayama", "oita": "oita",
    "kanazawa": "ishikawa", "fukuyama": "hiroshima", "amagasaki": "hyogo", "katsushika": "tokyo",
    "fujisawa": "kanagawa", "machida": "tokyo", "kashiwa": "chiba", "aomori": "aomori",
    "toyota": "aichi", "takamatsu": "kagawa", "shinagawa": "tokyo", "toyama": "toyama",
    "nagasaki": "nagasaki", "gifu": "gifu", "toyonaka": "osaka", "miyazaki": "miyazaki",
    "hirakata": "osaka", "yokosuka": "kanagawa", "okazaki": "aichi", "minamisuita": "osaka",
    "ichinomiya": "aichi", "takasaki": "gunma", "toyohashi": "aichi", "nagano": "nagano",
    "kawagoe": "saitama", "wakayama": "wakayama", "kita": "tokyo", "nara": "nara",
    "shinjuku": "tokyo", "takatsuki": "osaka", "nakano": "nagano", "otsu": "shiga",
    "koshigaya": "saitama", "tokorozawa": "saitama", "iwaki": "fukushima", "maebashi": "gunma",
    "asahikawa": "hokkaido", "koriyama": "fukushima", "kochi": "kochi", "naha": "okinawa",
    "yokkaichi": "mie", "kasugai": "aichi", "akita": "akita", "kurume": "fukuoka",
    "oakashicho": "hyogo", "toshima": "tokyo", "morioka": "iwate", "sumida": "tokyo",
    "fukushima": "fukushima", "ibaraki": "osaka", "meguro": "tokyo", "tsu": "mie",
    "mito": "ibaraki", "ichihara": "chiba", "nagaoka": "kyoto", "yao": "nara",
    "fukui": "fukui", "fuchu": "hiroshima", "minato": "tokyo", "hiratsuka": "kanagawa",
    "kakogawacho": "hyogo", "tokushima": "tokushima", "shinozaki": "fukuoka", "hakodate": "hokkaido",
    "soka": "saitama", "yamagata": "gifu", "tsukuba": "ibaraki", "fuji": "shizuoka",
    "sasebo": "nagasaki", "chigasaki": "kanagawa", "bunkyo": "tokyo", "yato": "kanagawa",
    "matsumoto": "nagano", "chofugaoka": "tokyo", "shibuya": "tokyo", "saga": "saga",
    "yoshiicho": "nagasaki", "kasukabe": "saitama", "neya": "osaka", "ageoshimo": "saitama",
    "atsugicho": "kanagawa", "hachinohe": "aomori", "takarazuka": "hyogo", "arakawa": "tokyo",
    "isesaki": "gunma", "kure": "hiroshima", "taito": "tokyo", "nagareyama": "chiba",
    "nishitokyo": "tokyo", "matsue": "shimane", "yachiyo": "chiba", "itami": "hyogo",
    "kodaira": "tokyo", "suzuka": "mie", "kamirenjaku": "tokyo", "kumagaya": "saitama",
    "yamaguchi": "yamaguchi", "hino": "tokyo", "odawara": "kanagawa", "anjomachi": "aichi",
    "kishiwada": "osaka", "numazu": "shizuoka", "tottori": "tottori", "joetsu": "niigata",
    "kofu": "yamanashi", "izuo": "osaka", "toyokawa": "aichi", "tachikawa": "tokyo",
    "uji": "kyoto", "narashino": "chiba", "kamakurayama": "kanagawa", "hitachi": "ibaraki",
    "izumo": "shimane", "tomakomai": "hokkaido", "urayasu": "chiba", "chuo": "tokyo",
    "nishio": "aichi", "hirosaki": "aomori", "oyama": "tochigi", "niiza": "saitama",
    "takaoka": "toyama", "kushiro": "hokkaido", "iwata": "shizuoka", "obihiro": "hokkaido",
    "hadano": "kanagawa", "ube": "yamaguchi", "miyakonojo": "miyazaki", "matsuzaka": "mie",
    "ogaki": "gifu", "daiwanishi": "hyogo", "noda": "chiba", "tochigi": "tochigi",
    "kariya": "hyogo", "ueda": "nagano", "imabari": "ehime", "kawashiri": "osaka",
    "higashimurayama": "tokyo", "kukichuo": "saitama", "musashino": "tokyo", "sayama": "osaka",
    "komaki": "aichi", "tama": "okayama", "yonago": "tottori", "iruma": "saitama",
    "asaka": "saitama", "kakamigahara": "gifu", "ashikaga": "tochigi", "toda": "saitama",
    "tsuchiura": "ibaraki", "okinawa": "okinawa", "misato": "chiba", "moriguchi": "osaka",
    "fujita": "shizuoka", "fukayacho": "saitama", "kusatsu": "shiga", "mino": "osaka",
    "ishizaki": "miyagi", "kuwana": "mie", "koga": "fukuoka", "shunan": "yamaguchi",
    "minoo": "osaka", "yaizu": "shizuoka", "kisarazu": "chiba", "ebina": "kanagawa",
    "inazawa": "aichi", "ome": "tokyo", "isahaya": "nagasaki", "zama": "kanagawa",
    "narita": "chiba", "abiko": "chiba", "onomichi": "hiroshima", "kokubunji": "tokyo",
    "iwakuni": "yamaguchi", "seto": "aichi", "omiyacho": "shizuoka", "koganei": "tokyo",
    "osaki": "miyagi", "iizuka": "fukuoka", "kirishima": "kagoshima", "ise": "mie",
    "uruma": "okinawa", "kashiwara": "osaka", "tsuruoka": "yamagata", "ebetsu": "hokkaido",
    "daitochỏ": "osaka", "kadoma": "osaka", "aizuwakamatsu": "fukushima", "matsubara": "osaka",
    "nobeoka": "miyazaki", "handa": "aichi", "kononu": "saitama", "ikoma": "nara",
    "karatsu": "saga", "nagahama": "shiga", "beppu": "oita", "urasoe": "okinawa",
    "nasushiobara": "tochigi", "koencho": "hokkaido", "niihama": "ehime", "hofu": "yamaguchi",
    "sano": "tochigi", "hatsukaichi": "hiroshima", "kakegawa": "shizuoka", "fujimino": "saitama",
    "hikone": "shiga", "tokai": "aichi", "kazo": "saitama", "oshu": "iwate",
    "higashiomi": "shiga", "akishima": "tokyo", "fujimi": "saitama", "ichinoseki": "iwate",
    "kasuga": "fukuoka", "shirayamamachi": "ishikawa", "omuta": "fukuoka", "kamagaya": "chiba",
    "sandacho": "hyogo", "marugame": "kagawa", "tonandabayashicho": "osaka", "komatsu": "ishikawa",
    "habikino": "osaka", "mineshita": "shizuoka", "tajimi": "gifu", "saijo": "ehime",
    "kiryu": "gunma", "ikeda": "hyogo", "toride": "ibaraki", "chikushino": "fukuoka",
    "inzai": "chiba", "hoyacho": "tokyo", "otaru": "hokkaido", "isehara": "kanagawa",
    "onojo": "fukuoka", "sakado": "saitama", "kawachinagano": "osaka", "kani": "gifu",
    "omura": "nagasaki", "izumisano": "osaka", "ginowan": "okinawa", "sakata": "yamagata",
    "itoshima": "fukuoka", "chikusei": "ibaraki", "kanoya": "kagoshima", "saku": "nagano",
    "chitose": "hokkaido", "tsuyama": "okayama", "munakata": "fukuoka", "kamisu": "ibaraki",
    "shimada": "shizuoka", "kanuma": "tochigi", "shibata": "niigata", "ashiya": "hyogo",
    "azumino": "nagano", "sanjo": "niigata", "inagi": "tokyo", "yashio": "saitama",
    "yotsukaido": "chiba", "nisshin": "aichi", "hanamaki": "iwate", "kitakami": "iwate",
    "satsumasendai": "kagoshima", "higashi-matsuyama": "saitama", "imizucho": "toyama",
    "mihara": "hiroshima", "koka": "shiga", "mobara": "chiba", "ama": "aichi",
    "takasagocho": "hyogo", "fukuroi": "shizuoka", "gotenba": "shizuoka", "settsu": "osaka",
    "kitanagoya": "aichi", "kameoka": "kyoto", "iga": "mie", "sekimachi": "gifu",
    "takayama": "gifu", "yokotemachi": "akita", "ushiku": "ibaraki", "komae": "tokyo",
    "kaizuka": "osaka", "higashiyamato": "saitama", "ayase": "kanagawa", "wako": "saitama",
    "kitakoriyamacho": "nara", "chita": "aichi", "moriyama": "shiga", "nakatsu": "fukuoka",
    "owariasahi": "aichi", "shikokuchuo": "ehime", "muroran": "hokkaido", "omihachiman": "shiga",
    "kashiwazaki": "niigata", "yonezawa": "yamagata", "echizen": "fukui", "hanno": "saitama",
    "gamagori": "aichi", "akiruno": "tokyo", "iwamizawa": "hokkaido", "natori": "miyagi",
    "nakatsugawa": "gifu", "maizuru": "kyoto", "mooka": "tochigi", "gyoda": "saitama",
    "kashiba": "nara", "kizugawa": "kyoto", "aira": "kagoshima", "toyomamachi": "miyagi",
    "katano": "osaka", "tsubame": "niigata", "fukuchiyama": "kyoto", "nikko": "tochigi",
    "nabari": "mie", "toyooka": "hyogo", "shinkai": "saitama", "daisen": "akita",
    "ryugasaki": "ibaraki", "kiyose": "saitama", "kai": "yamanashi", "kunitachi": "tokyo",
    "warabi": "saitama", "amakusa": "kumamoto", "sasagawa": "fukushima", "tosu": "saga",
    "katori": "chiba", "miki": "hyogo", "izumiotsu": "osaka", "okegawa": "saitama",
    "tatebayashi": "gunma", "kyotanabe": "kyoto", "tatsunacho": "hyogo", "yurihonjo": "akita",
    "kasama": "ibaraki", "inuyama": "gifu", "otawara": "tochigi", "shibukawa": "gunma",
    "hekinan": "aichi", "dazaifu": "fukuoka", "yukuhashi": "fukuoka", "yoshikawa": "saitama",
    "chiryu": "aichi", "ishioka": "ibaraki", "musashimurayama": "tokyo", "yawata": "kyoto",
    "eniwa": "hokkaido", "tsurugashima": "saitama", "kiyosu": "aichi", "minami-alps": "yamanashi",
    "uwajima": "ehime", "ritto": "shiga", "soja": "okayama", "toyoake": "aichi",
    "anan": "tokushima", "moriya": "ibaraki", "tanabe": "wakayama", "sabae": "fukui",
    "odate": "akita", "fuefuki": "yamanashi", "shiojiri": "nagano", "kashima": "ibaraki",
    "saiki": "oita", "yachimata": "chiba", "hashima": "gifu", "tsuruga": "fukui",
    "fukutsu": "fukuoka", "ina": "nagano", "kitamoto": "saitama", "tomigusuku": "okinawa",
    "fujioka": "gunma", "yanagawa": "fukuoka", "sodegaura": "chiba", "tenri": "nara",
    "kurihara": "miyagi", "asahi": "chiba", "fujiidera": "osaka", "nago": "okinawa",
    "mizuho": "gifu", "takazawa": "iwate", "takaishi": "osaka", "nogata": "fukuoka",
    "toki": "gifu", "shijonawate": "osaka", "chino": "nagano", "narutacho": "tokushima",
    "minamiuonuma": "niigata", "hidaka": "saitama", "annaka": "gunma", "choshi": "chiba",
    "nihonmatsu": "fukushima", "sakurai": "nara", "sado": "niigata", "hamura": "saitama",
    "funato": "wakayama", "mutsu": "aomori", "tokamachi": "niigata", "usa": "oita",
    "hanyu": "saitama", "minami-soma": "fukushima", "miyakojima": "okinawa", "shiraoka": "saitama",
    "shiogama": "miyagi", "imaricho": "saga", "tomiya": "miyagi", "bando": "ibaraki",
    "izumi": "kagoshima", "tsukubamirai": "ibaraki", "kyotango": "kyoto", "goshogawara": "aomori",
    "sakaidecho": "kagawa", "arao": "kumamoto", "nichinan": "miyazaki", "susono": "shizuoka",
    "yuki": "ibaraki", "nakagawa": "fukuoka", "hamada": "shimane", "yasu": "shiga",
    "satte": "saitama", "hannan": "osaka", "nanao": "ishikawa", "noshiromachi": "akita",
    "tomisato": "chiba", "kameyama": "mie", "asakura": "fukuoka", "midori": "gunma",
    "suwa": "nagano", "nomimachi": "ishikawa", "shimotsucho": "wakayama", "chikugo": "fukuoka",
    "takahama": "aichi", "omitama": "ibaraki", "suzukawa": "kanagawa", "ishigaki": "okinawa",
    "sanmu": "chiba", "yamaga": "kumamoto", "ena": "gifu", "kasuya": "fukuoka",
    "iwakura": "aichi", "higashine": "yamagata", "oamishirasato": "chiba", "tonami": "toyama",
    "hitachi-ota": "ibaraki", "okaya": "nagano", "nanto": "toyama", "gosen": "niigata",
    "kikugawa": "shizuoka", "takeocho": "saga", "noboribetsu": "hokkaido", "sanuki": "kagawa",
    "hioki": "kagoshima", "fujiyoshida": "yamanashi", "nakai": "kochi", "kurayoshi": "tottori",
    "takashima": "shiga", "izunokuni": "shizuoka", "kasaoka": "okayama", "kikuchi": "kumamoto",
    "tamagawa": "fukuoka", "hokota": "ibaraki", "kariya": "aichi", "hokuto": "yamanashi",
    "kitakata": "fukushima", "numata": "gunma", "inuma": "saitama", "masuda": "shimane",
    "iwanuma": "miyagi", "nanjo": "okinawa", "ogimachi": "saga", "hagi": "yamaguchi",
    "goshikicho": "hyogo", "himi": "toyama", "kumatori": "osaka", "inabe": "mie",
    "kobayashi": "miyazaki", "makinohara": "shizuoka", "yatomi": "aichi", "shimabara": "nagasaki",
    "awaji": "hyogo", "akaiwa": "okayama", "kasai": "hyogo", "maniwach": "okayama",
    "uozu": "toyama", "minamishimabara": "nagasaki", "miura": "kanagawa", "oizumi": "gunma",
    "yuzawa": "akita", "shimotsuma": "ibaraki", "kitaibaraki": "ibaraki", "komoro": "nagano",
    "sumoto": "hyogo", "amami": "okinawa", "unzen": "nagasaki", "nanbei": "kanagawa",
    "kato": "hyogo", "tamba-sasayama": "hyogo", "agano": "niigata", "itoigawa": "niigata",
    "kasumigaura": "ibaraki", "sagae": "yamagata", "kurobeshin": "toyama", "nakma": "fukuoka",
    "takikawa": "hokkaido", "mitsuke": "niigata", "higashimatsushima": "miyagi", "inashiki": "ibaraki",
    "hitachiomiya": "ibaraki", "gujo": "gifu", "yoshinogawa": "tokushima", "sakuragawa": "ibaraki",
    "ibusuki": "kagoshima", "nishiwaki": "hyogo", "miyoshidai": "saitama", "misawa": "aomori",
    "maibara": "shiga", "ibara": "okayama", "mizunami": "gifu", "katsuragi": "nara",
    "kanie": "aichi", "yasugicho": "shimane", "fuchucho": "hiroshima", "komatsushimacho": "tokushima",
    "uto": "kumamoto", "usuki": "oita", "sosa": "chiba", "miyajima": "kumamoto",
    "setouchi": "okayama", "isumi": "chiba", "unnan": "shimane", "aizumi": "tokushima",
    "kamata": "fukuoka", "kaizu": "gifu", "kahoku": "ishikawa", "iyo": "ehime",
    "tamura": "fukushima", "seiyo": "ehime", "abashiri": "hokkaido", "ofunato": "iwate",
    "nishihara": "okinawa", "motosu": "gifu", "goto": "nagasaki", "kuji": "iwate",
    "shinjo": "yamagata", "fuso": "aichi", "atami": "shizuoka", "toon": "ehime",
    "shiso": "hyogo", "uonuma": "niigata", "minamishiro": "saitama", "harima": "hyogo",
    "ojiya": "niigata", "asakuchi": "okayama", "shiroishi": "miyagi", "minamikyushu": "kagoshima",
    "bungoono": "oita", "kamaishi": "iwate", "wakabadai": "hokkaido", "yufu": "oita",
    "oharu": "aichi", "shimanto": "kochi", "minamisatsuma": "kagoshima", "omaezaki": "shizuoka",
    "shobara": "hiroshima", "namerikawa": "toyama", "nagato": "yamaguchi", "odamachi": "shimane",
    "bizen": "okayama", "sakaiminato": "shimane", "namegata": "ibaraki", "komagane": "nagano",
    "maebara": "chiba", "zentsujicho": "kagawa", "kuroishi": "aomori", "katagami": "akita",
    "ayabe": "kyoto", "yawatahama": "ehime", "hitoyoshi": "kumamoto", "shimatoba": "kyoto",
    "yaita": "tochigi", "nantan": "kyoto", "tsuruno": "aomori", "yunoshima": "gifu",
    "yanai": "yamaguchi", "motomiya": "fukushima", "kanzakimachi": "saga", "hirakawacho": "aomori",
    "nanyo": "yamagata", "tsuru": "yamanashi", "kaminoyama": "yamagata", "kitaakita": "akita",
    "shibushi": "kagoshima", "tomi": "nagano", "hanawa": "akita", "hirado": "nagasaki",
    "asago": "hyogo", "saito": "miyazaki", "aioi": "hyogo", "higashikagawa": "kagawa",
    "chatan": "okinawa", "nirasaki": "yamanashi", "obama": "fukui", "mima": "tokushima",
    "tainai": "niigata", "kakuda": "miyagi", "mizumaki": "fukuoka", "ouda": "nara",
    "kitsuki": "oita", "ninomiya": "kanagawa", "niimi": "okayama", "takahagi": "ibaraki",
    "shingu": "mie", "ukiha": "fukuoka", "itako": "ibaraki", "akitakata": "hiroshima",
    "nemuro": "hokkaido", "ichikikushikino": "kagoshima", "nayoro": "hokkaido",
    "gojo": "nara", "miyanaga": "fukuoka", "awara": "fukui", "kamo": "niigata",
    "otake": "hiroshima", "omachi": "nagano", "nagai": "yamagata", "kunisakimachi": "oita",
    "ginan": "gifu", "tosa": "kochi", "mimasaka": "okayama", "saikaicho": "nagasaki",
    "niinohe": "iwate"
}

# --- PHẦN 1: HÀM CÀO DỮ LIỆU TỔNG HỢP ALL REVIEWS ---
def get_full_data_flexible(city_name, dish_name):
    # Sửa lỗi 1: Tự động chuyển đổi thành phố nhập vào sang Tỉnh tương ứng trên URL
    clean_city = city_name.strip().lower()
    tabelog_pref = CITY_TO_PREF.get(clean_city, clean_city)
    
    url = f"https://tabelog.com/en/{tabelog_pref}/rstLst/?sw={dish_name.replace(' ', '+')}"
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

                # Truy cập vào trang tập hợp Review của quán
                rev_res = requests.get(full_review_url, headers=headers)
                rev_soup = BeautifulSoup(rev_res.content, "html.parser")
                
                # Sửa lỗi 2: Dùng find_all để nhặt SẠCH các khối review hiển thị trên trang
                comment_divs = rev_soup.find_all("div", class_="rvw-item__rvw-comment")
                
                all_restaurant_reviews = []
                for div in comment_divs:
                    if div.p:
                        all_restaurant_reviews.append(div.p.get_text(separator=' ', strip=True))
                
                # Gộp tất cả các comment của riêng quán này thành 1 đoạn văn duy nhất để tính điểm
                combined_reviews = " ".join(all_restaurant_reviews) if all_restaurant_reviews else "N/A"

                data_list.append({
                    "Restaurant": name,
                    "Rating": float(rating) if rating != "0.00" else 0.0,
                    "Full_Review": combined_reviews,
                    "Link": raw_link
                })
                time.sleep(1) # Delay nhẹ tránh bị block IP tại Nhật
            except:
                continue
        return pd.DataFrame(data_list)
    except Exception as e:
        st.error(f"Lỗi kết nối hệ thống: {e}")
        return pd.DataFrame()

# --- PHẦN 2: GIAO DIỆN WEB STREAMLIT ---
# Dòng st.set_page_config BẮT BUỘC phải nằm ngay đây (lệnh hiển thị đầu tiên) để không bị lỗi màn hình!
st.set_page_config(page_title="Japan Foodie Sentiment Platform", layout="wide")
st.title("🏯 Japan Foodie Sentiment Platform")

with st.sidebar:
    st.header("Cấu hình tìm kiếm")
    city_input = st.text_input("Nhập Thành phố hoặc Tỉnh (vd: nagoya, tokyo, himeji):", "nagoya")
    dish_input = st.text_input("Món ăn muốn tìm:", "Miso Katsu")
    search_button = st.button("Phân tích ngay")

# --- PHẦN 3: LOGIC KHỞI CHẠY ---
if search_button:
    with st.spinner(f"Đang bóc tách dữ liệu và phân tích từ khóa chuyên sâu cho món {dish_input}..."):
        final_df = get_full_data_flexible(city_input, dish_input)
        
        if not final_df.empty:
            # Chấm điểm cảm xúc dựa trên TỔNG HỢP review
            final_df['Vibe_Score'] = final_df['Full_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
            
            # Giao diện chia làm 2 cột cân xứng cực đẹp
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Trực quan hóa: Rating vs Vibe Score")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.scatterplot(data=final_df, x='Rating', y='Vibe_Score', s=120, ax=ax, color="orange")
                for i in range(final_df.shape[0]):
                    ax.text(final_df.Rating[i]+0.01, final_df.Vibe_Score[i], final_df.Restaurant[i], fontsize=8)
                st.pyplot(fig)
            
            with col2:
                st.subheader("☁️ Word Map: Các tính từ khóa đắt giá nhất")
                # Gom toàn bộ review của tất cả các quán lại thành một chuỗi lớn để làm WordCloud
                all_text = " ".join(final_df['Full_Review'].astype(str))
                
                # Cấu hình danh sách từ khóa vô nghĩa cần loại bỏ khỏi WordMap
                custom_stopwords = set(STOPWORDS)
                custom_stopwords.update(["restaurant", "food", "place", "order", "ordered", "table", "tabelog", "good", "nice", "eat", "came", "tokyo", "nagoya", "osaka"])
                
                # Khởi tạo mô hình WordCloud
                wordcloud = WordCloud(width=600, height=400, 
                                      background_color='white', 
                                      stopwords=custom_stopwords, 
                                      colormap='Set2').generate(all_text)
                
                fig_wc, ax_wc = plt.subplots(figsize=(6, 4))
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)
                
            # Bảng số liệu chi tiết phía dưới
            st.subheader("📋 Bảng xếp hạng Insights")
            st.dataframe(final_df[['Restaurant', 'Rating', 'Vibe_Score']].sort_values(by='Vibe_Score', ascending=False))
        else:
            st.warning("Không tìm thấy dữ liệu. Hãy kiểm tra xem bạn đã gõ đúng tên tiếng Anh chưa nha!")