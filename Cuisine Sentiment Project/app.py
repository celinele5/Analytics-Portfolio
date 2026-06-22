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
import plotly.express as px # THÊM CHÍNH XÁC DÒNG NÀY NÈ
st.set_page_config(page_title="Japan Foodie Sentiment Platform", layout="wide")


# Tải bộ từ điển cảm xúc (Chỉ chạy 1 lần và lưu vào bộ nhớ cache)
@st.cache_resource
def load_nltk():
    nltk.download('vader_lexicon')

load_nltk()
sia = SentimentIntensityAnalyzer()
@st.cache_resource

@st.cache_resource
def load_nltk():
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('punkt_tab')                 # THÊM CHÍNH XÁC DÒNG NÀY NÈ!
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng') # Thêm luôn dòng này phòng hờ bản NLTK mới bắt bẻ nhãn tiếng Anh

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
st.title("🏯 Japan Foodie Sentiment Platform")

with st.sidebar:
    st.header("Search Parameters")
    city_input = st.text_input("Enter City or Prefecture (e.g., nagoya, tokyo, himeji):", "nagoya")
    dish_input = st.text_input("Dish you want to search for:", "Miso Katsu")
    search_button = st.button("Analyze Now")

# --- PHẦN 3: LOGIC KHỞI CHẠY ---
if search_button:
    with st.spinner(f"Decomposing and analyzing specialized keywords for dish {dish_input}..."):
        final_df = get_full_data_flexible(city_input, dish_input)
        
        if not final_df.empty:
            # Chấm điểm cảm xúc dựa trên TỔNG HỢP review
            final_df['Vibe Score'] = final_df['Full_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
            
            # Giao diện chia làm 2 cột cân xứng cực đẹp
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Visualization: Rating vs Vibe Score")
                try:
                    # 1. Tạo bản sao dữ liệu và tính toán thứ hạng Vibe
                    plot_df = final_df.copy()
                    plot_df['Vibe Rank'] = plot_df['Vibe Score'].rank(ascending=False, method='min').astype(int)
                    
                    # Đổi tên cột trong DataFrame tạm để khi hiện lên bảng thông tin (hover box) nhìn đẹp mắt hơn
                    plot_df.columns = ['Restaurant', 'Rating', 'Full_Review', 'Link', 'Vibe Score', 'Vibe Rank']
                    
                    # 2. Vẽ biểu đồ Scatter tương tác bằng Plotly Express
                    fig_plotly = px.scatter(
                        plot_df,
                        x='Rating',
                        y='Vibe Score',
                        hover_name='Restaurant',          # Tên quán hiện to làm tiêu đề hộp thông tin
                        hover_data={'Vibe Rank': True, 'Rating': ':.2f', 'Vibe Score': ':.2f'}, # Hiện các thông số đã định dạng
                        title=None
                    )
                    
                    # 3. Thay đổi màu sắc và kích thước chấm tròn cho đồng bộ giao diện Streamlit
                    fig_plotly.update_traces(marker=dict(size=12, color='#FF4B4B', line=dict(width=1, color='White')))
                    
                    # 4. Đẩy biểu đồ Plotly lên giao diện Streamlit
                    st.plotly_chart(fig_plotly, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Cannot display interactive chart due to: {e}")

            with col2:
                st.subheader("☁️ Word Map: Uncover Vibe Beneath the Surface (NLP Filtered)")
                
                # 1. Gom text
                all_text = " ".join(final_df['Full_Review'].astype(str))
                
                # 2. Dùng NLTK tách từ và gắn nhãn từ loại (Tính từ, Danh từ...)
                words = nltk.word_tokenize(all_text)
                tagged_words = nltk.pos_tag(words)
                
                # 3. Lọc thông minh: Chỉ giữ lại từ là Tính từ (JJ) hoặc các từ vận hành cốt lõi
                intelligent_words = []
                core_operational_words = ["cash", "card", "lunch", "noon", "queue", "line", "wait", "staff"]
                
                for word, tag in tagged_words:
                    word_lower = word.lower()
                    
                    # Loại bỏ từ khóa tìm kiếm (Món ăn/Thành phố) để tránh loãng hình
                    if word_lower in city_input.lower() or word_lower in dish_input.lower():
                        continue
                        
                    # JJ: Adjective (Tính từ), JJR: Adjective Comparative, JJS: Adjective Superlative
                    if tag in ['JJ', 'JJR', 'JJS'] or word_lower in core_operational_words:
                        # Loại bỏ thêm vài tính từ khen ngợi sáo rỗng
                        if word_lower not in ["good", "nice", "great", "delicious", "amazing", "excellent"]:
                            intelligent_words.append(word_lower)
                
                # Biến danh sách từ đã lọc thành một chuỗi văn bản mới
                filtered_text = " ".join(intelligent_words)
                
                if filtered_text.strip():
                    # Vẽ WordCloud từ chuỗi đã được thuật toán lọc sạch dữ liệu rác
                    wordcloud = WordCloud(width=600, height=400, 
                                          background_color='white', 
                                          max_words=50,
                                          colormap='plasma',
                                          random_state=42).generate(filtered_text)
                    
                    fig_wc, ax_wc = plt.subplots(figsize=(6, 4))
                    ax_wc.imshow(wordcloud, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
                else:
                    st.warning("Not enough review data to perform intelligent lexical analysis.")

# ------------------------------------------------------------------
            # PHẦN HIỂN THỊ BẢNG DỮ LIỆU RATING & VIBE SCORE ĐÃ TỐI GIẢN
            # ------------------------------------------------------------------
            st.markdown("---") # Đường kẻ ngang phân cách cho đẹp
            
            st.subheader("📋 Table of Insights (Ascending order of Rating)")
            
            # 1. Lọc chỉ lấy 3 cột quan trọng: Restaurant, Rating, Vibe_Score
            # 2. Dùng .sort_values để ép sắp xếp theo Rating từ cao xuống thấp (ascending=False)
            display_df = final_df[['Restaurant','Link', 'Rating', 'Vibe Score']].sort_values(by='Rating', ascending=False)
            display_df['Vibe Rank'] = display_df['Vibe Score'].rank(ascending=False, method='min')
            # 3. Hiển thị bảng dạng DataFrame kéo giãn full màn hình cho dễ nhìn
            st.dataframe(display_df, use_container_width=True)