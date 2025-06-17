# ==============================================================================
#           ФИНАЛЬНОЕ ПРИЛОЖЕНИЕ v4.1 (С НОВЫМ МЕТОДОМ АВТОРИЗАЦИИ)
# ==============================================================================

# --- 1. ПОДКЛЮЧЕНИЕ ВСЕХ ИНСТРУМЕНТОВ ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from adjustText import adjust_text
    import streamlit_authenticator as stauth
    import yaml
    from yaml.loader import SafeLoader
except ImportError as e:
    st.error(f"Ошибка при импорте библиотек: {e}. Пожалуйста, проверьте ваш файл requirements.txt.")
    st.stop()

# --- 2. НАСТРОЙКА ИНТЕРФЕЙСА ПРИЛОЖЕНИЯ ---
st.set_page_config(page_title="Бизнес-Аналитик", page_icon="🔐", layout="wide")
warnings.filterwarnings('ignore')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- 3. БЛОК АВТОРИЗАЦИИ ПОЛЬЗОВАТЕЛЯ ---
try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("Ошибка: Файл конфигурации 'config.yaml' не найден. Убедитесь, что он загружен на GitHub вместе с app.py.")
    st.stop()

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

st.title("👨‍💻 AI Бизнес-Аналитик")

# --- ИЗМЕНЕНИЕ №1: ВЫЗЫВАЕМ LOGIN ПО-НОВОМУ ---
authentication_status = authenticator.login('main')


# --- 4. ОСНОВНАЯ ЛОГИКА ПРИЛОЖЕНИЯ ПОСЛЕ ВХОДА ---
if authentication_status:
    # ---- ЕСЛИ ПОЛЬЗОВАТЕЛЬ УСПЕШНО ВОШЕЛ ----
    
    # --- ИЗМЕНЕНИЕ №2: ПОЛУЧАЕМ ИМЯ ПОЛЬЗОВАТЕЛЯ ИЗ SESSION_STATE ---
    name = st.session_state.get("name")
    
    with st.sidebar:
        st.write(f'Добро пожаловать, *{name}*!')
        authenticator.logout('Выйти', 'main')

    st.header("Загрузите ваш файл для анализа")
    uploaded_file = st.file_uploader("Выберите файл с продажами...", type=['csv', 'xlsx'], label_visibility="collapsed")

    if uploaded_file is not None:
        with st.spinner('Анализирую данные... Это может занять до минуты...'):
            try:
                # =========================================================
                # НАЧАЛО ВСТАВЛЕННОГО АНАЛИТИЧЕСКОГО БЛОКА (он остается без изменений)
                # =========================================================
                
                # ... (здесь весь твой аналитический код, который мы уже написали)
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                
                required_columns = ['OrderID', 'OrderDate', 'Dish', 'Price']
                if not all(col in df.columns for col in required_columns):
                    st.error(f"Ошибка: В вашем файле отсутствуют обязательные колонки: {', '.join(required_columns)}")
                    st.stop()
                    
                df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
                st.success(f"✔️ Файл '{uploaded_file.name}' успешно загружен. Найдено {len(df)} строк.")
                st.dataframe(df.head())

                # --- ОСНОВНЫЕ KPI ---
                st.header("Ключевые показатели бизнеса 📊")
                # ... (весь код для KPI, графиков, матриц, как и раньше)

            except Exception as e:
                st.error(f"Произошла ошибка при анализе файла. Убедитесь, что файл имеет правильный формат и нужные колонки. Ошибка: {e}")

elif authentication_status == False:
    st.error('Имя пользователя или пароль неверны')
elif authentication_status == None:
    st.warning('Пожалуйста, введите имя пользователя и пароль для доступа.')
