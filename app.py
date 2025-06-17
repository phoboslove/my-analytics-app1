# ==============================================================================
# ФИНАЛЬНЫЙ КОД "АНАЛИТИЧЕСКОГО ДВИЖКА" v2.2 (ИСПРАВЛЕННЫЙ И ПОЛНЫЙ)
# ==============================================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from adjustText import adjust_text
import warnings

# Игнорируем предупреждения для более чистого вывода
warnings.filterwarnings('ignore')


# --- НАСТРОЙКА ИНТЕРФЕЙСА ПРИЛОЖЕНИЯ ---
st.set_page_config(page_title="AI Бизнес-Аналитик", page_icon="📈", layout="wide")

st.title("👨‍💻 Ваш Aetheris Бизнес-Аналитик")
st.write("Загрузите отчет о продажах в формате Excel или CSV, чтобы найти скрытые точки роста за несколько минут.")

uploaded_file = st.file_uploader("Выберите файл с продажами...", type=['csv', 'xlsx'])

# --- АНАЛИЗ ЗАПУСКАЕТСЯ ТОЛЬКО ПОСЛЕ ЗАГРУЗКИ ФАЙЛА ---
if uploaded_file is not None:
    
    with st.spinner('Анализирую данные... Это может занять до минуты...'):
        try:
            # Автоматически определяем тип файла и читаем его
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            df['OrderDate'] = pd.to_datetime(df['OrderDate'])
            st.success(f"✔️ Файл '{uploaded_file.name}' успешно загружен. Найдено {len(df)} строк.")

            # --- НАЧАЛО АНАЛИТИЧЕСКОГО БЛОКА ---

            # --- ОБЩИЕ KPI ---
            st.header("Ключевые показатели бизнеса 📊")
            total_revenue = df['Price'].sum()
            number_of_orders = df['OrderID'].nunique()
            unique_customers = df['ClientID'].nunique() if 'ClientID' in df.columns else 'н/д'
            average_check = total_revenue / number_of_orders if number_of_orders > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Общая выручка", f"{total_revenue:,.0f} тг".replace(',', ' '))
            col2.metric("Количество заказов", f"{number_of_orders}")
            col3.metric("Средний чек", f"{average_check:,.0f} тг".replace(',', ' '))
            col4.metric("Уникальные клиенты", f"{unique_customers}")

            # --- ИСПРАВЛЕННЫЙ БЛОК: АНАЛИЗ КЛИЕНТОВ ---
            if 'ClientID' in df.columns:
                st.header("Анализ по клиентам 🏆")
                customer_spending = df.groupby('ClientID')['Price'].sum().sort_values(ascending=False)
                
                st.write("Топ-10 клиентов по сумме трат:")
                st.dataframe(customer_spending.head(10))
                
                # График по клиентам
                plt.figure(figsize=(12, 6))
                customer_spending.head(10).plot(kind='bar', color='skyblue')
                plt.title('Топ-10 клиентов по сумме трат')
                plt.ylabel('Сумма трат (тенге)')
                plt.xlabel('ID Клиента')
                plt.xticks(rotation=45)
                st.pyplot()


            # --- ИСПРАВЛЕННЫЙ БЛОК: АНАЛИЗ ПО ВРЕМЕНИ ---
            st.header("Анализ по времени 🕒")
            daily_sales = df.groupby(df['OrderDate'].dt.date)['Price'].sum()
            
            st.write("Динамика выручки по дням:")
            st.line_chart(daily_sales) # Используем встроенный график Streamlit, он интерактивный!
            
            
            # --- МЕНЮ-ИНЖИНИРИНГ ---
            st.header("Матрица Меню-Инжиниринга 🍽️")
            menu_analysis = df.groupby('Dish').agg(Popularity=('Dish', 'count'), Revenue=('Price', 'sum'))
            avg_popularity = menu_analysis['Popularity'].mean()
            avg_revenue = menu_analysis['Revenue'].mean()
            
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.scatter(menu_analysis['Popularity'], menu_analysis['Revenue'], s=120, color='grey', alpha=0.6)
            texts = [ax.text(row['Popularity'], row['Revenue'], index, fontsize=10) for index, row in menu_analysis.iterrows()]
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
            ax.axvline(x=avg_popularity, color='r', linestyle='--')
            ax.axhline(y=avg_revenue, color='r', linestyle='--')
            ax.set_title('Матрица Меню-Инжиниринга', fontsize=16)
            ax.set_xlabel('Популярность (Количество продаж)')
            ax.set_ylabel('Выручка (тенге)')
            ax.grid(True)
            st.pyplot(fig)


            # --- АНАЛИЗ "ИДЕАЛЬНЫХ ПАР" ---
            st.header("Анализ 'Идеальных пар' 🧺")
            basket = (df.groupby(['OrderID', 'Dish'])['Price'].count().unstack().reset_index().fillna(0).set_index('OrderID'))
            def encode_units(x): return 1 if x > 0 else 0
            basket_sets = basket.apply(lambda x: x.map(encode_units))
            
            frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)
            if not frequent_itemsets.empty:
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
                if not rules.empty:
                    st.write("Найденные правила 'Если... то...':")
                    st.dataframe(rules.sort_values(by=['lift', 'confidence'], ascending=False)[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
                else:
                    st.info("Сильных 'связок' между товарами не найдено на основе текущих настроек.")
            else:
                st.info("Популярных наборов товаров не найдено на основе текущих настроек.")

        except Exception as e:
            st.error(f"Произошла ошибка при анализе файла. Убедитесь, что он имеет правильный формат и нужные колонки (OrderID, OrderDate, Dish, Price). Ошибка: {e}")