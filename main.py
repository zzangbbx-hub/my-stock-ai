import streamlit as st
import FinanceDataReader as fdr
from pykrx import stock
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
import os
import time

# 페이지 설정
st.set_page_config(page_title="단타 전투 머신 (Final)", layout="wide")

# 윈도우 폰트 깨짐 방지
if os.name == 'nt':
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False

# --- 1. 날짜 및 기초 함수 ---
def get_latest_business_day():
    kst_now = datetime.utcnow() + timedelta(hours=9)
    weekday = kst_now.weekday()
    if weekday == 5: target = kst_now - timedelta(days=1)
    elif weekday == 6: target = kst_now - timedelta(days=2)
    else:
        if kst_now.hour < 9:
            target = kst_now - timedelta(days=1)
            if target.weekday() >= 5: target = target - timedelta(days=(target.weekday() - 4))
        else: target = kst_now
    return target.strftime("%Y%m%d")

def get_date_str(date_str):
    d = datetime.strptime(date_str, "%Y%m%d")
    days = ["월", "화", "수", "목", "금", "토", "일"]
    return d.strftime(f"%m월 %d일 ({days[d.weekday()]})")

# --- 2. 데이터 수집 (시세 + 수급) ---
@st.cache_data(ttl=300)
def get_market_data(date_str):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f_k = executor.submit(stock.get_market_ohlcv_by_ticker, date_str, market="KOSPI")
        f_q = executor.submit(stock.get_market_ohlcv_by_
