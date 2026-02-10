import streamlit as st
import FinanceDataReader as fdr
from pykrx import stock
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
import os
import requests
import re

# 페이지 설정
st.set_page_config(page_title="단타 전투 머신 (Real-Time Pro)", layout="wide")

# 윈도우 폰트 깨짐 방지
if os.name == 'nt':
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False

# 매매 일지 초기화
if 'my_trade_log' not in st.session_state:
    st.session_state.my_trade_log = []

# --- 1. 날짜 및 기초 함수 ---
kst_now = datetime.utcnow() + timedelta(hours=9)
today_str = kst_now.strftime("%Y%m%d")
display_date = kst_now.strftime("%m월 %d일")

# --- 2. 데이터 수집 (시세) ---
@st.cache_data(ttl=300)
def get_market_data():
    target_date = today_str
    if kst_now.hour < 9:
        d = kst_now - timedelta(days=1)
        if d.weekday() == 6: d -= timedelta(days=2)
        elif d.weekday() == 5: d -= timedelta(days=1)
        target_date = d.strftime("%Y%m%d")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        f_k = executor.submit(stock.get_market_ohlcv_by_ticker, target_date, market="KOSPI")
        f_q = executor.submit(stock.get_market_ohlcv_by_ticker, target_date, market="KOSDAQ")
        df_k = f_k.result()
        df_q = f_q.result()
        
    df = pd.concat([df_k, df_q])
    if df.empty: return pd.DataFrame()
    
    df = df.sort_values(by='거래대금', ascending=False).head(100)
    
    ticker_list = df.index.tolist()
    name_map = {}
    
    def fetch_name(t): return t, stock.get_market_ticker_name(t)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(fetch_name, ticker_list)
        for t, name in results: name_map[t] = name
            
    df['종목명'] = df.index.map(name_map)
    df['거래대금(억)'] = (df['거래대금'] / 100000000).astype(int)
    
    prev = df['종가'] / (1 + df['등락률']/100)
    df['시가갭'] = ((df['시가'] - prev) / prev * 100).round(2)
    pivot = (df['고가'] + df['저가'] + df['종가']) / 3
    df['2차저항'] = (pivot + (df['고가'] - df['저가'])).astype(int)
    
    def get_sig(r):
        if r['종가'] >= r['2차저항']: return "🔥돌파"
        elif r['종가'] >= r['2차저항'] * 0.98: return "👀임박"
        else: return "-"
    df['신호'] = df.apply(get_sig, axis=1)

    return df

# --- [핵심] 네이버 금융 정밀 파싱 (빈 줄 건너뛰기) ---
@st.cache_data(ttl=600)
def get_naver_realtime_supply():
    url_foreign = "https://finance.naver.com/sise/sise_deal_rank.naver?investor_gubun=9000&type=buy"
    url_inst = "https://finance.naver.com/sise/sise_deal_rank.naver?investor_gubun=1000&type=buy"
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    def parse_table(url):
        try:
            res = requests.get(url, headers=headers)
            res.raise_for_status()
            # 네이버는 euc-kr 인코딩
            dfs = pd.read_html(res.text, encoding='euc-kr', attrs={"class": "type_2"})
            
            if not dfs: return pd.DataFrame()
            df = dfs[0]
            
            # [중요] 데이터 정제: '종목명'이 없는 행(빈 줄, 구분선)을 다 지움
            df = df.dropna(subset=['종목명'])
            
            # 컬럼 위치로 데이터 뽑기 (네이버 표 구조: 순위, 종목명, 현재가, 전일비, 등락률, 순매수량)
            # iloc[:, [1, 2, 4, 5]] -> 종목명, 현재가, 등락률, 순매수량
            result = df.iloc[:, [1, 2, 4, 5]].copy()
            result.columns = ['종목명', '현재가', '등락률', '수급량']
            
            # 데이터 클렌징 (글자, 쉼표, 기호 제거 후 숫자 변환)
            result['종목명'] = result['종목명'].astype(str).str.strip()
            
            def clean_float(x):
                try: return float(str(x).replace('%', '').replace('+', '').strip())
                except: return 0.0
                
            def clean_int(x):
                try: return int(str(x).replace(',', '').strip())
                except: return 0
            
            result['등락률'] = result['등락률'].apply(clean_float)
            result['수급량'] = result['수급량'].apply(clean
