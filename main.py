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
        f_q = executor.submit(stock.get_market_ohlcv_by_ticker, date_str, market="KOSDAQ")
        df_k = f_k.result()
        df_q = f_q.result()
        
    df = pd.concat([df_k, df_q])
    df = df.sort_values(by='거래대금', ascending=False).head(50) # Top 50
    
    ticker_list = df.index.tolist()
    name_map = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(stock.get_market_ticker_name, t): t for t in ticker_list}
        for future in concurrent.futures.as_completed(futures):
            name_map[future.result()[0]] = future.result()[1]
            
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

@st.cache_data(ttl=600)
def get_investor_data(date_str):
    try:
        df = stock.get_market_net_purchases_of_equities_by_ticker(date_str, "ALL")
        df = df[['종목명', '종가', '등락률', '외국인', '기관합계']]
        return df.sort_values(by='외국인', ascending=False)
    except: return pd.DataFrame()

# --- 3. 분석 함수들 ---
def run_scanners(code_list):
    results = []
    
    # [NEW] 로딩바 UI 요소 생성
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(code_list)
    
    def analyze_one(code):
        try:
            df = fdr.DataReader(code).tail(120)
            if len(df) < 60: return None
            
            c = df['Close']
            ma20 = c.rolling(20).mean()
            ma60 = c.rolling(60).mean()
            std = c.rolling(20).std()
            upper = ma20 + (std * 2)
            lower = ma20 - (std * 2)
            band_w = (upper - lower) / ma20
            
            curr = df.iloc[-1]
            prev1 = df.iloc[-2]
            tags = []
            
            if len(df) >= 3:
                p2 = df.iloc[-3]
                if p2['Close'] > p2['Open'] and prev1['Close'] < prev1['Open'] and curr['Close'] > curr['Open']:
                    tags.append("양음양")
            if band_w.iloc[-1] < 0.15: tags.append("용수철")
            if curr['Close'] > ma60.iloc[-1] and abs(curr['Close'] - ma20.iloc[-1])/curr['Close'] < 0.03:
                tags.append("안전빵")
                
            if tags: return {'code': code, '특이사항': ", ".join(tags), 'price': curr['Close']}
            return None
        except: return None
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(analyze_one, code): code for code in code_list}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            if res: results.append(res)
            
            # [NEW] 로딩바 업데이트
            if i % 2 == 0: # 너무 자주 갱신하면 느려지니 2번에 1번만
                prog = (i + 1) / total
                progress_bar.progress(prog)
                status_text.caption(f"⚡ AI 분석 중... ({i+1}/{total})")
                
    status_text.empty()
    progress_bar.empty()
    return results

def analyze_deep(code, name):
    try:
        df = fdr.DataReader(code).tail(120)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        high_p = df['High'].tail(60).max()
        low_p = df['Low'].tail(60).min()
        fibo_618 = high_p - ((high_p - low_p) * 0.618)
        
        vol_ratio = (df['Volume'].iloc[-1] / df['Volume'].tail(5).mean()) * 100
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios':
