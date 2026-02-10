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
    # 1. 시세 가져오기 (병렬)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f_k = executor.submit(stock.get_market_ohlcv_by_ticker, date_str, market="KOSPI")
        f_q = executor.submit(stock.get_market_ohlcv_by_ticker, date_str, market="KOSDAQ")
        df_k = f_k.result()
        df_q = f_q.result()
        
    df = pd.concat([df_k, df_q])
    df = df.sort_values(by='거래대금', ascending=False).head(50) # Top 50
    
    # 2. [수정됨] 종목명 가져오기 (오류 수정)
    ticker_list = df.index.tolist()
    name_map = {}
    
    def fetch_name(t): 
        return t, stock.get_market_ticker_name(t)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(fetch_name, ticker_list)
        for t, name in results:
            name_map[t] = name
            
    df['종목명'] = df.index.map(name_map)
    df['거래대금(억)'] = (df['거래대금'] / 100000000).astype(int)
    
    # 3. 지표 계산
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
            
            if i % 2 == 0: 
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
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.plot(df.index, df['Close'], label='Price', color='blue')
        ax1.plot(df.index, df['Close'].rolling(20).mean(), label='MA20', color='green', alpha=0.5)
        ax1.plot(df.index, df['Close'].rolling(60).mean(), label='MA60', color='gray', alpha=0.3)
        ax1.axhline(fibo_618, color='orange', linestyle='--', label='Fibo 0.618')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Analysis: {code}")
        
        ax2.plot(df.index, rsi, color='purple', label='RSI')
        ax2.axhline(70, color='red', linestyle='--')
        ax2.axhline(30, color='blue', linestyle='--')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig, rsi.iloc[-1], fibo_618, vol_ratio
    except: return None, 0, 0, 0

# --- 메인 UI ---
target_date = get_latest_business_day()
st.title(f"⚡ 단타 전투 머신 (Final)")
st.caption(f"기준: {get_date_str(target_date)}")

c1, c2, c3 = st.columns(3)
indices = {"KOSPI": "KS11", "KOSDAQ": "KQ11", "나스닥": "NQ=F"}
for i, (k, v) in enumerate(indices.items()):
    try:
        d = fdr.DataReader(v).iloc[-2:]
        val = d['Close'].iloc[-1]
        diff = val - d['Close'].iloc[-2]
        c1.metric(k, f"{val:.0f}", f"{diff:+.0f}") if i==0 else \
        c2.metric(k, f"{val:.0f}", f"{diff:+.0f}") if i==1 else \
        c3.metric(k, f"{val:.0f}", f"{diff:+.0f}")
    except: pass

st.divider()

# 데이터 로드
all_df = get_market_data(target_date)

# 탭 구성
tab1, tab2, tab3, tab4 = st.tabs(["🏆 스나이퍼", "💰 수급 포착", "🔮 정밀 분석", "📡 AI 스캐너"])

def color_surplus(val):
    color = 'red' if val > 0 else 'blue' if val < 0 else 'black'
    return f'color: {color}'

# [Tab 1] 스나이퍼
with tab1:
    if not all_df.empty:
        st.markdown("### 🔫 오늘의 대장주 (거래대금 상위)")
        
        t1 = all_df['거래대금(억)'] >= 200
        t2 = all_df['신호'].isin(["🔥돌파", "👀임박"])
        cand = all_df[t1 & t2].sort_values(by='등락률', ascending=False)
        
        if cand.empty:
            best = all_df.sort_values(by='등락률', ascending=False).iloc[0]
            st.warning("😓 돌파 종목 없음. 상승률 1위 표시.")
        else:
            best = cand.iloc[0]
            if "돌파" in best['신호']: st.success(f"🚀 **[{best['종목명']}]** 저항 돌파! 강력 매수")
            else: st.warning(f"👀 **[{best['종목명']}]** 돌파 임박! 관망")

        i1, i2, i3, i4 = st.columns(4)
        i1.metric("현재가", f"{best['종가']:,}")
        i2.metric("목표가", f"{best['2차저항']:,}")
        i3.metric("신호", best['신호'])
        i4.metric("대금", f"{best['거래대금(억)']}억")
        
        st.divider()
        st.caption("※ 거래대금 Top 50 리스트")
        
        st.dataframe(
            all_df[['종목명', '종가', '등락률', '신호', '거래대금(억)']]
            .head(20)
            .style
            .format({'종가': '{:,}', '거래대금(억)': '{:,}', '등락률': '{:.2f}%'})
            .map(color_surplus, subset=['등락률']), 
            hide_index=True, 
            use_container_width=True
        )

# [Tab 2] 수급 포착
with tab2:
    st.markdown("### 🦁 큰손들이 사는 종목 (외인/기관)")
    if st.button("💰 수급 데이터 불러오기 (클릭)"):
        with st.spinner("KRX에서 데이터를 가져오는 중..."):
            inv_df = get_investor_data(target_date)
            
            if not inv_df.empty:
                top_for = inv_df.sort_values('외국인', ascending=False).head(30)
                top_ins = inv_df.sort_values('기관합계', ascending=False).head(30)
                
                double_buy = pd.merge(top_for, top_ins, on=['종목명', '종가', '등락률'], suffixes=('_F', '_I'))
                double_buy = double_buy[['종목명', '종가', '등락률', '외국인', '기관합계']]
                
                if not double_buy.empty:
                    st.success(f"🚀 **쌍끌이 포착! (외인+기관 동시 매수)** - {len(double_buy)}종목")
                    st.dataframe(
                        double_buy.style.format({'종가': '{:,}', '외국인': '{:,}', '기관합계': '{:,}', '등락률': '{:.2f}%'})
                        .map(color_surplus, subset=['등락률']),
                        hide_index=True, use_container_width=True
                    )
                else:
                    st.info("오늘 뚜렷한 쌍끌이 종목이 없습니다.")
                
                st.divider()
                
                c_f, c_i = st.columns(2)
                with c_f:
                    st.subheader("🦁 외국인 순매수 Top 10")
                    st.dataframe(
                        inv_df.sort_values('외국인', ascending=False).head(10)[['종목명', '등락률', '외국인']]
                        .style.format({'등락률': '{:.2f}%', '외국인': '{:,}'})
                        .map(color_surplus, subset=['등락률']), 
                        hide_index=True
                    )
                with c_i:
                    st.subheader("🐯 기관 순매수 Top 10")
                    st.dataframe(
                        inv_df.sort_values('기관합계', ascending=False).head(10)[['종목명', '등락률', '기관합계']]
                        .style.format({'등락률': '{:.2f}%', '기관합계': '{:,}'})
                        .map(color_surplus, subset=['등락률']), 
                        hide_index=True
                    )
            else:
                st.error("데이터를 가져오지 못했습니다. 장 시작 전이거나 휴장일입니다.")

# [Tab 3] 정밀 분석
with tab3:
    if not all_df.empty:
        opts = ["선택"] + [f"{r['종목명']} ({r['종가']:,})" for i, r in all_df.head(50).iterrows()]
        sel_str = st.selectbox("종목 선택", opts)
        
        curr = 0
        code = ""
        name = ""
        if sel_str != "선택":
            name = sel_str.split(' (')[0]
            code = all_df[all_df['종목명'] == name].index[0]
            curr = all_df.loc[code]['종가']
            st.info(f"💰 현재가: **{curr:,}원**")

        mode = st.radio("입력 방식", ["주수", "금액"], horizontal=True)
        qty = 0
        amt = 0
        
        if mode == "주수":
            qty = st.number_input("주수", 1, 1000, 10)
            amt = curr * qty
            st.caption(f"필요 금액: {amt:,}원")
        else:
            in_money = st.number_input("금액", 10000, 10000000, 1000000)
            if curr > 0:
                qty = int(in_money // curr)
                amt = curr * qty
                st.caption(f"매수 가능: {qty:,}주")

        if st.button("⚖️ 판결 보기"):
            if sel_str != "선택" and qty > 0:
                fig, rsi, f618, vol = analyze_deep(code, name)
                if fig:
                    score = 0
                    reasons = []
                    if 40 <= rsi <= 60: score += 20; reasons.append("안정")
                    elif rsi < 30: score += 30; reasons.append("과매도")
                    elif rsi > 70: score -= 20; reasons.append("과매수")
                    if vol > 150: score += 30; reasons.append("거래폭발")
                    
                    st.divider()
                    if score >= 70: st.success(f"✅ 진입 승인 ({score}점)")
                    elif score >= 50: st.warning(f"⚠️ 보류 ({score}점)")
                    else: st.error(f"❌ 위험 ({score}점)")
                    st.caption(f"이유: {reasons}")
                    st.pyplot(fig)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.info(f"매수: {qty:,}주")
                    c2.success(f"익절: {int(curr*1.03):,}")
                    c3.error(f"손절: {int(curr*0.98):,}")

# [Tab 4] AI 스캐너
with tab4:
    if not all_df.empty:
        st.subheader("📡 실시간 패턴 스캐너")
        if st.button("🚀 스캔 시작"):
            scan_codes = all_df.index.tolist()
            results = run_scanners(scan_codes)
            
            if results:
                st.toast(f"🔔 {len(results)}개 포착!", icon="🎉")
                for res in results:
                    name = all_df.loc[res['code']]['종목명']
                    price = res['price']
                    tags = res['특이사항']
                    with st.container():
                        st.write(f"**[{name}]** ({int(price):,}원)")
                        st.info(f"👉 {tags}")
                        if "안전빵" in tags: st.caption("└ 🛡️ **안전빵:** 60일선 위+20일선 지지")
                        st.divider()
            else: st.info("없음")
