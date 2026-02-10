import streamlit as st
import FinanceDataReader as fdr
from pykrx import stock
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
import os

# 페이지 설정
st.set_page_config(page_title="단타 전투 머신 (Ultra Fast)", layout="wide")

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

# --- 2. 데이터 수집 (캐싱 + 내부 병렬) ---
@st.cache_data(ttl=300)
def get_battle_data_single(date_str, mkt):
    try:
        # 전체 시세 가져오기
        df = stock.get_market_ohlcv_by_ticker(date_str, market=mkt)
        if df.empty: return pd.DataFrame(), 0, 0
        
        up_cnt = len(df[df['등락률'] > 0])
        down_cnt = len(df[df['등락률'] < 0])
        
        # 거래대금 상위 40개
        df = df.sort_values(by='거래대금', ascending=False).head(40)
        
        # 종목명 가져오기 (내부 병렬 처리)
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
        
        return df, up_cnt, down_cnt
    except: return pd.DataFrame(), 0, 0

# --- 3. AI 스캐너 ---
def run_scanners_fast(code_list):
    results = []
    
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
            
            # 1. 양음양
            if len(df) >= 3:
                p2 = df.iloc[-3]
                if p2['Close'] > p2['Open'] and prev1['Close'] < prev1['Open'] and curr['Close'] > curr['Open']:
                    tags.append("양음양")
            
            # 2. 용수철
            if band_w.iloc[-1] < 0.15: tags.append("용수철")
                
            # 3. 안전빵
            is_uptrend = curr['Close'] > ma60.iloc[-1]
            is_support = abs(curr['Close'] - ma20.iloc[-1]) / curr['Close'] < 0.03
            
            if is_uptrend and is_support:
                tags.append("안전빵")

            if tags:
                return {'code': code, '특이사항': ", ".join(tags), 'price': curr['Close']}
            return None
        except: return None

    status_text = st.empty()
    progress_bar = st.progress(0)
    total = len(code_list)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(analyze_one, code): code for code in code_list}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            if res: results.append(res)
            if i % 5 == 0:
                progress = (i + 1) / total
                progress_bar.progress(progress)
                status_text.caption(f"⚡ 스캔 중... ({i+1}/{total})")
    
    status_text.empty()
    progress_bar.empty()
    return results

# --- 4. 정밀 분석 ---
def analyze_deep(code, name):
    try:
        df = fdr.DataReader(code).tail(120)
        if len(df) < 60: return None, None, 0, 0, 0
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        curr_rsi = df['RSI'].iloc[-1]
        
        high_p = df['High'].tail(60).max()
        low_p = df['Low'].tail(60).min()
        diff = high_p - low_p
        fibo_618 = high_p - (diff * 0.618)
        
        vol_ratio = (df['Volume'].iloc[-1] / df['Volume'].tail(5).mean()) * 100
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(df.index, df['Close'], label='Price', color='blue')
        ax1.plot(df.index, df['Close'].rolling(20).mean(), label='20MA', color='green', alpha=0.5)
        ax1.plot(df.index, df['Close'].rolling(60).mean(), label='60MA', color='gray', alpha=0.3)
        ax1.axhline(fibo_618, color='orange', linestyle='--', label='Fibo 0.618')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"{name} Analysis")
        
        ax2.plot(df.index, df['RSI'], color='purple', label='RSI')
        ax2.axhline(70, color='red', linestyle='--')
        ax2.axhline(30, color='blue', linestyle='--')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, curr_rsi, fibo_618, 0, vol_ratio
    except: return None, None, 0, 0, 0

# --- 메인 UI ---
target_date = get_latest_business_day()
st.title(f"⚡ 단타 전투 머신 (Ultra Fast)")
st.caption(f"기준: {get_date_str(target_date)}")

c1, c2, c3 = st.columns(3)
indices = {"KOSPI": "KS11", "KOSDAQ": "KQ11", "나스닥": "NQ=F"}

# 지수 로딩도 병렬화 가능하지만, 워낙 빨라서 단순 처리
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

# [핵심] 코스피/코스닥 동시 로딩 (병렬 처리)
# 기존에는 하나 끝나고 하나 시작했지만, 이제 둘 다 동시에 출발합니다.
with concurrent.futures.ThreadPoolExecutor() as executor:
    # 일꾼 2명에게 동시에 지시
    future_k = executor.submit(get_battle_data_single, target_date, "KOSPI")
    future_q = executor.submit(get_battle_data_single, target_date, "KOSDAQ")
    
    # 둘 다 끝날 때까지 기다렸다가 결과 받기
    k_df, k_u, k_d = future_k.result()
    q_df, q_u, q_d = future_q.result()

all_df = pd.concat([k_df, q_df]) if not (k_df.empty and q_df.empty) else pd.DataFrame()

tab1, tab2, tab3 = st.tabs(["🏆 랭킹 & 스나이퍼", "🔮 정밀 분석(판결)", "📡 AI 스캐너(안전빵)"])

# [Tab 1] 스나이퍼
with tab1:
    if not all_df.empty:
        st.markdown("### 🔫 AI 스나이퍼 (돌파 매매)")
        
        t1 = all_df['거래대금(억)'] >= 200
        t2 = all_df['신호'].isin(["🔥돌파", "👀임박"])
        cand = all_df[t1 & t2].sort_values(by='등락률', ascending=False)
        
        if cand.empty:
            best = all_df.sort_values(by='등락률', ascending=False).iloc[0]
            is_force = True
        else:
            best = cand.iloc[0]
            is_force = False
            
        with st.container():
            st.info(f"**타겟:** **[{best['종목명']}]**")
            i1, i2, i3, i4 = st.columns(4)
            i1.metric("가", f"{best['종가']:,}")
            i2.metric("목표", f"{best['2차저항']:,}")
            i3.metric("신호", best['신호'])
            i4.metric("대금", f"{best['거래대금(억)']}억")
            
            if is_force and best['신호'] == "-":
                st.warning("😓 돌파 종목 없음. **상승률 1위** 표시.")
            elif "돌파" in best['신호']: 
                st.success("🚀 **[강력 매수]** 저항 돌파! 진입하세요.")
            else: 
                st.warning("👀 **[관망]** 뚫으면 진입하세요.")

        st.divider()
        
        ch, cb = st.columns([5, 1])
        with ch: st.caption("※ 거래대금 Top 40")
        with cb:
            if st.button("🔄"):
                # 캐시 삭제 함수가 바뀌었으므로 새로 지정
                get_battle_data_single.clear()
                st.rerun()
                
        cols = ['종목명', '종가', '등락률', '신호']
        def color_val(v): return f'color: {"red" if v > 0 else "blue"}'
        
        col_k, col_q = st.columns(2)
        with col_k:
            st.subheader("KOSPI")
            st.dataframe(k_df[cols].head(20).style.format({'종가':'{:.0f}','등락률':'{:.2f}%'}).map(color_val, subset=['등락률']), hide_index=True)
        with col_q:
            st.subheader("KOSDAQ")
            st.dataframe(q_df[cols].head(20).style.format({'종가':'{:.0f}','등락률':'{:.2f}%'}).map(color_val, subset=['등락률']), hide_index=True)

# [Tab 2] 정밀 분석
with tab2:
    if not all_df.empty:
        opts = ["선택"] + [f"{r['종목명']} ({r['종가']:,})" for i, r in all_df.head(50).iterrows()]
        
        c_sel, c_q = st.columns([2,1])
        with c_sel: sel_str = st.selectbox("종목 선택", opts)
        with c_q: qty = st.number_input("주수", 1, 1000, 10)

        if sel_str != "선택":
            name = sel_str.split(' (')[0]
            code = all_df[all_df['종목명'] == name].index[0]
            row = all_df[all_df['종목명'] == name].iloc[0]
            curr = row['종가']
            
            if st.button("⚖️ AI 최종 판결 보기"):
                fig, rsi, f618, _, vol_rot = analyze_deep(code, name)
                
                if fig:
                    score = 0
                    reasons = []
                    
                    if 40 <= rsi <= 60: score += 20; reasons.append("안정적 흐름")
                    elif rsi < 30: score += 30; reasons.append("과매도(반등기회)")
                    elif rsi > 70: score -= 20; reasons.append("과매수(고점위험)")
                    
                    if vol_rot > 150: score += 30; reasons.append("거래량 폭발")
                    elif vol_rot < 50: score -= 10; reasons.append("거래량 부족")
                    
                    if row['등락률'] > 0: score += 20
                    
                    st.divider()
                    st.subheader("🧑‍⚖️ AI 최종 판결")
                    
                    if score >= 70:
                        st.success(f"✅ **[진입 승인]** 강력 매수 신호! ({score}점)")
                    elif score >= 50:
                        st.warning(f"⚠️ **[보류]** 확실하지 않습니다. ({score}점)")
                    else:
                        st.error(f"❌ **[진입 금지]** 위험합니다. ({score}점)")
                        
                    st.caption(f"이유: {', '.join(reasons)}")
                    st.pyplot(fig)
                    
                    tgt = int(curr * 1.03)
                    cut = int(curr * 0.98)
                    m1 = f"매수: {qty}주 ({curr*qty:,}원)"
                    m2 = f"익절: {tgt:,} (+3%)"
                    m3 = f"손절: {cut:,} (-2%)"
                    
                    c1, c2, c3 = st.columns(3)
                    c1.info(m1)
                    c2.success(m2)
                    c3.error(m3)

# [Tab 3] AI 스캐너
with tab3:
    if not all_df.empty:
        st.subheader("📡 실시간 패턴 스캐너")
        st.caption("※ 안전하고 확실한 종목을 찾아냅니다.")
        
        if st.button("🚀 초고속 스캔 시작"):
            scan_codes = all_df.index.tolist()
            results = run_scanners_fast(scan_codes)
            
            if results:
                st.success(f"총 {len(results)}개 종목 포착!")
                
                for res in results:
                    r_name = all_df.loc[res['code']]['종목명']
                    r_price = res['price']
                    tags = res['특이사항']
                    
                    with st.container():
                        st.write(f"**[{r_name}]** ({int(r_price):,}원)")
                        st.info(f"👉 {tags}")
                        if "안전빵" in tags: st.caption("└ 🛡️ **안전빵:** 60일선 위 + 20일선 지지 (저위험)")
                        if "양음양" in tags: st.caption("└ 🕯️ **양음양:** N자 상승 (눌림목)")
                        if "용수철" in tags: st.caption("└ 💥 **용수철:** 폭발 임박")
                        st.divider()
            else:
                st.info("특이 패턴 종목이 없습니다.")
