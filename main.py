import streamlit as st
import FinanceDataReader as fdr
from pykrx import stock
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
import os
import numpy as np

# 페이지 설정
st.set_page_config(page_title="단타 전투 머신 (Chart Master)", layout="wide")

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
    # 장 시작 전(09시)이면 어제 날짜 기준
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
    
    # 거래대금 상위 150개로 넉넉하게
    df = df.sort_values(by='거래대금', ascending=False).head(150)
    
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

# --- 3. 통합 스캐너 (목록용) ---
def run_all_scanners(code_list):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(code_list)
    
    def analyze_one(code):
        try:
            df = fdr.DataReader(code).tail(60)
            if len(df) < 30: return None
            
            curr = df.iloc[-1]
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            
            # 간단 필터링
            tags = []
            score = 0
            
            if curr['Close'] > ma20: 
                tags.append("추세양호")
                score += 10
            
            vol_ratio = (curr['Volume'] / df['Volume'].rolling(5).mean().iloc[-1]) * 100
            if vol_ratio >= 200:
                tags.append("거래폭발")
                score += 20
                
            if tags:
                return {'code': code, 'tags': ", ".join(tags), 'price': curr['Close'], 'score': score}
            return None
        except: return None
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_one, code): code for code in code_list}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            if res: results.append(res)
            if i % 10 == 0:
                prog = (i + 1) / total
                progress_bar.progress(prog)
                status_text.caption(f"⚡ 스캔 중... ({i+1}/{total})")
                
    status_text.empty()
    progress_bar.empty()
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

# --- [핵심] 정밀 진단 AI 엔진 ---
def analyze_deep_pro(code, name):
    try:
        # 넉넉하게 1년치 데이터
        df = fdr.DataReader(code).tail(240)
        if len(df) < 60:
            return None, 0, [], "데이터 부족"

        # 1. 보조지표 계산
        c = df['Close']
        # 이동평균선
        ma5 = c.rolling(5).mean()
        ma20 = c.rolling(20).mean()
        ma60 = c.rolling(60).mean()
        ma120 = c.rolling(120).mean()
        
        # 볼린저밴드
        std = c.rolling(20).std()
        upper = ma20 + (std * 2)
        lower = ma20 - (std * 2)
        bandwidth = (upper - lower) / ma20
        
        # RSI
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = c.ewm(span=12, adjust=False).mean()
        exp26 = c.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # 현재 값
        curr_price = c.iloc[-1]
        curr_rsi = rsi.iloc[-1]
        curr_vol = df['Volume'].iloc[-1]
        avg_vol = df['Volume'].tail(5).mean()
        
        # 2. AI 점수 채점 (100점 만점)
        score = 0
        reasons = []
        
        # (1) 추세 점수 (40점)
        if curr_price > ma20.iloc[-1]:
            score += 20
            reasons.append("✅ 단기 상승 추세 (20일선 위)")
        if curr_price > ma60.iloc[-1]:
            score += 10
            reasons.append("✅ 중기 추세 살아있음 (60일선 위)")
        if ma5.iloc[-1] > ma20.iloc[-1]: # 정배열 초입
            score += 10
            reasons.append("✅ 5일>20일 골든크로스 구간")
            
        # (2) 모멘텀/보조지표 (30점)
        if 40 <= curr_rsi <= 70:
            score += 10
            reasons.append("✅ RSI 안정적 (과열 아님)")
        elif curr_rsi < 30:
            score += 20
            reasons.append("🔥 RSI 과매도 (기술적 반등 위치)")
            
        if macd.iloc[-1] > signal.iloc[-1]:
            score += 10
            reasons.append("✅ MACD 매수 신호 유지")
            
        # (3) 거래량/패턴 (30점)
        if curr_vol > avg_vol * 1.5:
            score += 20
            reasons.append("💪 거래량 실림 (수급 유입)")
        
        if bandwidth.iloc[-1] < 0.15: # 밴드폭이 좁아짐 (힘 응축)
            score += 10
            reasons.append("⚡ 볼린저밴드 수축 (변동성 폭발 임박)")

        # 3. 차트 그리기
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df.index, c, label='Price', color='black')
        ax1.plot(df.index, ma20, label='MA20', color='green', alpha=0.7)
        ax1.plot(df.index, ma60, label='MA60', color='orange', alpha=0.7)
        ax1.plot(df.index, upper, color='gray', linestyle='--', alpha=0.3)
        ax1.plot(df.index, lower, color='gray', linestyle='--', alpha=0.3)
        ax1.set_title(f"Analysis: {name} ({code})")
        ax1.legend()
        ax1.grid(True, alpha=0.2)
        
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(df.index, rsi, label='RSI', color='purple')
        ax2.axhline(30, color='blue', linestyle='--')
        ax2.axhline(70, color='red', linestyle='--')
        ax2.legend()
        ax2.grid(True, alpha=0.2)
        
        plt.tight_layout()
        
        return fig, score, reasons, curr_price
        
    except Exception as e:
        return None, 0, [], 0

# --- 메인 UI ---
st.title(f"⚔️ 단타 전투 머신 (Chart Master)")
st.caption(f"기준: {display_date}")

c1, c2, c3 = st.columns(3)
indices = {"KOSPI": "KS11", "KOSDAQ": "KQ11", "나스닥": "NQ=F"}
for i, (k, v) in enumerate(indices.items()):
    try:
        d = fdr.DataReader(v).tail(5)
        if len(d) >= 2:
            val = d['Close'].iloc[-1]
            diff = val - d['Close'].iloc[-2]
            c1.metric(k, f"{val:.0f}", f"{diff:+.0f}") if i==0 else \
            c2.metric(k, f"{val:.0f}", f"{diff:+.0f}") if i==1 else \
            c3.metric(k, f"{val:.0f}", f"{diff:+.0f}")
    except: pass

st.divider()

all_df = get_market_data()

if all_df.empty:
    st.error("⚠️ 데이터 로드 실패. 잠시 후 다시 시도하세요.")
else:
    # 탭 구성 (수급 탭 삭제 -> 정밀분석 강화)
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 스나이퍼", "📡 통합 스캐너", "🩺 정밀 분석(진입판결)", "📝 매매 일지"
    ])

    def color_surplus(val):
        if isinstance(val, str): return 'color: black'
        color = 'red' if val > 0 else 'blue' if val < 0 else 'black'
        return f'color: {color}'

    # [Tab 1] 스나이퍼
    with tab1:
        st.markdown("### 🔫 오늘의 대장주")
        t1 = all_df['거래대금(억)'] >= 200
        t2 = all_df['신호'].isin(["🔥돌파", "👀임박"])
        cand = all_df[t1 & t2].sort_values(by='등락률', ascending=False)
        
        if cand.empty:
            best = all_df.sort_values(by='등락률', ascending=False).iloc[0]
            st.warning("😓 돌파 종목 없음. 상승률 1위 표시.")
        else:
            best = cand.iloc[0]
            st.success(f"🚀 **[{best['종목명']}]** 포착! 대금 {best['거래대금(억)']}억")

        i1, i2, i3, i4 = st.columns(4)
        i1.metric("현재가", f"{best['종가']:,}")
        i2.metric("목표가", f"{best['2차저항']:,}")
        i3.metric("신호", best['신호'])
        i4.metric("대금", f"{best['거래대금(억)']}억")
        
        st.divider()
        st.dataframe(
            all_df[['종목명', '종가', '등락률', '신호', '거래대금(억)']].head(50).style
            .format({'종가': '{:,}', '거래대금(억)': '{:,}', '등락률': '{:.2f}%'})
            .map(color_surplus, subset=['등락률']), 
            hide_index=True, use_container_width=True
        )

    # [Tab 2] 통합 스캐너
    with tab2:
        st.markdown("### 📡 AI 패턴 정밀 스캔")
        if st.button("🚀 스캔 시작"):
            scan_codes = all_df.index.tolist()
            results = run_all_scanners(scan_codes)
            
            if results:
                st.toast(f"🔔 {len(results)}개 포착!", icon="🥇")
                for res in results:
                    name = all_df.loc[res['code']]['종목명']
                    price = res['price']
                    tags = res['tags']
                    score = res['score']
                    
                    st.markdown(f"**[{name}]** `{int(price):,}원`")
                    st.info(f"👉 {tags}")
                    st.divider()
            else: st.info("특이 패턴 종목이 없습니다.")

    # [Tab 3] 정밀 분석 (수급 포기 -> 정밀 진단 기능 강화)
    with tab3:
        st.markdown("### 🩺 AI 주치의 정밀 진단")
        st.caption("※ 종목을 선택하고 **'진단 시작'**을 누르면, AI가 차트를 분석해 **진입 여부**를 알려줍니다.")
        
        # 종목 선택창
        opts = ["선택"] + [f"{r['종목명']} ({r['종가']:,})" for i, r in all_df.head(100).iterrows()]
        sel = st.selectbox("진단할 종목 선택", opts)
        
        if sel != "선택":
            name = sel.split(' (')[0]
            code = all_df[all_df['종목명'] == name].index[0]
            
            # 진단 버튼
            if st.button(f"🔍 '{name}' 정밀 진단 시작", type="primary"):
                with st.spinner("AI가 차트를 뜯어보는 중입니다..."):
                    fig, score, reasons, curr_price = analyze_deep_pro(code, name)
                    
                    if fig:
                        st.divider()
                        
                        # 결과 표시 (점수판)
                        c1, c2 = st.columns([2, 3])
                        
                        with c1:
                            st.markdown("#### 🧑‍⚖️ AI 판결문")
                            if score >= 80:
                                st.error(f"👑 **S급 (강력 매수)**\n\n점수: **{score}점**")
                                st.markdown("👉 **지금 당장 봐야 할 종목!** 추세/거래량 완벽.")
                            elif score >= 60:
                                st.warning(f"🥇 **A급 (매수 고려)**\n\n점수: **{score}점**")
                                st.markdown("👉 **좋습니다.** 분할 매수로 접근하세요.")
                            elif score >= 40:
                                st.info(f"🥈 **B급 (관심)**\n\n점수: **{score}점**")
                                st.markdown("👉 나쁘진 않은데, 확실한 신호를 기다리세요.")
                            else:
                                st.markdown(f"💀 **진입 금지**\n\n점수: **{score}점**")
                                st.markdown("👉 **위험합니다.** 추세가 꺾였거나 거래량이 없습니다.")
                            
                            st.markdown("---")
                            st.markdown("**[채점 사유]**")
                            for r in reasons:
                                st.write(r)
                                
                        with c2:
                            st.pyplot(fig)
                            
                        # 전략 제안
                        st.success(f"**[전략]** 현재가 {int(curr_price):,}원 기준")
                        col_a, col_b = st.columns(2)
                        col_a.info(f"🛑 손절가: {int(curr_price * 0.97):,}원 (-3%)")
                        col_b.error(f"💰 목표가: {int(curr_price * 1.05):,}원 (+5%)")
                        
                    else:
                        st.error("데이터 부족으로 분석할 수 없습니다.")

    # [Tab 4] 매매 일지
    with tab4:
        st.markdown("### 📝 매매 복기장")
        with st.form("trade_form"):
            c1, c2, c3 = st.columns(3)
            t_name = c1.text_input("종목명")
            t_buy = c2.number_input("매수가", 0)
            t_sell = c3.number_input("매도가", 0)
            memo = st.text_area("메모")
            if st.form_submit_button("기록"):
                p = (t_sell - t_buy)*100/t_buy if t_buy > 0 else 0
                st.session_state.my_trade_log.append({
                    "날짜": datetime.now().strftime("%Y-%m-%d"),
                    "종목": t_name, "수익률": f"{p:.2f}%", "메모": memo
                })
                st.success("저장!")
        if st.session_state.my_trade_log:
            st.dataframe(pd.DataFrame(st.session_state.my_trade_log), use_container_width=True)
