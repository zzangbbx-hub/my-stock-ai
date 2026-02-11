import streamlit as st
import FinanceDataReader as fdr
from pykrx import stock
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
import os
import requests
import xml.etree.ElementTree as ET

# 페이지 설정
st.set_page_config(page_title="단타 전투 머신 (Google News)", layout="wide")

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
    
    # 거래대금 상위 150개
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

# --- [핵심] 뉴스 엔진 교체 (Google News RSS) ---
# 구글은 봇 차단을 거의 하지 않으며, XML로 데이터를 줘서 확실함
@st.cache_data(ttl=300)
def get_stock_news(stock_name):
    try:
        # 구글 뉴스 RSS (한국어, 한국 설정)
        url = f"https://news.google.com/rss/search?q={stock_name}+주가&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(url, timeout=5)
        
        # XML 파싱
        root = ET.fromstring(res.content)
        
        news_items = []
        # 상위 7개 뉴스 추출
        for item in root.findall('./channel/item')[:7]:
            title = item.find('title').text
            pubDate = item.find('pubDate').text
            source = item.find('source').text if item.find('source') is not None else "Google News"
            
            # 날짜 포맷 정리 (지저분한 GMT 제거)
            try:
                dt = datetime.strptime(pubDate, "%a, %d %b %Y %H:%M:%S %Z")
                date_str = dt.strftime("%Y-%m-%d %H:%M") # 한국 시간 변환은 생략(복잡성 방지)
            except:
                date_str = pubDate[:16]

            news_items.append({
                '제목': title,
                '출처': source,
                '시간': date_str
            })
            
        if news_items:
            return pd.DataFrame(news_items)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        return pd.DataFrame()

# --- 3. 정밀 진단 로직 ---
def calculate_score(df):
    if len(df) < 60: return 0, [], 0, 0, 0
    
    c = df['Close']
    ma5 = c.rolling(5).mean()
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    std = c.rolling(20).std()
    bandwidth = ((ma20 + (std * 2)) - (ma20 - (std * 2))) / ma20
    
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    exp12 = c.ewm(span=12, adjust=False).mean()
    exp26 = c.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    
    curr_price = c.iloc[-1]
    curr_rsi = rsi.iloc[-1]
    curr_vol = df['Volume'].iloc[-1]
    avg_vol = df['Volume'].tail(5).mean()
    
    score = 0
    reasons = []
    
    if curr_price > ma20.iloc[-1]: score += 20; reasons.append("20일선 위")
    if curr_price > ma60.iloc[-1]: score += 10; reasons.append("60일선 위")
    if ma5.iloc[-1] > ma20.iloc[-1]: score += 10; reasons.append("골든크로스")
    if 40 <= curr_rsi <= 70: score += 10; reasons.append("RSI안정")
    elif curr_rsi < 30: score += 20; reasons.append("RSI과매도")
    if macd.iloc[-1] > signal.iloc[-1]: score += 10; reasons.append("MACD매수")
    if avg_vol > 0 and curr_vol > avg_vol * 1.5: score += 20; reasons.append("거래량폭발")
    if bandwidth.iloc[-1] < 0.15: score += 10; reasons.append("밴드수축")
        
    return score, reasons, curr_price, curr_rsi, curr_vol

# --- 4. 정밀 분석 (차트 포함) ---
def analyze_deep_pro(code, name):
    try:
        df = fdr.DataReader(code).tail(240)
        score, reasons, curr_price, curr_rsi, curr_vol = calculate_score(df)
        
        c = df['Close']
        ma20 = c.rolling(20).mean()
        ma60 = c.rolling(60).mean()
        upper = ma20 + (c.rolling(20).std() * 2)
        lower = ma20 - (c.rolling(20).std() * 2)
        
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df.index, c, label='Price', color='black')
        ax1.plot(df.index, ma20, label='MA20', color='green', alpha=0.7)
        ax1.plot(df.index, ma60, label='MA60', color='orange', alpha=0.7)
        ax1.fill_between(df.index, lower, upper, color='gray', alpha=0.1)
        ax1.set_title(f"Analysis: {name} ({code})")
        ax1.legend()
        ax1.grid(True, alpha=0.2)
        
        ax2 = fig.add_subplot(gs[1])
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs_series = gain / loss
        rsi_series = 100 - (100 / (1 + rs_series))
        
        ax2.plot(df.index, rsi_series, label='RSI', color='purple')
        ax2.axhline(30, color='blue', linestyle='--')
        ax2.axhline(70, color='red', linestyle='--')
        ax2.legend()
        ax2.grid(True, alpha=0.2)
        
        plt.tight_layout()
        return fig, score, reasons, curr_price
    except: return None, 0, [], 0

# --- 5. 전수 조사 ---
def scan_all_candidates(code_name_list):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(code_name_list)
    
    def process_one(item):
        code, name = item
        try:
            df = fdr.DataReader(code).tail(120)
            score, reasons, price, rsi, vol = calculate_score(df)
            if score >= 50:
                return {
                    '종목명': name,
                    '현재가': price,
                    '점수': score,
                    '등급': 'S급' if score >= 80 else 'A급' if score >= 60 else 'B급',
                    '사유': ", ".join(reasons)
                }
        except: pass
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_one, item): item for item in code_name_list}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            if res: results.append(res)
            if i % 5 == 0:
                prog = (i + 1) / total
                progress_bar.progress(prog)
                status_text.caption(f"🔍 {i+1}/{total} 종목 정밀 진단 중...")
                
    progress_bar.empty()
    status_text.empty()
    results.sort(key=lambda x: x['점수'], reverse=True)
    return results

# --- 메인 UI ---
st.title(f"⚔️ 단타 전투 머신 (Google News)")
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 스나이퍼", "📡 통합 스캐너", "🩺 정밀 분석+뉴스", "📝 매매 일지"
    ])

    def color_surplus(val):
        if isinstance(val, str): return 'color: black'
        color = 'red' if val > 0 else 'blue' if val < 0 else 'black'
        return f'color: {color}'

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

    with tab2:
        st.markdown("### 📡 AI 패턴 정밀 스캔")
        if st.button("🚀 스캔 시작"):
            st.info("정밀 분석 탭의 '전수 조사' 기능을 이용하시면 더 강력합니다!")

    with tab3:
        st.markdown("### 🩺 AI 주치의 + 📰 구글 뉴스")
        
        with st.expander("🚀 전체 스캔 & 유망주 발굴 (Click)", expanded=True):
            if st.button("🔥 Top 150 전수 조사 시작", type="primary"):
                target_list = list(zip(all_df.index, all_df['종목명']))
                with st.spinner("AI가 150개 차트를 모두 분석 중입니다..."):
                    scan_results = scan_all_candidates(target_list)
                    
                if scan_results:
                    st.success(f"✅ 분석 완료! 유망 종목 {len(scan_results)}개를 발견했습니다.")
                    res_df = pd.DataFrame(scan_results)
                    s_cnt = len(res_df[res_df['점수'] >= 80])
                    a_cnt = len(res_df[(res_df['점수'] >= 60) & (res_df['점수'] < 80)])
                    c1, c2 = st.columns(2)
                    c1.metric("👑 S급 (강력 매수)", f"{s_cnt}개")
                    c2.metric("🥇 A급 (매수 고려)", f"{a_cnt}개")
                    st.dataframe(
                        res_df[['등급', '점수', '종목명', '현재가', '사유']].style
                        .format({'현재가': '{:,}', '점수': '{:.0f}'})
                        .map(lambda x: 'color: red; font-weight: bold' if x == 'S급' else 'color: orange' if x == 'A급' else 'color: blue', subset=['등급']),
                        hide_index=True, use_container_width=True
                    )
                else: st.warning("조건에 맞는 종목을 찾지 못했습니다.")

        st.divider()

        st.markdown("#### 🔍 개별 종목 상세 진단 (차트 + 뉴스)")
        opts = ["선택"] + [f"{r['종목명']} ({r['종가']:,})" for i, r in all_df.head(150).iterrows()]
        sel = st.selectbox("진단할 종목 선택", opts)
        
        if sel != "선택":
            name = sel.split(' (')[0]
            code = all_df[all_df['종목명'] == name].index[0]
            
            if st.button(f"'{name}' 분석 및 뉴스 탐색"):
                # 1. 차트 분석
                with st.spinner("1단계: 차트 정밀 진단 중..."):
                    fig, score, reasons, curr_price = analyze_deep_pro(code, name)
                
                # 2. 뉴스 검색 (Google News)
                with st.spinner(f"2단계: 구글에서 '{name}' 뉴스 수집 중..."):
                    news_df = get_stock_news(name)
                
                if fig:
                    c1, c2 = st.columns([2, 3])
                    with c1:
                        st.markdown(f"### 점수: **{score}점**")
                        for r in reasons: st.write(f"- {r}")
                        st.success(f"목표가: {int(curr_price*1.05):,}원")
                        st.error(f"손절가: {int(curr_price*0.97):,}원")
                        
                        st.markdown("---")
                        st.markdown("#### 📰 최신 뉴스 (Google)")
                        if not news_df.empty:
                            st.dataframe(news_df, hide_index=True, use_container_width=True)
                            st.caption("※ 구글 뉴스 검색 결과입니다. 호재를 체크하세요.")
                        else:
                            st.info("뉴스가 없습니다. (조용한 종목일 수 있음)")
                            
                    with c2:
                        st.pyplot(fig)

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
