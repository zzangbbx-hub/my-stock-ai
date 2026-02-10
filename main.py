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
st.set_page_config(page_title="단타 전투 머신 (Final Pro)", layout="wide")

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
            result['수급량'] = result['수급량'].apply(clean_int)
            result['현재가'] = result['현재가'].apply(clean_int)
            
            return result.head(20) # 상위 20개
            
        except Exception as e:
            return pd.DataFrame()

    # 병렬 호출
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f_f = executor.submit(parse_table, url_foreign)
        f_i = executor.submit(parse_table, url_inst)
        df_f = f_f.result()
        df_i = f_i.result()
        
    # 합치기 (쌍끌이 계산)
    merged = pd.DataFrame()
    if not df_f.empty and not df_i.empty:
        # 외국인, 기관 데이터 합치기
        merged = pd.merge(df_f, df_i[['종목명', '수급량']], on='종목명', suffixes=('_F', '_I'))
        merged.rename(columns={'수급량_F': '외국인', '수급량_I': '기관'}, inplace=True)
        
        # 합계 계산
        merged['합계'] = merged['외국인'] + merged['기관']
        merged = merged.sort_values(by='합계', ascending=False)
        
    return df_f, df_i, merged

# --- 3. 통합 스캐너 ---
def run_all_scanners(code_list):
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
            prev = df.iloc[-2]
            
            vol_avg = df['Volume'].rolling(5).mean().iloc[-1]
            vol_ratio = (curr['Volume'] / vol_avg) * 100 if vol_avg > 0 else 0
            
            rsi = 100 - (100 / (1 + (c.diff().clip(lower=0).rolling(14).mean() / (-c.diff().clip(upper=0).rolling(14).mean()).replace(0, 1e-9)))).iloc[-1]
            
            tags = []
            score = 0
            
            if curr['Close'] > ma60.iloc[-1] and abs(curr['Close'] - ma20.iloc[-1])/curr['Close'] < 0.03:
                tags.append("🛡️안전빵")
                score += 40
            
            if len(df) >= 3 and df.iloc[-3]['Close'] > df.iloc[-3]['Open'] and prev['Close'] < prev['Open'] and curr['Close'] > curr['Open']:
                tags.append("🕯️양음양")
                score += 30

            if vol_ratio >= 200:
                tags.append("💪거래폭발")
                score += 20
            
            if band_w.iloc[-1] < 0.15:
                tags.append("💥용수철")
                score += 10
            
            if (curr['Open'] - prev['Close']) / prev['Close'] >= 0.03:
                tags.append("🚀갭상승")
                score += 10

            if rsi <= 30:
                tags.append("📉과낙폭")
                score += 10

            if tags:
                return {'code': code, 'tags': ", ".join(tags), 'price': curr['Close'], 'score': score}
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
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

# --- 4. 정밀 분석 ---
def analyze_deep(code, name):
    try:
        df = fdr.DataReader(code).tail(120)
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        curr_rsi = rsi.iloc[-1]
        
        high_p = df['High'].tail(60).max()
        low_p = df['Low'].tail(60).min()
        fibo_618 = high_p - ((high_p - low_p) * 0.618)
        
        vol_ratio = (df['Volume'].iloc[-1] / df['Volume'].tail(5).mean()) * 100
        
        df['Weekday'] = df.index.day_name()
        weekday_stats = df.groupby('Weekday')['Close'].apply(lambda x: x.pct_change().mean() * 100)
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        weekday_stats = weekday_stats.reindex(days_order)
        
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
        
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df.index, df['Close'], label='Price', color='blue')
        ax1.plot(df.index, df['Close'].rolling(20).mean(), label='MA20', color='green', alpha=0.5)
        ax1.axhline(fibo_618, color='orange', linestyle='--', label='Fibo 0.618')
        ax1.set_title(f"Analysis: {code}")
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(df.index, rsi, color='purple', label='RSI')
        ax2.axhline(70, color='red', linestyle='--')
        ax2.axhline(30, color='blue', linestyle='--')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        ax3 = fig.add_subplot(gs[2])
        colors = ['red' if v > 0 else 'blue' for v in weekday_stats.fillna(0).values]
        ax3.bar(weekday_stats.index.str[:3], weekday_stats.fillna(0).values, color=colors)
        ax3.set_title("Weekday Return (%)")
        ax3.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig, curr_rsi, fibo_618, vol_ratio
    except: return None, 0, 0, 0

# --- 메인 UI ---
st.title(f"⚔️ 단타 전투 머신 (Real-Time Pro)")
st.caption(f"접속일: {display_date}")

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
    st.error("⚠️ 시세 데이터를 불러오지 못했습니다. 잠시 후 새로고침 해주세요.")
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 스나이퍼", "📡 통합 스캐너(등급)", "💰 수급 포착(네이버)", "🔮 정밀 분석", "📝 매매 일지"
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
            if "돌파" in best['신호']: st.success(f"🚀 **[{best['종목명']}]** 저항 돌파! 강력 매수")
            else: st.warning(f"👀 **[{best['종목명']}]** 돌파 임박! 관망")

        i1, i2, i3, i4 = st.columns(4)
        i1.metric("현재가", f"{best['종가']:,}")
        i2.metric("목표가", f"{best['2차저항']:,}")
        i3.metric("신호", best['신호'])
        i4.metric("대금", f"{best['거래대금(억)']}억")
        
        st.divider()
        c_sort, c_blank = st.columns([1, 4])
        with c_sort:
            sort_opt = st.radio("정렬 기준", ["거래대금순", "등락률순"], horizontal=True)
            
        view_df = all_df.copy()
        if sort_opt == "등락률순":
            view_df = view_df.sort_values(by='등락률', ascending=False)
        else:
            view_df = view_df.sort_values(by='거래대금(억)', ascending=False)
            
        st.dataframe(
            view_df[['종목명', '종가', '등락률', '신호', '거래대금(억)']].style
            .format({'종가': '{:,}', '거래대금(억)': '{:,}', '등락률': '{:.2f}%'})
            .map(color_surplus, subset=['등락률']), 
            hide_index=True, use_container_width=True
        )

    # [Tab 2] 통합 스캐너
    with tab2:
        st.markdown("### 📡 AI 패턴 정밀 스캔")
        st.caption("※ **S급(50점+) > A급(30점+) > B급** 순으로 보여줍니다.")
        
        if st.button("🚀 스캔 & 등급 판정"):
            scan_codes = all_df.index.tolist()
            results = run_all_scanners(scan_codes)
            
            if results:
                st.toast(f"🔔 {len(results)}개 포착! S급부터 보여줍니다.", icon="🥇")
                for res in results:
                    name = all_df.loc[res['code']]['종목명']
                    price = res['price']
                    tags = res['tags']
                    score = res['score']
                    
                    if score >= 50: st.markdown(f"### 🔴 S급 (강력 추천) - {name}")
                    elif score >= 30: st.markdown(f"### 🟠 A급 (매수 우수) - {name}")
                    else: st.markdown(f"### 🔵 B급 (관심 단계) - {name}")
                    
                    st.write(f"**가격:** {int(price):,}원 | **점수:** {score}점")
                    st.info(f"👉 **포착 사유:** {tags}")
                    st.divider()
            else: st.info("특이 패턴 종목이 없습니다.")

    # [Tab 3] 수급 포착 (네이버 금융 완벽 파싱)
    with tab3:
        st.markdown("### 🦁 네이버 금융 수급 랭킹")
        st.caption("※ 네이버 금융에서 **실시간 상위 종목**을 긁어옵니다.")
        
        if st.button("💰 수급 데이터 불러오기"):
            with st.spinner("네이버 금융 정밀 접속 중..."):
                df_f, df_i, merged = get_naver_realtime_supply()
                
                if not merged.empty:
                    st.success(f"🚀 **쌍끌이(외인+기관) 포착: {len(merged)}종목**")
                    st.dataframe(
                        merged[['종목명', '현재가', '등락률', '외국인', '기관']].style
                        .format({'현재가': '{:,}', '외국인': '{:,}', '기관': '{:,}', '등락률': '{:.2f}%'})
                        .map(color_surplus, subset=['등락률']),
                        hide_index=True, use_container_width=True
                    )
                else:
                    st.info("쌍끌이(동시 매수) 종목이 없거나 데이터를 가져오지 못했습니다.")
                
                st.divider()
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**🦁 외국인 순매수 Top 10**")
                    if not df_f.empty: 
                        st.dataframe(
                            df_f[['종목명', '등락률', '수급량']].head(10).style
                            .format({'수급량': '{:,}', '등락률': '{:.2f}%'})
                            .map(color_surplus, subset=['등락률']), 
                            hide_index=True
                        )
                    else: st.error("데이터 없음")
                with c2:
                    st.markdown("**🐯 기관 순매수 Top 10**")
                    if not df_i.empty: 
                        st.dataframe(
                            df_i[['종목명', '등락률', '수급량']].head(10).style
                            .format({'수급량': '{:,}', '등락률': '{:.2f}%'})
                            .map(color_surplus, subset=['등락률']), 
                            hide_index=True
                        )
                    else: st.error("데이터 없음")

    # [Tab 4] 정밀 분석
    with tab4:
        opts = ["선택"] + [f"{r['종목명']} ({r['종가']:,})" for i, r in all_df.head(100).iterrows()]
        sel = st.selectbox("종목 선택", opts)
        
        if sel != "선택":
            name = sel.split(' (')[0]
            code = all_df[all_df['종목명'] == name].index[0]
            curr = all_df.loc[code]['종가']
            st.info(f"💰 현재가: **{curr:,}원**")
            
            mode = st.radio("기준", ["주수", "금액"], horizontal=True)
            qty = 0
            if mode == "주수":
                q = st.number_input("주수", 1, 10000, 10)
                st.caption(f"필요 금액: {q*curr:,}원")
                qty = q
            else:
                m = st.number_input("금액", 10000, 100000000, 1000000)
                qty = int(m // curr)
                st.caption(f"매수 가능: {qty:,}주")
                
            if st.button("⚖️ AI 최종 판결 보기"):
                fig, rsi, fibo, vol = analyze_deep(code, name)
                if fig:
                    score = 0
                    reasons = []
                    if 40 <= rsi <= 60: score += 20; reasons.append("안정적 흐름")
                    elif rsi < 30: score += 30; reasons.append("과매도(반등기회)")
                    elif rsi > 70: score -= 20; reasons.append("과매수(고점위험)")
                    if vol > 150: score += 30; reasons.append("거래량 폭발")
                    if all_df.loc[code]['등락률'] > 0: score += 20
                    
                    st.divider()
                    st.subheader("🧑‍⚖️ AI 최종 판결")
                    if score >= 70: st.success(f"✅ **[진입 승인]** 강력 매수 신호! ({score}점)")
                    elif score >= 50: st.warning(f"⚠️ **[보류]** 확실하지 않습니다. ({score}점)")
                    else: st.error(f"❌ **[진입 금지]** 위험합니다. ({score}점)")
                    st.caption(f"이유: {', '.join(reasons)}")
                    
                    st.pyplot(fig)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.info(f"매수: {qty:,}주")
                    c2.success(f"익절: {int(curr*1.03):,}")
                    c3.error(f"손절: {int(curr*0.98):,}")

    # [Tab 5] 매매 일지
    with tab5:
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
