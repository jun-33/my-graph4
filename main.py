import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연평균 기온 데이터를 이용해 회귀 직선을 만들고 "
    "연도별 예상 기온을 확인해 봅니다."
)


# =========================================================
# 데이터 주소
# =========================================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# =========================================================
# 데이터 불러오기
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

    # 날짜를 날짜 형식으로 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온을 숫자로 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 연도 만들기
    df["연도"] = df["날짜"].dt.year

    # 필요한 데이터만 사용
    df = df.dropna(
        subset=["날짜", "연도", "평균기온"]
    )

    return df


try:
    df = load_data()

except Exception as e:
    st.error("기온 데이터를 불러오지 못했습니다.")
    st.stop()


# =========================================================
# 연도별 데이터 만들기
# =========================================================

# 수업 기준 기간인 2025년까지만 사용
df = df[df["연도"] <= 2025].copy()


# 각 연도별 관측일 수 계산
year_count = (
    df.groupby("연도")
    .size()
    .reset_index(name="관측일수")
)


# 각 연도의 평균기온 계산
year_temp = (
    df.groupby("연도")["평균기온"]
    .mean()
    .reset_index()
)


# 관측일수와 연평균기온 합치기
annual = pd.merge(
    year_temp,
    year_count,
    on="연도",
    how="inner"
)


# 관측일이 300일 이상인 해만 사용
annual = annual[
    annual["관측일수"] >= 300
].copy()


# 연도순 정렬
annual = annual.sort_values("연도").reset_index(drop=True)


# =========================================================
# 회귀 분석
# =========================================================

if len(annual) < 2:
    st.error("회귀 분석에 사용할 수 있는 연도 데이터가 부족합니다.")
    st.stop()


# 독립 변수:
# 1908년부터 몇 년이 지났는지
annual["지난연수"] = annual["연도"] - 1908


# 종속 변수:
# 연평균기온
x = annual["지난연수"].to_numpy()
y = annual["평균기온"].to_numpy()


# 1차 선형회귀
slope, intercept = np.polyfit(x, y, 1)


# 회귀선의 예상값
annual["회귀예상기온"] = (
    slope * annual["지난연수"] + intercept
)


# 상관계수
correlation = np.corrcoef(x, y)[0, 1]


# 결정계수
r_squared = correlation ** 2


# =========================================================
# 기본 정보 표시
# =========================================================

start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())
year_count_used = len(annual)


st.subheader("📊 회귀 분석 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "사용한 연도 수",
        f"{year_count_used}년"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{start_year}년"
    )

with col3:
    st.metric(
        "끝 연도",
        f"{end_year}년"
    )

with col4:
    st.metric(
        "상관계수",
        f"{correlation:.3f}"
    )


# =========================================================
# 회귀식 표시
# =========================================================

st.write("### 📐 회귀 직선")

st.latex(
    f"기온 = {slope:.4f} \\times (연도 - 1908) "
    f"+ {intercept:.4f}"
)

st.write(
    f"결정계수(R²): **{r_squared:.3f}**"
)


# =========================================================
# 산점도 + 회귀선
# =========================================================

st.subheader("📈 서울 연평균 기온과 회귀 직선")

fig = go.Figure()


# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=8
        ),
        customdata=np.column_stack(
            [annual["관측일수"]]
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)


# 회귀선
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["회귀예상기온"],
        mode="lines",
        name="회귀 직선",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "%{x}년 예상 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    hovermode="x unified",
    height=550,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 연도 선택 및 예상 기온
# =========================================================

st.subheader("🔮 연도별 예상 기온")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# 선택한 연도의 예상 기온 계산
selected_x = selected_year - 1908

predicted_temp = (
    slope * selected_x + intercept
)


# 예상 기온 크게 표시
st.markdown(
    f"""
    <div style="
        padding: 30px;
        border-radius: 15px;
        background-color: #f5f7fa;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
    ">
        <div style="
            font-size: 24px;
            color: #555;
            margin-bottom: 10px;
        ">
            {selected_year}년 예상 연평균 기온
        </div>

        <div style="
            font-size: 55px;
            font-weight: bold;
        ">
            {predicted_temp:.2f}℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 선택한 연도를 그래프에 표시
# =========================================================

prediction_fig = go.Figure()


# 실제 데이터
prediction_fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7
        )
    )
)


# 회귀선 전체 구간
line_years = np.arange(
    1900,
    2101
)

line_x = line_years - 1908

line_temps = (
    slope * line_x + intercept
)


prediction_fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="회귀 직선",
        line=dict(
            width=3
        )
    )
)


# 선택한 연도
prediction_fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(
            size=16,
            symbol="star"
        ),
        hovertemplate=(
            f"<b>{selected_year}년</b><br>"
            f"예상 기온: {predicted_temp:.2f}℃"
            "<extra></extra>"
        )
    )
)


prediction_fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    xaxis=dict(
        range=[1900, 2100],
        dtick=10
    ),
    height=500,
    hovermode="closest"
)


st.plotly_chart(
    prediction_fig,
    use_container_width=True
)


# =========================================================
# 데이터 표
# =========================================================

with st.expander("📋 회귀 분석에 사용된 연도별 데이터 보기"):
    display_df = annual[
        [
            "연도",
            "평균기온",
            "관측일수",
            "지난연수",
            "회귀예상기온"
        ]
    ].copy()

    display_df.columns = [
        "연도",
        "연평균기온(℃)",
        "관측일수",
        "1908년부터 지난 연수",
        "회귀 예상기온(℃)"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 데이터 처리 기준 안내
# =========================================================

st.info(
    "분석 기준: 2025년까지의 데이터만 사용하고, "
    "연간 관측일수가 300일 미만인 해는 제외했습니다. "
    "회귀의 독립 변수는 '연도 - 1908'입니다."
)
