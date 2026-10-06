import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="서울 기온 선형회귀 비교",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 선형회귀 비교")
st.write(
    "과거의 연평균 기온으로 선형회귀 모델을 만들고 "
    "최근 20년의 기온을 얼마나 잘 예측하는지 비교합니다."
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
        encoding="utf-8"
    )

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 필요한 데이터만 사용
    df = df.dropna(
        subset=["연도", "평균기온"]
    )

    # 2025년까지만 사용
    df = df[df["연도"] <= 2025]

    return df


try:
    daily = load_data()

except Exception:
    st.error("서울 기온 데이터를 불러오지 못했습니다.")
    st.stop()


# =========================================================
# 연도별 데이터 만들기
# =========================================================

# 연도별 관측일 수
count_by_year = (
    daily
    .groupby("연도")
    .size()
    .reset_index(name="관측일수")
)

# 연도별 평균기온
temp_by_year = (
    daily
    .groupby("연도")["평균기온"]
    .mean()
    .reset_index(name="연평균기온")
)

# 두 데이터 합치기
annual = pd.merge(
    temp_by_year,
    count_by_year,
    on="연도"
)

# 관측일이 300일 이상인 연도만 사용
annual = annual[
    annual["관측일수"] >= 300
].copy()

annual = annual.sort_values("연도").reset_index(drop=True)


# =========================================================
# 독립 변수 만들기
# =========================================================

# 1908년부터 몇 년이 지났는지
annual["지난연수"] = annual["연도"] - 1908


# =========================================================
# 회귀 모델 함수
# =========================================================

def make_regression(data):

    x = data["지난연수"].to_numpy()
    y = data["연평균기온"].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)

    return slope, intercept


def predict(years, slope, intercept):

    years = np.asarray(years)

    x = years - 1908

    return slope * x + intercept


# =========================================================
# 평가 지표 함수
# =========================================================

def evaluate(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    # MAE
    mae = np.mean(
        np.abs(actual - predicted)
    )

    # MSE
    mse = np.mean(
        (actual - predicted) ** 2
    )

    # R²
    ss_res = np.sum(
        (actual - predicted) ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    r2 = 1 - (ss_res / ss_tot)

    return mae, mse, r2


# =========================================================
# 데이터 구분
# =========================================================

# 훈련 데이터
train_50 = annual[
    (annual["연도"] >= 1956) &
    (annual["연도"] <= 2005)
].copy()

train_100 = annual[
    (annual["연도"] >= 1906) &
    (annual["연도"] <= 2005)
].copy()


# 공통 테스트 데이터
test = annual[
    (annual["연도"] >= 2006) &
    (annual["연도"] <= 2025)
].copy()


if len(train_50) < 2 or len(train_100) < 2 or len(test) < 2:
    st.error("회귀 분석에 필요한 데이터가 부족합니다.")
    st.stop()


# =========================================================
# 회귀 모델 만들기
# =========================================================

slope_50, intercept_50 = make_regression(
    train_50
)

slope_100, intercept_100 = make_regression(
    train_100
)


# =========================================================
# 테스트 데이터 예측
# =========================================================

test["50년 모델 예측"] = predict(
    test["연도"],
    slope_50,
    intercept_50
)

test["100년 모델 예측"] = predict(
    test["연도"],
    slope_100,
    intercept_100
)


# =========================================================
# 테스트 데이터 평가
# =========================================================

mae_50, mse_50, r2_50 = evaluate(
    test["연평균기온"],
    test["50년 모델 예측"]
)

mae_100, mse_100, r2_100 = evaluate(
    test["연평균기온"],
    test["100년 모델 예측"]
)


# =========================================================
# 제목
# =========================================================

st.header("1. 훈련 데이터와 테스트 데이터")

st.write(
    "두 모델 모두 2006~2025년을 공통 테스트 데이터로 사용합니다."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "최근 50년 훈련",
        f"{len(train_50)}년"
    )
    st.caption("1956~2005년")

with col2:
    st.metric(
        "최근 100년 훈련",
        f"{len(train_100)}년"
    )
    st.caption("1906~2005년")

with col3:
    st.metric(
        "공통 테스트",
        f"{len(test)}년"
    )
    st.caption("2006~2025년")


# =========================================================
# 훈련 데이터 시각화
# =========================================================

st.header("2. 훈련 데이터와 회귀선")

fig_train = go.Figure()


# 실제 전체 연평균 기온
fig_train.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="연평균 기온",
        marker=dict(size=7),
        hovertemplate=(
            "%{x}년<br>"
            "연평균 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 50년 회귀선
years_50 = np.arange(
    1956,
    2006
)

fig_train.add_trace(
    go.Scatter(
        x=years_50,
        y=predict(
            years_50,
            slope_50,
            intercept_50
        ),
        mode="lines",
        name="1956~2005년 회귀선",
        line=dict(width=3),
        hovertemplate=(
            "%{x}년<br>"
            "예상 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 100년 회귀선
years_100 = np.arange(
    1906,
    2006
)

fig_train.add_trace(
    go.Scatter(
        x=years_100,
        y=predict(
            years_100,
            slope_100,
            intercept_100
        ),
        mode="lines",
        name="1906~2005년 회귀선",
        line=dict(width=3),
        hovertemplate=(
            "%{x}년<br>"
            "예상 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


fig_train.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    height=550,
    hovermode="x unified"
)

st.plotly_chart(
    fig_train,
    use_container_width=True
)


# =========================================================
# 회귀선 기울기 비교
# =========================================================

st.header("3. 회귀선의 기울기 비교")

col1, col2 = st.columns(2)

with col1:

    st.subheader("최근 50년 모델")

    st.metric(
        "기울기",
        f"{slope_50:.4f} ℃/년"
    )

    st.write(
        f"1956~2005년의 데이터를 사용했습니다."
    )

with col2:

    st.subheader("최근 100년 모델")

    st.metric(
        "기울기",
        f"{slope_100:.4f} ℃/년"
    )

    st.write(
        f"1906~2005년의 데이터를 사용했습니다."
    )


# =========================================================
# 테스트 데이터 평가
# =========================================================

st.header("4. 최근 20년 테스트 데이터 평가")

st.write(
    "두 회귀 모델은 학습에 사용하지 않은 "
    "2006~2025년 데이터를 이용해 평가했습니다."
)


# 평가표
result = pd.DataFrame({
    "모델": [
        "최근 50년 모델",
        "최근 100년 모델"
    ],
    "훈련 기간": [
        "1956~2005",
        "1906~2005"
    ],
    "MAE (℃)": [
        mae_50,
        mae_100
    ],
    "MSE (℃²)": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ],
    "기울기 (℃/년)": [
        slope_50,
        slope_100
    ]
})

st.dataframe(
    result.style.format({
        "MAE (℃)": "{:.3f}",
        "MSE (℃²)": "{:.3f}",
        "R²": "{:.3f}",
        "기울기 (℃/년)": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 어떤 모델이 더 좋은가
# =========================================================

st.subheader("🏆 테스트 성능 비교")

col1, col2, col3 = st.columns(3)

with col1:

    if mae_50 < mae_100:
        better_mae = "최근 50년"
    else:
        better_mae = "최근 100년"

    st.metric(
        "MAE가 더 작은 모델",
        better_mae
    )

with col2:

    if mse_50 < mse_100:
        better_mse = "최근 50년"
    else:
        better_mse = "최근 100년"

    st.metric(
        "MSE가 더 작은 모델",
        better_mse
    )

with col3:

    if r2_50 > r2_100:
        better_r2 = "최근 50년"
    else:
        better_r2 = "최근 100년"

    st.metric(
        "R²가 더 큰 모델",
        better_r2
    )


# =========================================================
# 테스트 데이터 실제값 vs 예측값
# =========================================================

st.header("5. 최근 20년 실제 기온과 예측 기온 비교")

fig_test = go.Figure()


# 실제값
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="lines+markers",
        name="실제 연평균 기온",
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate=(
            "%{x}년<br>"
            "실제 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 50년 모델
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["50년 모델 예측"],
        mode="lines+markers",
        name="50년 모델 예측",
        line=dict(dash="dash"),
        hovertemplate=(
            "%{x}년<br>"
            "50년 모델: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 100년 모델
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["100년 모델 예측"],
        mode="lines+markers",
        name="100년 모델 예측",
        line=dict(dash="dot"),
        hovertemplate=(
            "%{x}년<br>"
            "100년 모델: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


fig_test.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=1
    ),
    height=550,
    hovermode="x unified"
)

st.plotly_chart(
    fig_test,
    use_container_width=True
)


# =========================================================
# 테스트 오차
# =========================================================

st.header("6. 연도별 예측 오차")

error_df = test[
    [
        "연도",
        "연평균기온",
        "50년 모델 예측",
        "100년 모델 예측"
    ]
].copy()

error_df["50년 모델 오차"] = (
    error_df["50년 모델 예측"]
    - error_df["연평균기온"]
)

error_df["100년 모델 오차"] = (
    error_df["100년 모델 예측"]
    - error_df["연평균기온"]
)


fig_error = go.Figure()

fig_error.add_trace(
    go.Bar(
        x=error_df["연도"],
        y=error_df["50년 모델 오차"],
        name="50년 모델 오차"
    )
)

fig_error.add_trace(
    go.Bar(
        x=error_df["연도"],
        y=error_df["100년 모델 오차"],
        name="100년 모델 오차"
    )
)

fig_error.add_hline(
    y=0,
    line_width=2
)

fig_error.update_layout(
    xaxis_title="연도",
    yaxis_title="예측 기온 - 실제 기온 (℃)",
    barmode="group",
    height=500
)

st.plotly_chart(
    fig_error,
    use_container_width=True
)


# =========================================================
# 해석
# =========================================================

st.header("7. 결과 해석")

st.write(
    f"""
    **최근 50년 모델**의 테스트 MAE는 **{mae_50:.3f}℃**이고,
    **최근 100년 모델**의 테스트 MAE는 **{mae_100:.3f}℃**입니다.

    MAE는 실제 기온과 예측 기온이 평균적으로 얼마나 차이 나는지를
    나타내므로 **작을수록 좋은 모델**입니다.

    또한 최근 50년 모델의 회귀선 기울기는
    **{slope_50:.4f}℃/년**,
    최근 100년 모델의 기울기는
    **{slope_100:.4f}℃/년**입니다.

    두 기울기를 비교하면 어떤 기간의 데이터를 학습했을 때
    서울의 장기적인 기온 상승 추세를 더 크게 또는 작게 판단하는지
    확인할 수 있습니다.
    """
)


# =========================================================
# 데이터 기준
# =========================================================

with st.expander("📌 데이터 처리 기준"):

    st.write(
        """
        - 2025년까지의 데이터만 사용
        - 연간 관측일수가 300일 미만인 연도는 제외
        - 연도별 일평균기온을 평균하여 연평균기온 계산
        - 독립 변수는 `연도 - 1908`
        - 1956~2005년 데이터를 최근 50년 훈련 데이터로 사용
        - 1906~2005년 데이터를 최근 100년 훈련 데이터로 사용
        - 2006~2025년은 두 모델의 공통 테스트 데이터
        - MAE, MSE, R²를 이용하여 테스트 성능 평가
        """
    )
