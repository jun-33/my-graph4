```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 페이지 설정
# =========================================================

st.set_page_config(
    page_title="서울 기온 선형회귀 비교",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균기온 선형회귀 비교")

st.write(
    "서울의 연평균기온을 이용해 선형회귀 모델을 만들고, "
    "최근 50년과 최근 100년의 학습 결과를 비교합니다."
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

    # 평균기온 숫자로 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 2025년 이후 데이터 제외
    df = df[df["연도"] <= 2025].copy()

    return df


df = load_data()


# =========================================================
# 연평균기온 계산
# =========================================================

yearly = (
    df.dropna(subset=["평균기온"])
      .groupby("연도")
      .agg(
          연평균기온=("평균기온", "mean"),
          관측일수=("평균기온", "count")
      )
      .reset_index()
)

# 관측일수가 300일 이상인 해만 사용
yearly = yearly[
    yearly["관측일수"] >= 300
].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)


# =========================================================
# 독립 변수 만들기
# =========================================================
#
# 회귀에서는 연도를 그대로 사용하지 않고
# "1908년부터 몇 년이 지났는가"를 사용
#
# 예:
# 1908년 → 0
# 1956년 → 48
# 2005년 → 97
# 2025년 → 117
#
# =========================================================

yearly["경과연수"] = yearly["연도"] - 1908


# =========================================================
# 데이터 확인
# =========================================================

st.subheader("📊 사용한 연평균기온 데이터")

st.write(
    f"전체 사용 연도: **{yearly['연도'].min()}~{yearly['연도'].max()}년**"
)

st.write(
    f"사용한 연도 수: **{len(yearly)}년**"
)

st.write(
    "2025년까지의 데이터 중 연간 관측일수가 300일 이상인 연도만 사용했습니다."
)


# =========================================================
# 전체 데이터 회귀
# =========================================================

X_all = yearly[["경과연수"]]
y_all = yearly["연평균기온"]

model_all = LinearRegression()
model_all.fit(X_all, y_all)

yearly["전체회귀예측"] = model_all.predict(X_all)

all_slope = model_all.coef_[0]
all_intercept = model_all.intercept_

all_mae = mean_absolute_error(
    y_all,
    yearly["전체회귀예측"]
)

all_mse = mean_squared_error(
    y_all,
    yearly["전체회귀예측"]
)

all_r2 = r2_score(
    y_all,
    yearly["전체회귀예측"]
)


# =========================================================
# 학습 / 테스트 데이터
# =========================================================

train_50 = yearly[
    (yearly["연도"] >= 1956) &
    (yearly["연도"] <= 2005)
].copy()

train_100 = yearly[
    (yearly["연도"] >= 1906) &
    (yearly["연도"] <= 2005)
].copy()

test = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()


# =========================================================
# 50년 학습 모델
# =========================================================

X_train_50 = train_50[["경과연수"]]
y_train_50 = train_50["연평균기온"]

X_test = test[["경과연수"]]
y_test = test["연평균기온"]

model_50 = LinearRegression()
model_50.fit(
    X_train_50,
    y_train_50
)

pred_50 = model_50.predict(X_test)

slope_50 = model_50.coef_[0]
intercept_50 = model_50.intercept_

mae_50 = mean_absolute_error(
    y_test,
    pred_50
)

mse_50 = mean_squared_error(
    y_test,
    pred_50
)

r2_50 = r2_score(
    y_test,
    pred_50
)


# =========================================================
# 100년 학습 모델
# =========================================================

X_train_100 = train_100[["경과연수"]]
y_train_100 = train_100["연평균기온"]

model_100 = LinearRegression()
model_100.fit(
    X_train_100,
    y_train_100
)

pred_100 = model_100.predict(X_test)

slope_100 = model_100.coef_[0]
intercept_100 = model_100.intercept_

mae_100 = mean_absolute_error(
    y_test,
    pred_100
)

mse_100 = mean_squared_error(
    y_test,
    pred_100
)

r2_100 = r2_score(
    y_test,
    pred_100
)


# =========================================================
# 전체 데이터 회귀 결과
# =========================================================

st.subheader("1️⃣ 전체 데이터로 만든 회귀선")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "회귀에 사용한 연도",
        f"{len(yearly)}년"
    )

with col2:
    st.metric(
        "기울기",
        f"{all_slope:.4f} ℃/년"
    )

with col3:
    st.metric(
        "MAE",
        f"{all_mae:.3f} ℃"
    )

with col4:
    st.metric(
        "R²",
        f"{all_r2:.3f}"
    )

st.write(
    f"회귀식: **연평균기온 = "
    f"{all_slope:.4f} × (연도 - 1908) "
    f"+ {all_intercept:.2f}**"
)


# =========================================================
# 테스트 데이터에 대한 두 모델 비교
# =========================================================

st.subheader("2️⃣ 최근 20년 테스트 성능 비교")

comparison = pd.DataFrame({
    "모델": [
        "최근 50년 학습 (1956~2005)",
        "최근 100년 학습 (1906~2005)"
    ],
    "학습기간": [
        "1956~2005",
        "1906~2005"
    ],
    "학습연도수": [
        len(train_50),
        len(train_100)
    ],
    "기울기 (℃/년)": [
        slope_50,
        slope_100
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
    ]
})

st.dataframe(
    comparison.style.format({
        "기울기 (℃/년)": "{:.4f}",
        "MAE (℃)": "{:.3f}",
        "MSE (℃²)": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 두 회귀선 비교 그래프
# =========================================================

st.subheader("3️⃣ 학습기간에 따른 회귀선 비교")

fig = go.Figure()


# 실제 연평균기온
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 50년 회귀선
prediction_years = np.arange(
    yearly["연도"].min(),
    2026
)

prediction_x = (
    prediction_years - 1908
).reshape(-1, 1)

prediction_50 = model_50.predict(
    prediction_x
)

fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_50,
        mode="lines",
        name="1956~2005 학습 회귀선",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "50년 학습 예상: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 100년 회귀선
prediction_100 = model_100.predict(
    prediction_x
)

fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_100,
        mode="lines",
        name="1906~2005 학습 회귀선",
        line=dict(
            width=3,
            dash="dash"
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "100년 학습 예상: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 테스트 기간 배경 표시
fig.add_vrect(
    x0=2006,
    x1=2025,
    fillcolor="gray",
    opacity=0.15,
    line_width=0,
    annotation_text="테스트 기간",
    annotation_position="top left"
)


fig.update_layout(
    title="학습기간에 따른 서울 연평균기온 회귀선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=600,
    hovermode="x unified",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 테스트 실제값 vs 예측값
# =========================================================

st.subheader("4️⃣ 2006~2025년 실제값과 예측값")

test_result = test[
    ["연도", "연평균기온"]
].copy()

test_result["50년 학습 예측"] = pred_50
test_result["100년 학습 예측"] = pred_100

fig2 = go.Figure()


fig2.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["연평균기온"],
        mode="lines+markers",
        name="실제값"
    )
)

fig2.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["50년 학습 예측"],
        mode="lines",
        name="50년 학습 예측"
    )
)

fig2.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["100년 학습 예측"],
        mode="lines",
        name="100년 학습 예측"
    )
)

fig2.update_layout(
    title="최근 20년 테스트 데이터 예측 결과",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=550,
    xaxis=dict(
        tickmode="linear",
        dtick=2
    )
)

st.plotly_chart(
    fig2,
    use_container_width=True
)


# =========================================================
# 결과 해석
# =========================================================

st.subheader("5️⃣ 결과 해석")

if mae_50 < mae_100:
    mae_result = "최근 50년 학습 모델의 MAE가 더 작아 테스트 기간의 평균적인 오차가 더 작았습니다."
elif mae_50 > mae_100:
    mae_result = "최근 100년 학습 모델의 MAE가 더 작아 테스트 기간의 평균적인 오차가 더 작았습니다."
else:
    mae_result = "두 모델의 MAE가 거의 같습니다."

if mse_50 < mse_100:
    mse_result = "최근 50년 학습 모델의 MSE가 더 작아 큰 오차까지 고려했을 때도 더 좋았습니다."
elif mse_50 > mse_100:
    mse_result = "최근 100년 학습 모델의 MSE가 더 작아 큰 오차까지 고려했을 때 더 좋았습니다."
else:
    mse_result = "두 모델의 MSE가 거의 같습니다."

if r2_50 > r2_100:
    r2_result = "최근 50년 학습 모델의 R²가 더 높아 테스트 기간의 기온 변화를 더 잘 설명했습니다."
elif r2_50 < r2_100:
    r2_result = "최근 100년 학습 모델의 R²가 더 높아 테스트 기간의 기온 변화를 더 잘 설명했습니다."
else:
    r2_result = "두 모델의 R²가 거의 같습니다."

st.write("**MAE 비교:** " + mae_result)
st.write("**MSE 비교:** " + mse_result)
st.write("**R² 비교:** " + r2_result)


# =========================================================
# 기울기 비교
# =========================================================

slope_difference = slope_50 - slope_100

st.write(
    f"**기울기 비교:** "
    f"최근 50년 회귀선의 기울기는 **{slope_50:.4f} ℃/년**, "
    f"최근 100년 회귀선의 기울기는 **{slope_100:.4f} ℃/년**입니다."
)

st.write(
    f"두 기울기의 차이는 **{slope_difference:.4f} ℃/년**입니다."
)

st.caption(
    "기울기가 클수록 연도가 1년 증가할 때 회귀선이 예측하는 연평균기온의 증가 폭이 큽니다."
)
```
