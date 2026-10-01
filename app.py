import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Ship Design Regression",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Ship Design Regression")
st.caption("선박 설계 데이터를 불러와 선형·비선형 회귀분석을 수행하는 데이터 분석 프로그램")

st.sidebar.header("1. 데이터 불러오기")
uploaded_file = st.sidebar.file_uploader(
    "CSV 또는 Excel 파일을 선택하세요.",
    type=["csv", "xlsx", "xls"]
)

@st.cache_data
def load_uploaded_file(file_bytes, file_name):
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes))

if uploaded_file is not None:
    try:
        df = load_uploaded_file(uploaded_file.getvalue(), uploaded_file.name)
        st.sidebar.success("파일을 불러왔습니다.")
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()
else:
    try:
        df = pd.read_csv("ship_design_sample.csv")
        st.sidebar.info("기본 예제 데이터가 사용되고 있습니다.")
    except FileNotFoundError:
        st.error("ship_design_sample.csv 파일을 찾을 수 없습니다.")
        st.stop()

st.subheader("2. 데이터 확인")
st.write(f"데이터 크기: **{df.shape[0]}행 × {df.shape[1]}열**")
st.dataframe(df, use_container_width=True, height=260)

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_columns) < 2:
    st.error("회귀분석을 위해 숫자로 된 열이 2개 이상 필요합니다.")
    st.stop()

st.subheader("3. 분석 변수 선택")
col1, col2, col3 = st.columns(3)

with col1:
    x_col = st.selectbox("X축 변수", numeric_columns, index=0)

with col2:
    y_default = 1 if len(numeric_columns) > 1 else 0
    y_col = st.selectbox("Y축 변수", numeric_columns, index=y_default)

with col3:
    analysis_type = st.selectbox(
        "회귀분석 종류",
        ["선형회귀", "비선형회귀(다항식)"]
    )

if x_col == y_col:
    st.warning("X축과 Y축은 서로 다른 변수를 선택해주세요.")
    st.stop()

analysis_df = df[[x_col, y_col]].dropna().copy()
X = analysis_df[[x_col]].values
y = analysis_df[y_col].values

st.subheader("4. 데이터 시각화")
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(X, y, alpha=0.7, label="Data")

if analysis_type == "선형회귀":
    model = LinearRegression()
    model.fit(X, y)
    x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_line = model.predict(x_line)

    ax.plot(x_line, y_line, linewidth=2, label="Linear Regression")
    predictions = model.predict(X)

else:
    degree = st.slider("다항식 차수", min_value=2, max_value=5, value=2)
    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )
    model.fit(X, y)
    x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_line = model.predict(x_line)

    ax.plot(x_line, y_line, linewidth=2, label=f"Polynomial Regression (degree={degree})")
    predictions = model.predict(X)

ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
ax.set_title(f"{x_col} vs {y_col}")
ax.grid(True, alpha=0.25)
ax.legend()
st.pyplot(fig)

st.subheader("5. 회귀분석 결과")

r2 = r2_score(y, predictions)
mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))

m1, m2, m3 = st.columns(3)
m1.metric("R²", f"{r2:.4f}")
m2.metric("MAE", f"{mae:.4f}")
m3.metric("RMSE", f"{rmse:.4f}")

if analysis_type == "선형회귀":
    slope = model.coef_[0]
    intercept = model.intercept_
    st.info(
        f"회귀식: **{y_col} = {slope:.6f} × {x_col} + {intercept:.6f}**"
    )
else:
    st.info("다항식 회귀 모델을 사용하여 곡선 형태의 관계를 분석했습니다.")

st.subheader("6. 해석")
st.write(
    f"선택한 변수는 **{x_col} → {y_col}** 입니다. "
    f"현재 데이터에서 결정계수 R²는 **{r2:.4f}**이며, "
    "R²는 모델이 관측값의 변동을 얼마나 설명하는지 나타내는 지표입니다."
)

st.download_button(
    "분석에 사용한 데이터 CSV 다운로드",
    data=analysis_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="analysis_data.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption("※ 기본 예제 데이터는 프로그램 기능 테스트를 위한 합성 데이터입니다.")
