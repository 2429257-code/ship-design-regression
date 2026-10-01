import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 페이지 설정
# ============================================================

st.set_page_config(
    page_title="선박 데이터 회귀분석",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 선박 데이터 회귀분석 프로그램")
st.write(
    "선박 데이터를 불러와 최소제곱법과 경사하강법을 이용하여 "
    "선형 및 비선형 회귀분석을 수행하는 프로그램입니다."
)


# ============================================================
# 함수 1. 최소제곱법
# ============================================================

def least_squares(x, y):
    """
    최소제곱법을 이용한 단순 선형회귀

    y = ax + b

    a = Σ(x-x평균)(y-y평균) / Σ(x-x평균)^2
    b = y평균 - a*x평균
    """

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum(
        (x - x_mean) * (y - y_mean)
    )

    denominator = np.sum(
        (x - x_mean) ** 2
    )

    if denominator == 0:
        return None, None

    a = numerator / denominator
    b = y_mean - a * x_mean

    return a, b


# ============================================================
# 함수 2. 경사하강법
# ============================================================

def gradient_descent_polynomial(
    x,
    y,
    degree=2,
    learning_rate=0.01,
    epochs=5000
):
    """
    다항회귀를 경사하강법으로 계산

    degree = 2이면

    y = w0 + w1*x + w2*x^2

    형태의 비선형 회귀가 된다.
    """

    # 데이터 정규화
    x_mean = np.mean(x)
    x_std = np.std(x)

    if x_std == 0:
        x_std = 1

    x_scaled = (x - x_mean) / x_std

    # 다항식 행렬 생성
    X = np.column_stack(
        [x_scaled ** i for i in range(degree + 1)]
    )

    # 가중치 초기값
    weights = np.zeros(degree + 1)

    n = len(y)

    loss_history = []

    for epoch in range(epochs):

        # 예측값
        y_pred = X @ weights

        # 오차
        error = y_pred - y

        # 평균제곱오차
        loss = np.mean(error ** 2)

        loss_history.append(loss)

        # Gradient 계산
        gradient = (2 / n) * (X.T @ error)

        # 가중치 업데이트
        weights = weights - learning_rate * gradient

    return weights, x_mean, x_std, loss_history


# ============================================================
# 함수 3. R² 계산
# ============================================================

def calculate_r2(y, y_pred):

    ss_total = np.sum(
        (y - np.mean(y)) ** 2
    )

    ss_residual = np.sum(
        (y - y_pred) ** 2
    )

    if ss_total == 0:
        return 0

    return 1 - (
        ss_residual / ss_total
    )


# ============================================================
# 함수 4. MSE 계산
# ============================================================

def calculate_mse(y, y_pred):

    return np.mean(
        (y - y_pred) ** 2
    )


# ============================================================
# 사이드바
# ============================================================

st.sidebar.header("📂 데이터 설정")

uploaded_file = st.sidebar.file_uploader(
    "CSV 또는 Excel 파일을 업로드하세요.",
    type=["csv", "xlsx", "xls"]
)


# ============================================================
# 파일 불러오기
# ============================================================

df = None

if uploaded_file is not None:

    try:

        if uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

        st.sidebar.success(
            "데이터를 성공적으로 불러왔습니다."
        )

    except Exception as e:

        st.error(
            f"파일을 불러오는 중 오류가 발생했습니다: {e}"
        )


# ============================================================
# 기본 샘플 데이터
# ============================================================

if df is None:

    st.info(
        "왼쪽에서 CSV 또는 Excel 파일을 업로드하세요."
    )

    st.write("### 📌 프로그램 사용 순서")

    st.write(
        """
        1. 선박 데이터를 CSV 또는 Excel 파일로 준비합니다.
       
        2. 왼쪽의 파일 업로드 버튼을 이용하여 데이터를 불러옵니다.
       
        3. X축과 Y축으로 사용할 변수를 선택합니다.
       
        4. 최소제곱법을 이용한 선형회귀를 수행합니다.
       
        5. 경사하강법을 이용한 비선형회귀를 수행합니다.
       
        6. 회귀식, R², MSE 및 그래프를 확인합니다.
        """
    )

    st.stop()


# ============================================================
# 데이터 확인
# ============================================================

st.header("📊 데이터 확인")

st.write(
    f"데이터 크기: **{df.shape[0]}행 × {df.shape[1]}열**"
)

st.dataframe(
    df,
    use_container_width=True
)


# ============================================================
# 숫자형 데이터 선택
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


if len(numeric_columns) < 2:

    st.error(
        "분석을 위해 숫자로 이루어진 열이 최소 2개 필요합니다."
    )

    st.stop()


# ============================================================
# X / Y 선택
# ============================================================

st.header("⚙️ 분석 변수 선택")

col1, col2 = st.columns(2)

with col1:

    x_column = st.selectbox(
        "X축 변수",
        numeric_columns,
        index=0
    )

with col2:

    y_column = st.selectbox(
        "Y축 변수",
        numeric_columns,
        index=1 if len(numeric_columns) > 1 else 0
    )


# ============================================================
# 데이터 전처리
# ============================================================

data = df[
    [x_column, y_column]
].copy()

data[x_column] = pd.to_numeric(
    data[x_column],
    errors="coerce"
)

data[y_column] = pd.to_numeric(
    data[y_column],
    errors="coerce"
)

data = data.dropna()

x = data[x_column].values.astype(float)
y = data[y_column].values.astype(float)


if len(x) < 3:

    st.error(
        "분석을 위해 최소 3개의 유효한 데이터가 필요합니다."
    )

    st.stop()


# ============================================================
# 기본 산점도
# ============================================================

st.header("📈 데이터 산점도")

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.scatter(
    x,
    y
)

ax.set_xlabel(
    x_column
)

ax.set_ylabel(
    y_column
)

ax.set_title(
    f"{y_column} vs {x_column}"
)

ax.grid(
    alpha=0.3
)

st.pyplot(fig)


# ============================================================
# 최소제곱법 선형회귀
# ============================================================

st.header("1️⃣ 최소제곱법을 이용한 선형회귀")

a, b = least_squares(
    x,
    y
)

if a is None:

    st.error(
        "X 데이터의 값이 모두 같아서 회귀분석을 수행할 수 없습니다."
    )

else:

    y_linear = a * x + b

    r2_linear = calculate_r2(
        y,
        y_linear
    )

    mse_linear = calculate_mse(
        y,
        y_linear
    )

    st.write("### 회귀식")

    st.latex(
        f"y = {a:.6f}x + {b:.6f}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "기울기",
            f"{a:.6f}"
        )

    with col2:

        st.metric(
            "절편",
            f"{b:.6f}"
        )

    with col3:

        st.metric(
            "R²",
            f"{r2_linear:.6f}"
        )

    st.write(
        f"평균제곱오차(MSE): **{mse_linear:.6f}**"
    )

    # 그래프용 정렬
    order = np.argsort(x)

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.scatter(
        x,
        y,
        label="Observed Data"
    )

    ax.plot(
        x[order],
        y_linear[order],
        linewidth=2,
        label="Least Squares Regression"
    )

    ax.set_xlabel(
        x_column
    )

    ax.set_ylabel(
        y_column
    )

    ax.set_title(
        "Linear Regression using Least Squares"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    st.pyplot(fig)

    with st.expander("📐 최소제곱법 계산 원리"):

        st.write(
            "선형회귀식은 다음과 같이 설정합니다."
        )

        st.latex(
            r"y = ax + b"
        )

        st.write(
            "기울기 a는 다음 식을 이용하여 계산합니다."
        )

        st.latex(
            r"a = \frac{\sum (x-\bar{x})(y-\bar{y})}"
            r"{\sum (x-\bar{x})^2}"
        )

        st.write(
            "절편 b는 다음 식으로 계산합니다."
        )

        st.latex(
            r"b = \bar{y} - a\bar{x}"
        )


# ============================================================
# 경사하강법 비선형회귀
# ============================================================

st.header("2️⃣ 경사하강법을 이용한 비선형회귀")

degree = st.slider(
    "다항식 차수",
    min_value=2,
    max_value=5,
    value=2
)

learning_rate = st.number_input(
    "학습률(Learning Rate)",
    min_value=0.00001,
    max_value=1.0,
    value=0.01,
    step=0.001,
    format="%.5f"
)

epochs = st.number_input(
    "반복 횟수(Epochs)",
    min_value=100,
    max_value=20000,
    value=5000,
    step=100
)


weights, x_mean, x_std, loss_history = (
    gradient_descent_polynomial(
        x,
        y,
        degree=int(degree),
        learning_rate=learning_rate,
        epochs=int(epochs)
    )
)


# 예측값
x_scaled = (
    x - x_mean
) / x_std

X = np.column_stack(
    [
        x_scaled ** i
        for i in range(int(degree) + 1)
    ]
)

y_nonlinear = X @ weights

r2_nonlinear = calculate_r2(
    y,
    y_nonlinear
)

mse_nonlinear = calculate_mse(
    y,
    y_nonlinear
)


# ============================================================
# 비선형회귀 결과
# ============================================================

st.write("### 분석 결과")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "R²",
        f"{r2_nonlinear:.6f}"
    )

with col2:

    st.metric(
        "MSE",
        f"{mse_nonlinear:.6f}"
    )

with col3:

    st.metric(
        "최종 Loss",
        f"{loss_history[-1]:.6f}"
    )


# ============================================================
# 비선형 회귀 그래프
# ============================================================

x_plot = np.linspace(
    np.min(x),
    np.max(x),
    300
)

x_plot_scaled = (
    x_plot - x_mean
) / x_std

X_plot = np.column_stack(
    [
        x_plot_scaled ** i
        for i in range(int(degree) + 1)
    ]
)

y_plot = X_plot @ weights


fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.scatter(
    x,
    y,
    label="Observed Data"
)

ax.plot(
    x_plot,
    y_plot,
    linewidth=2,
    label=f"Polynomial Regression (Degree {degree})"
)

ax.set_xlabel(
    x_column
)

ax.set_ylabel(
    y_column
)

ax.set_title(
    "Nonlinear Regression using Gradient Descent"
)

ax.legend()

ax.grid(
    alpha=0.3
)

st.pyplot(fig)


# ============================================================
# Loss 변화 그래프
# ============================================================

st.write("### 📉 경사하강법 Loss 변화")

fig, ax = plt.subplots(
    figsize=(10, 4)
)

ax.plot(
    loss_history
)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "MSE Loss"
)

ax.set_title(
    "Gradient Descent Learning Process"
)

ax.grid(
    alpha=0.3
)

st.pyplot(fig)


# ============================================================
# 계산 원리
# ============================================================

with st.expander("📚 경사하강법 계산 원리 보기"):

    st.write(
        "비선형회귀에서는 다항식을 이용하여 다음과 같은 모델을 구성합니다."
    )

    st.latex(
        r"y = w_0 + w_1x + w_2x^2 + \cdots + w_nx^n"
    )

    st.write(
        "예측값과 실제값의 차이를 이용하여 손실함수(MSE)를 계산합니다."
    )

    st.latex(
        r"MSE = \frac{1}{n}\sum(y_{pred}-y)^2"
    )

    st.write(
        "경사하강법에서는 손실함수의 기울기를 계산한 후 "
        "가중치를 반복적으로 업데이트합니다."
    )

    st.latex(
        r"w_{new} = w_{old} - \alpha \frac{\partial L}{\partial w}"
    )

    st.write(
        "여기서 α는 학습률(Learning Rate)입니다."
    )


# ============================================================
# 최종 비교
# ============================================================

st.header("📋 회귀분석 결과 비교")

comparison = pd.DataFrame(
    {
        "분석 방법": [
            "최소제곱법 선형회귀",
            "경사하강법 비선형회귀"
        ],
        "R²": [
            r2_linear,
            r2_nonlinear
        ],
        "MSE": [
            mse_linear,
            mse_nonlinear
        ]
    }
)

st.dataframe(
    comparison,
    use_container_width=True
)


st.success(
    "분석이 완료되었습니다. "
    "X축과 Y축을 변경하여 다른 선박 변수의 관계도 분석할 수 있습니다."
)
