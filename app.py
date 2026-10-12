import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(
    page_title="선박 인공신경망 예측",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 선박 데이터 인공신경망 예측 프로그램")
st.write("공개된 선박 데이터를 인공신경망으로 학습하고 예측합니다.")

st.sidebar.header("데이터 불러오기")
uploaded_file = st.sidebar.file_uploader(
    "선박 데이터 파일 업로드",
    type=["csv", "xlsx", "xls", "data", "txt"]
)

if uploaded_file is None:
    st.info("왼쪽 메뉴에서 선박 데이터 파일을 업로드하세요.")
    st.markdown("""
    ### 프로그램 사용 방법
    1. 데이터 파일을 업로드합니다.
    2. 입력 변수와 예측 변수를 선택합니다.
    3. 인공신경망을 학습합니다.
    4. 예측 성능과 그래프를 확인합니다.
    5. 새로운 데이터를 입력해 예측합니다.

    추천 데이터: UCI Yacht Hydrodynamics
    """)
    st.stop()

try:
    filename = uploaded_file.name.lower()

    if filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    elif filename.endswith((".data", ".txt")):
        df = pd.read_csv(
            uploaded_file,
            sep=r"\s+",
            header=None
        )
    else:
        df = pd.read_csv(uploaded_file)

    if filename.endswith((".data", ".txt")) and df.shape[1] == 7:
        df.columns = [
            "중심부력 위치",
            "프리즘 계수",
            "길이-배수량 비",
            "폭-흘수 비",
            "길이-폭 비",
            "프루드 수",
            "잉여저항"
        ]

except Exception as e:
    st.error(f"파일을 읽지 못했습니다: {e}")
    st.stop()

numeric_df = df.select_dtypes(include=np.number).copy()
numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

if numeric_df.shape[1] < 2:
    st.error("분석 가능한 숫자형 열이 2개 이상 필요합니다.")
    st.stop()

st.header("1. 데이터 확인")
st.write(f"데이터 크기: {df.shape[0]}행 × {df.shape[1]}열")
st.dataframe(df, use_container_width=True)

st.header("2. 예측 변수 선택")
columns = numeric_df.columns.tolist()

target = st.selectbox(
    "예측할 변수 (Y)",
    columns,
    index=len(columns) - 1
)

features = st.multiselect(
    "입력 변수 (X)",
    [c for c in columns if c != target],
    default=[c for c in columns if c != target]
)

if not features:
    st.warning("입력 변수를 하나 이상 선택하세요.")
    st.stop()

data = numeric_df[features + [target]].dropna()

if len(data) < 10:
    st.error("학습을 위해 유효한 데이터가 최소 10개 필요합니다.")
    st.stop()

st.write(f"분석에 사용할 데이터: {len(data)}개")

test_size = st.slider(
    "평가용 데이터 비율 (%)",
    min_value=10,
    max_value=40,
    value=20,
    step=5
)

hidden_units = st.slider(
    "은닉층 노드 수",
    min_value=4,
    max_value=64,
    value=16,
    step=4
)

max_iter = st.slider(
    "학습 반복 횟수",
    min_value=500,
    max_value=3000,
    value=1500,
    step=250
)

if st.button("🧠 인공신경망 학습 및 예측", type="primary"):
    try:
        X = data[features].to_numpy(dtype=float)
        y = data[target].to_numpy(dtype=float)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size / 100,
            random_state=42
        )

        x_scaler = StandardScaler()
        X_train_scaled = x_scaler.fit_transform(X_train)
        X_test_scaled = x_scaler.transform(X_test)

        model = MLPRegressor(
            hidden_layer_sizes=(hidden_units, hidden_units // 2),
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=42
        )

        with st.spinner("인공신경망을 학습하고 있습니다..."):
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)

        result = pd.DataFrame({
            "실제값": y_test,
            "예측값": predictions,
            "오차": y_test - predictions
        })

        st.session_state["ann_result"] = {
            "result": result,
            "model": model,
            "scaler": x_scaler,
            "features": list(features),
            "target": target,
            "r2": r2_score(y_test, predictions),
            "mse": mean_squared_error(y_test, predictions),
            "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
            "mae": mean_absolute_error(y_test, predictions)
        }

    except Exception as e:
        st.error(f"학습 중 오류가 발생했습니다: {e}")

if "ann_result" in st.session_state:
    saved = st.session_state["ann_result"]

    if saved["features"] != list(features) or saved["target"] != target:
        st.warning("변수 선택이 변경되었습니다. 인공신경망을 다시 학습하세요.")
        st.stop()

    st.header("3. 예측 성능 결과")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{saved['r2']:.4f}")
    c2.metric("MSE", f"{saved['mse']:.4f}")
    c3.metric("RMSE", f"{saved['rmse']:.4f}")
    c4.metric("MAE", f"{saved['mae']:.4f}")

    st.caption(
        "R²는 예측 성능을 나타내며, MSE·RMSE·MAE는 "
        "실제값과 예측값 사이의 오차를 나타냅니다."
    )

    st.header("4. 실제값과 예측값 비교")
    st.dataframe(saved["result"], use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        saved["result"]["실제값"],
        saved["result"]["예측값"]
    )

    low = min(
        saved["result"]["실제값"].min(),
        saved["result"]["예측값"].min()
    )
    high = max(
        saved["result"]["실제값"].max(),
        saved["result"]["예측값"].max()
    )

    ax.plot([low, high], [low, high], "--")
    ax.set_xlabel("실제값")
    ax.set_ylabel("예측값")
    ax.set_title("ANN: Actual vs Predicted")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.header("5. 새로운 데이터 예측")
    st.write("입력값을 변경한 뒤 예측 버튼을 눌러 보세요.")

    with st.form("new_prediction_form"):
        input_values = []
        input_columns = st.columns(2)

        for i, feature in enumerate(saved["features"]):
            with input_columns[i % 2]:
                value = st.number_input(
                    feature,
                    value=float(data[feature].median()),
                    key=f"new_input_{i}"
                )
                input_values.append(value)

        submitted = st.form_submit_button("새 데이터 예측")

    if submitted:
        new_X = np.array(input_values).reshape(1, -1)
        new_X_scaled = saved["scaler"].transform(new_X)
        prediction = saved["model"].predict(new_X_scaled)[0]

        st.success(
            f"예측된 {saved['target']}: {prediction:.6f}"
        )

    st.download_button(
        "예측 결과 CSV 다운로드",
        data=saved["result"].to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="ann_prediction_results.csv",
        mime="text/csv"
    )

    with st.expander("인공신경망 알고리즘 설명"):
        st.markdown("""
        1. **데이터 전처리:** 입력 변수의 크기를 표준화합니다.
        2. **데이터 분리:** 데이터를 학습용과 평가용으로 나눕니다.
        3. **순전파:** 입력값을 신경망에 전달하여 예측값을 계산합니다.
        4. **오차 계산:** 예측값과 실제값의 차이를 계산합니다.
        5. **역전파 및 가중치 갱신:** 오차를 줄이도록 가중치를 조정합니다.
        6. **성능 평가:** R², MSE, RMSE, MAE를 계산합니다.
        """)
