import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_risco_defasagem_pipeline.joblib"
THRESHOLD_PATH = BASE_DIR / "threshold_risco_defasagem.json"
FILE_2022 = BASE_DIR / "pede_2022_tratado.csv"
FILE_2023 = BASE_DIR / "pede_2023_tratado.csv"
FILE_2024 = BASE_DIR / "pede_2024_tratado.csv"


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de modelo não encontrado em {MODEL_PATH.name}. "
            f"Certifique-se de que o arquivo esteja no mesmo diretório do app."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metadata() -> Dict[str, Any]:
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado em {THRESHOLD_PATH.name}. "
            f"Certifique-se de que o arquivo esteja no mesmo diretório do app."
        )
    with THRESHOLD_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_feature_columns_and_types(
    metadata: Dict[str, Any],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Retorna lista de colunas de features e uma separação simples
    entre numéricas e categóricas, baseada em heurísticas de nome.

    Essa função NÃO altera o modelo; apenas organiza a interface.
    """
    feature_columns: List[str] = metadata.get("feature_columns", [])

    numeric_keywords = [
        "INDE",
        "Idade",
        "Ano_ingresso",
        "N_Av",
        "IAA",
        "IEG",
        "IPS",
        "IPP",
        "IDA",
        "Mat",
        "Por",
        "Ing",
        "IPV",
        "IAN",
        "Cg",
        "Cf",
        "Ct",
        "Defasagem",
    ]

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in feature_columns:
        if any(kw in col for kw in numeric_keywords):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return feature_columns, numeric_cols, categorical_cols


def build_input_form(
    feature_columns: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Dict[str, Any]:
    st.subheader("Informações do aluno")
    st.markdown(
        "Preencha apenas os campos que você conhece. "
        "Quanto mais completo o preenchimento, mais confiável tende a ser a estimativa."
    )

    inputs: Dict[str, Any] = {}

    with st.form("form_risco_defasagem"):
        # Campos principais de 2023 (ano mais recente, foco do modelo)
        st.markdown("### 1. Indicadores essenciais (2023)")
        main_2023_cols = [
            "Idade_2023",
            "Gênero_2023",
            "Fase_Ideal_2023",
            "INDE_2023",
            "IAA_2023",
            "IEG_2023",
            "IPS_2023",
            "IPP_2023",
        ]
        main_2023_cols = [c for c in main_2023_cols if c in feature_columns]

        # Demais colunas de 2023 ficam como avançadas (opcionais)
        all_2023_cols = [
            c for c in feature_columns if c.endswith("_2023") and c not in main_2023_cols
        ]

        genero_options = ["", "Feminino", "Masculino"]
        fase_ideal_options = ["", "Quartzo", "Ágata", "Ametista", "Topázio"]

        for col in main_2023_cols:
            label = col.replace("_2023", "").replace("_", " ")
            if "Gênero" in col:
                inputs[col] = st.selectbox(
                    label=label,
                    options=genero_options,
                    index=0,
                    key=f"sel_{col}",
                )
            elif "Fase_Ideal" in col:
                inputs[col] = st.selectbox(
                    label="Fase ideal",
                    options=fase_ideal_options,
                    index=0,
                    key=f"sel_{col}",
                )
            elif col in numeric_cols:
                inputs[col] = st.number_input(
                    label=label,
                    value=None,
                    step=0.01,
                    format="%.4f",
                    key=f"num_{col}",
                )
            elif col in categorical_cols:
                inputs[col] = st.text_input(label=label, key=f"cat_{col}")

        # Histórico 2022 e demais campos ficam em seções opcionais,
        # para não poluir a interface, mas ainda respeitar feature_columns.
        with st.expander("2. Histórico 2022 (opcional)"):
            year_2022_cols = [c for c in feature_columns if c.endswith("_2022")]
            for col in year_2022_cols:
                label = col.replace("_2022", "").replace("_", " ")
                if "Gênero" in col:
                    inputs[col] = st.selectbox(
                        label=label,
                        options=genero_options,
                        index=0,
                        key=f"sel_{col}",
                    )
                elif "Fase_Ideal" in col:
                    inputs[col] = st.selectbox(
                        label="Fase ideal (2022)",
                        options=fase_ideal_options,
                        index=0,
                        key=f"sel_{col}",
                    )
                elif col in numeric_cols:
                    inputs[col] = st.number_input(
                        label=label,
                        value=None,
                        step=0.01,
                        format="%.4f",
                        key=f"num_{col}",
                    )
                elif col in categorical_cols:
                    inputs[col] = st.text_input(label=label, key=f"cat_{col}")

        with st.expander("3. Campos avançados (opcional)"):
            advanced_cols = [
                c
                for c in feature_columns
                if (c in all_2023_cols) or (c not in main_2023_cols and not c.endswith("_2022"))
            ]
            for col in advanced_cols:
                label = col.replace("_", " ")
                if col in numeric_cols:
                    inputs[col] = st.number_input(
                        label=label,
                        value=None,
                        step=0.01,
                        format="%.4f",
                        key=f"num_{col}",
                    )
                elif col in categorical_cols:
                    inputs[col] = st.text_input(label=label, key=f"cat_{col}")

        submitted = st.form_submit_button("Calcular risco")

    return inputs if submitted else {}


def build_input_dataframe(
    feature_columns: List[str],
    raw_inputs: Dict[str, Any],
) -> pd.DataFrame:
    """
    Constrói um DataFrame com uma linha e colunas exatamente em feature_columns.
    Campos não preenchidos são enviados como NaN para o pipeline tratar.
    """
    data_row: Dict[str, Any] = {}
    for col in feature_columns:
        val = raw_inputs.get(col, None)
        if val is None or (isinstance(val, str) and val.strip() == ""):
            data_row[col] = np.nan
        else:
            data_row[col] = val

    # Cálculo automático de IDA quando aplicável (se notas existirem e IDA estiver vazio)
    mat_2023, por_2023, ing_2023 = data_row.get("Mat_2023"), data_row.get("Por_2023"), data_row.get("Ing_2023")
    if "IDA_2023" in data_row and pd.isna(data_row["IDA_2023"]):
        notas_2023 = [n for n in [mat_2023, por_2023, ing_2023] if not pd.isna(n)]
        if notas_2023:
            data_row["IDA_2023"] = float(np.mean(notas_2023))

    mat_2022, por_2022, ing_2022 = data_row.get("Mat_2022"), data_row.get("Por_2022"), data_row.get("Ing_2022")
    if "IDA_2022" in data_row and pd.isna(data_row["IDA_2022"]):
        notas_2022 = [n for n in [mat_2022, por_2022, ing_2022] if not pd.isna(n)]
        if notas_2022:
            data_row["IDA_2022"] = float(np.mean(notas_2022))

    return pd.DataFrame([data_row], columns=feature_columns)


def build_recommendations(input_df: pd.DataFrame, proba_risco: float, threshold: float) -> List[str]:
    recs: List[str] = []

    ieg = input_df.at[0, "IEG_2023"] if "IEG_2023" in input_df.columns else np.nan
    ida = input_df.at[0, "IDA_2023"] if "IDA_2023" in input_df.columns else np.nan
    ian = input_df.at[0, "IAN_2023"] if "IAN_2023" in input_df.columns else np.nan
    ips = input_df.at[0, "IPS_2023"] if "IPS_2023" in input_df.columns else np.nan

    if proba_risco >= threshold:
        recs.append("Priorizar acompanhamento individual nas próximas semanas com plano de intervenção curto.")
        recs.append("Realizar reunião de alinhamento entre coordenação pedagógica, psicopedagogia e responsáveis.")
    else:
        recs.append("Manter monitoramento mensal dos indicadores para detectar mudanças precoces.")

    if not pd.isna(ieg) and ieg < 6:
        recs.append("Reforçar estratégias de engajamento (metas curtas, tutoria e participação em atividades orientadas).")
    if not pd.isna(ida) and ida < 6:
        recs.append("Criar trilha de reforço em Língua Portuguesa e Matemática com revisões semanais.")
    if not pd.isna(ian) and ian <= 5:
        recs.append("Intensificar ações de recomposição de aprendizagem para reduzir defasagem no ciclo atual.")
    if not pd.isna(ips) and ips < 6:
        recs.append("Encaminhar para suporte psicossocial e acompanhar evolução socioemocional no próximo bimestre.")

    if not recs:
        recs.append("Coletar mais indicadores para aumentar a precisão da análise e personalizar intervenções.")
    return recs


def get_shap_explanation(model, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna as top contribuições SHAP quando aplicável ao modelo.
    Em caso de indisponibilidade/incompatibilidade, retorna DataFrame vazio.
    """
    try:
        import shap  # type: ignore
    except Exception:
        return pd.DataFrame()

    try:
        prep = model.named_steps.get("prep")
        estimator = model.named_steps.get("model")
        if prep is None or estimator is None:
            return pd.DataFrame()

        x_trans = prep.transform(input_df)
        feature_names = prep.get_feature_names_out()

        # SHAP para modelos baseados em árvore (ex.: RandomForest)
        if hasattr(estimator, "estimators_"):
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(x_trans)

            # Em classificação binária: pode vir lista [classe0, classe1]
            if isinstance(shap_values, list) and len(shap_values) > 1:
                values = shap_values[1][0]
            elif isinstance(shap_values, list):
                values = shap_values[0][0]
            else:
                values = shap_values[0]

            contrib = pd.DataFrame(
                {
                    "feature": feature_names,
                    "shap_value": values,
                    "impacto_abs": np.abs(values),
                }
            ).sort_values("impacto_abs", ascending=False)
            return contrib.head(8).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_base_dados() -> Dict[int, pd.DataFrame]:
    if not FILE_2022.exists() or not FILE_2023.exists() or not FILE_2024.exists():
        raise FileNotFoundError("Arquivos de base (2022/2023/2024) não encontrados na raiz do projeto.")
    return {
        2022: pd.read_csv(FILE_2022),
        2023: pd.read_csv(FILE_2023),
        2024: pd.read_csv(FILE_2024),
    }


def to_numeric_safe(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return out


def prepare_analytics_data(raw: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
    numeric_cols = ["INDE", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV", "IAN", "Defasagem", "Mat", "Por", "Ing"]
    prepared: Dict[int, pd.DataFrame] = {}
    for ano, df in raw.items():
        dfx = to_numeric_safe(df, numeric_cols)
        dfx["Ano"] = ano
        prepared[ano] = dfx
    return prepared


def render_visao_geral(dfs: Dict[int, pd.DataFrame]):
    st.header("Análise de Dados de Performance Estudantil")
    st.subheader("Bem-vindo(a) à Análise de Dados da Passos Mágicos")
    st.write(
        "Este dashboard explora padrões de desempenho estudantil, engajamento, autoavaliação e "
        "aspectos psicossociais/psicopedagógicos para apoiar decisões de intervenção."
    )

    df24 = dfs[2024]
    inde24 = df24["INDE"].dropna()
    col1, col2, col3 = st.columns(3)
    col1.metric("Média INDE 2024", f"{inde24.mean():.2f}" if not inde24.empty else "N/D")
    col2.metric("Desvio padrão INDE 2024", f"{inde24.std():.2f}" if not inde24.empty else "N/D")
    col3.metric("Alunos 2024", f"{len(df24)}")

    st.markdown("### Prévia dos dados (2024)")
    cols_preview = [c for c in ["RA", "INDE", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV", "IAN", "Fase_Ideal"] if c in df24.columns]
    st.dataframe(df24[cols_preview].head(5), use_container_width=True)


def render_q1_ian(dfs: Dict[int, pd.DataFrame]):
    st.header("Pergunta 1: Adequação do nível (IAN)")
    st.caption("Qual é o perfil geral de defasagem dos alunos (IAN) e como ele evolui?")

    medias = []
    for ano, df in dfs.items():
        if "IAN" in df.columns:
            medias.append({"Ano": ano, "IAN_medio": df["IAN"].mean()})
    if medias:
        st.line_chart(pd.DataFrame(medias).set_index("Ano"))

    df24 = dfs[2024]
    if "IAN" in df24.columns:
        bins = [-np.inf, 5, 7, np.inf]
        labels = ["Severa (<=5)", "Moderada (5-7]", "Adequada (>7)"]
        faixa = pd.cut(df24["IAN"], bins=bins, labels=labels)
        dist = faixa.value_counts().reindex(labels)
        st.bar_chart(dist)
        st.write("Distribuição de defasagem (IAN) em 2024.")


def render_q2_ida(dfs: Dict[int, pd.DataFrame]):
    st.header("Pergunta 2: Desempenho acadêmico (IDA)")
    st.caption("O desempenho acadêmico médio (IDA) está melhorando, estagnado ou caindo?")

    data = []
    for ano, df in dfs.items():
        data.append({"Ano": ano, "IDA_medio": df["IDA"].mean(), "INDE_medio": df["INDE"].mean()})
    evol = pd.DataFrame(data).set_index("Ano")
    st.line_chart(evol)
    st.dataframe(pd.DataFrame(data), use_container_width=True)


def render_q3_ieg(dfs: Dict[int, pd.DataFrame]):
    st.header("Pergunta 3: Engajamento nas atividades (IEG)")
    st.caption("O IEG tem relação com IDA e IPV?")

    df = dfs[2023].copy()
    cols = [c for c in ["IEG", "IDA", "IPV"] if c in df.columns]
    if len(cols) >= 2:
        corr = df[cols].corr(numeric_only=True)
        st.write("Correlação entre indicadores (2023):")
        st.dataframe(corr, use_container_width=True)
    if all(c in df.columns for c in ["IEG", "IDA"]):
        st.scatter_chart(df[["IEG", "IDA"]].dropna(), x="IEG", y="IDA")
    if all(c in df.columns for c in ["IEG", "IPV"]):
        st.scatter_chart(df[["IEG", "IPV"]].dropna(), x="IEG", y="IPV")


def render_q4_iaa(dfs: Dict[int, pd.DataFrame]):
    st.header("Pergunta 4: Autoavaliação (IAA)")
    st.caption("As percepções dos alunos sobre si mesmos são coerentes com desempenho e engajamento?")
    df = dfs[2023].copy()
    if all(c in df.columns for c in ["IAA", "IDA", "IEG"]):
        st.scatter_chart(df[["IAA", "IDA"]].dropna(), x="IAA", y="IDA")
        st.scatter_chart(df[["IAA", "IEG"]].dropna(), x="IAA", y="IEG")
        st.write("Regra simples: distância entre IAA e IDA indica potencial super/subestimação.")


def render_q5_ips(dfs: Dict[int, pd.DataFrame]):
    st.header("Pergunta 5: Aspectos Psicossociais (IPS)")
    st.caption("Há padrões de IPS que antecedem quedas de desempenho?")

    d23 = dfs[2023][["RA", "INDE", "IPS"]].rename(columns={"INDE": "INDE_2023", "IPS": "IPS_2023"})
    d24 = dfs[2024][["RA", "INDE"]].rename(columns={"INDE": "INDE_2024"})
    m = d23.merge(d24, on="RA", how="inner")
    m["delta_INDE_24_23"] = m["INDE_2024"] - m["INDE_2023"]
    st.scatter_chart(m[["IPS_2023", "delta_INDE_24_23"]].dropna(), x="IPS_2023", y="delta_INDE_24_23")
    st.write(f"Correlação IPS_2023 vs ΔINDE(24-23): {m[['IPS_2023','delta_INDE_24_23']].corr().iloc[0,1]:.3f}")


def render_q6_ipp(dfs: Dict[int, pd.DataFrame]):
    st.header("Pergunta 6: Aspectos Psicopedagógicos (IPP)")
    st.caption("IPP confirma ou contradiz a defasagem identificada por IAN?")
    df = dfs[2023]
    if all(c in df.columns for c in ["IPP", "IAN"]):
        st.scatter_chart(df[["IPP", "IAN"]].dropna(), x="IPP", y="IAN")
        st.write(f"Correlação IPP vs IAN (2023): {df[['IPP','IAN']].corr().iloc[0,1]:.3f}")


def render_q7_ipv(dfs: Dict[int, pd.DataFrame]):
    st.header("Pergunta 7: Ponto de Virada (IPV)")
    st.caption("Quais indicadores mais se relacionam ao IPV?")
    df = dfs[2023]
    cols = [c for c in ["IDA", "IEG", "IPS", "IAA", "IPP", "IPV"] if c in df.columns]
    if len(cols) > 1:
        corr = df[cols].corr(numeric_only=True)[["IPV"]].drop(index="IPV", errors="ignore").sort_values("IPV", ascending=False)
        st.dataframe(corr, use_container_width=True)
    st.write("Simulação rápida de tendência de IPV:")
    ida = st.slider("IDA (simulação)", 0.0, 10.0, 6.0, 0.1)
    ieg = st.slider("IEG (simulação)", 0.0, 10.0, 6.0, 0.1)
    ips = st.slider("IPS (simulação)", 0.0, 10.0, 6.0, 0.1)
    sim_ipv = 0.4 * ida + 0.35 * ieg + 0.25 * ips
    st.metric("IPV simulado (proxy explicativa)", f"{sim_ipv:.2f}")


def render_q8_multidim(dfs: Dict[int, pd.DataFrame]):
    st.header("Pergunta 8: Multidimensionalidade dos Indicadores")
    st.caption("Quais combinações de IDA + IEG + IPS + IPP elevam o INDE?")
    df = dfs[2023]
    cols = [c for c in ["INDE", "IDA", "IEG", "IPS", "IPP"] if c in df.columns]
    corr = df[cols].corr(numeric_only=True)
    st.dataframe(corr, use_container_width=True)

    if set(["INDE", "IDA", "IEG", "IPS", "IPP"]).issubset(df.columns):
        d = df[["INDE", "IDA", "IEG", "IPS", "IPP"]].dropna()
        d["grupo"] = np.where(d["INDE"] >= d["INDE"].median(), "Alto INDE", "Baixo INDE")
        st.dataframe(d.groupby("grupo")[["IDA", "IEG", "IPS", "IPP"]].mean().round(2), use_container_width=True)


def render_q9_ml(model, metadata, feature_columns, numeric_cols, categorical_cols):
    st.header("Pergunta 9: Previsão de Risco com Machine Learning")
    st.caption("Probabilidade de risco de defasagem, classificação por threshold e explicabilidade quando aplicável.")

    raw_inputs = build_input_form(feature_columns, numeric_cols, categorical_cols)
    if not raw_inputs:
        st.info("Preencha o formulário e clique em **Calcular risco** para ver o resultado.")
        return

    input_df = build_input_dataframe(feature_columns, raw_inputs)
    proba_risco = float(model.predict_proba(input_df)[0, 1])
    threshold = float(metadata.get("threshold", 0.5))

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Probabilidade de risco", f"{proba_risco * 100:.1f}%")
        st.write("Classificação:", "Em risco de defasagem" if proba_risco >= threshold else "Sem risco identificado")
    with col2:
        if proba_risco >= threshold:
            st.error("Aluno em risco de defasagem. Recomenda-se intervenção pedagógica prioritária.")
        else:
            st.success("Sem risco elevado identificado no momento.")

    st.markdown("### Recomendações pedagógicas automáticas")
    for rec in build_recommendations(input_df, proba_risco, threshold):
        st.write(f"- {rec}")

    st.markdown("### Explicabilidade (SHAP)")
    shap_df = get_shap_explanation(model, input_df)
    if shap_df.empty:
        st.info("SHAP não disponível para este ambiente/modelo no momento.")
    else:
        st.dataframe(shap_df[["feature", "shap_value"]], use_container_width=True, hide_index=True)


def render_q10_efetividade(dfs: Dict[int, pd.DataFrame]):
    st.header("Pergunta 10: Efetividade do Programa")
    st.caption("Os indicadores mostram melhora consistente ao longo do ciclo e das fases?")

    frames = []
    for ano, df in dfs.items():
        if "Fase_Ideal" in df.columns and "INDE" in df.columns:
            tmp = df.groupby("Fase_Ideal", dropna=False)["INDE"].mean().reset_index()
            tmp["Ano"] = ano
            frames.append(tmp)
    if frames:
        allf = pd.concat(frames, ignore_index=True)
        st.dataframe(allf.pivot_table(index="Fase_Ideal", columns="Ano", values="INDE"), use_container_width=True)

    inde_medias = pd.DataFrame([{"Ano": ano, "INDE_medio": df["INDE"].mean()} for ano, df in dfs.items()])
    st.line_chart(inde_medias.set_index("Ano"))


def render_q11_insights(dfs: Dict[int, pd.DataFrame]):
    st.header("Pergunta 11: Insights e Criatividade")
    st.caption("Cruzamentos adicionais para apoiar decisões estratégicas.")

    df24 = dfs[2024]
    if "Instituição_de_ensino" in df24.columns and "INDE" in df24.columns:
        st.markdown("### INDE médio por tipo de instituição (2024)")
        inst = df24.groupby("Instituição_de_ensino")["INDE"].mean().sort_values(ascending=False)
        st.bar_chart(inst)

    text_cols = [c for c in ["Destaque_IEG", "Destaque_IDA", "Destaque_IPV"] if c in df24.columns]
    if text_cols:
        st.markdown("### Frequência de preenchimento dos destaques (2024)")
        freq = {c: int(df24[c].notna().sum()) for c in text_cols}
        st.bar_chart(pd.Series(freq))


def main():
    st.set_page_config(
        page_title="Risco de defasagem - Passos Mágicos",
        layout="wide",
    )

    # Tema e ajustes visuais básicos
    st.markdown(
        """
        <style>
        .main {
            background-color: #020617;
            color: #e5e7eb;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 900px;
        }
        div.stButton > button {
            background: linear-gradient(90deg, #f97316, #ef4444);
            color: white;
            border-radius: 999px;
            height: 3rem;
            border: none;
            font-weight: 600;
        }
        div.stButton > button:hover {
            filter: brightness(1.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        metadata = load_metadata()
    except FileNotFoundError as e:
        st.error(str(e))
        metadata = {"feature_columns": []}
    except Exception as e:
        st.error(f"Erro ao carregar metadados: {e}")
        metadata = {"feature_columns": []}

    st.markdown("### 🎓 Previsão de Defasagem Educacional")
    st.caption(
        "Analise a probabilidade de um aluno apresentar **risco de defasagem educacional** "
        "com base em seus indicadores de desempenho, engajamento e contexto."
    )

    # Badge com modelo e threshold
    st.markdown(
        f"""
        <div style="padding: 0.5rem 0.75rem; border-radius: 999px;
                    background: rgba(15,23,42,0.9); display: inline-flex;
                    gap: 0.75rem; align-items: center; font-size: 0.8rem;
                    margin-top: 0.5rem; margin-bottom: 1.5rem;">
            <span>Modelo: <strong>{metadata.get("model", "N/D")}</strong></span>
            <span>|</span>
            <span>Threshold: <strong>{metadata.get("threshold", 0.5):.2f}</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Como interpretar")
        st.markdown(
            "- Preencha apenas os campos que você souber.\n"
            "- Probabilidade alta **não é diagnóstico**, é um alerta.\n"
            "- Use em conjunto com a avaliação da equipe pedagógica."
        )
        st.markdown("---")
        nav = st.radio(
            "Escolha uma Análise",
            [
                "Visão Geral",
                "Predição (Analisar aluno)",
                "Pergunta 1: Adequação do nível (IAN)",
                "Pergunta 2: Desempenho acadêmico (IDA)",
                "Pergunta 3: Engajamento nas atividades (IEG)",
                "Pergunta 4: Autoavaliação (IAA)",
                "Pergunta 5: Aspectos Psicossociais (IPS)",
                "Pergunta 6: Aspectos Psicopedagógicos (IPP)",
                "Pergunta 7: Ponto de Virada (IPV)",
                "Pergunta 8: Multidimensionalidade dos Indicadores",
                "Pergunta 9: Previsão de Risco com Machine Learning",
                "Pergunta 10: Efetividade do Programa",
                "Pergunta 11: Insights e Criatividade",
            ],
        )
        st.markdown("---")
        st.caption(
            "Ferramenta desenvolvida no Datathon Passos Mágicos (2022–2024)."
        )

    try:
        model = load_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        st.stop()

    feature_columns, numeric_cols, categorical_cols = get_feature_columns_and_types(
        metadata
    )

    if not feature_columns:
        st.error(
            "A lista de features não foi encontrada nos metadados. "
            "Verifique o arquivo threshold_risco_defasagem.json."
        )
        st.stop()

    try:
        base_raw = load_base_dados()
        dfs = prepare_analytics_data(base_raw)
    except Exception as e:
        st.error(f"Erro ao carregar bases analíticas: {e}")
        st.stop()

    if nav == "Visão Geral":
        render_visao_geral(dfs)
    elif nav == "Predição (Analisar aluno)":
        render_q9_ml(model, metadata, feature_columns, numeric_cols, categorical_cols)
    elif nav == "Pergunta 1: Adequação do nível (IAN)":
        render_q1_ian(dfs)
    elif nav == "Pergunta 2: Desempenho acadêmico (IDA)":
        render_q2_ida(dfs)
    elif nav == "Pergunta 3: Engajamento nas atividades (IEG)":
        render_q3_ieg(dfs)
    elif nav == "Pergunta 4: Autoavaliação (IAA)":
        render_q4_iaa(dfs)
    elif nav == "Pergunta 5: Aspectos Psicossociais (IPS)":
        render_q5_ips(dfs)
    elif nav == "Pergunta 6: Aspectos Psicopedagógicos (IPP)":
        render_q6_ipp(dfs)
    elif nav == "Pergunta 7: Ponto de Virada (IPV)":
        render_q7_ipv(dfs)
    elif nav == "Pergunta 8: Multidimensionalidade dos Indicadores":
        render_q8_multidim(dfs)
    elif nav == "Pergunta 9: Previsão de Risco com Machine Learning":
        render_q9_ml(model, metadata, feature_columns, numeric_cols, categorical_cols)
    elif nav == "Pergunta 10: Efetividade do Programa":
        render_q10_efetividade(dfs)
    elif nav == "Pergunta 11: Insights e Criatividade":
        render_q11_insights(dfs)


if __name__ == "__main__":
    main()

