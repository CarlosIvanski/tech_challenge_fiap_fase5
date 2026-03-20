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

    # Card do formulário principal
    st.markdown(
        """
        <div style="padding: 1.25rem 1.5rem; border-radius: 0.75rem;
                    border: 1px solid rgba(148,163,184,0.25);
                    background: rgba(15,23,42,0.85); margin-bottom: 1rem;">
        """,
        unsafe_allow_html=True,
    )
    raw_inputs = build_input_form(feature_columns, numeric_cols, categorical_cols)
    st.markdown("</div>", unsafe_allow_html=True)

    if not raw_inputs:
        st.info("Preencha o formulário e clique em **Calcular risco** para ver o resultado.")
        return

    input_df = build_input_dataframe(feature_columns, raw_inputs)

    try:
        proba_risco = float(model.predict_proba(input_df)[0, 1])
    except Exception as e:
        st.error(f"Ocorreu um erro ao calcular a probabilidade de risco: {e}")
        return

    threshold = float(metadata.get("threshold", 0.5))

    # Card de resultado
    st.markdown(
        """
        <div style="padding: 1.25rem 1.5rem; border-radius: 0.75rem;
                    border: 1px solid rgba(148,163,184,0.25);
                    background: rgba(15,23,42,0.85); margin-top: 1rem;">
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Resultado da predição")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            label="Probabilidade estimada de risco",
            value=f"{proba_risco * 100:.1f}%",
        )
        st.write(
            f"Classificação: {'Em risco de defasagem' if proba_risco >= threshold else 'Sem risco identificado'}"
        )
        if "IDA_2023" in input_df.columns and not pd.isna(input_df.at[0, "IDA_2023"]):
            st.caption(f"IDA 2023 utilizado: {input_df.at[0, 'IDA_2023']:.2f}")

    with col2:
        if proba_risco >= threshold:
            st.error(
                "Aluno classificado **em risco de defasagem** pelo modelo.\n\n"
                "- Considere aprofundar a avaliação pedagógica.\n"
                "- Revise o histórico de INDE, IEG, IPS e IPP.\n"
                "- Planeje intervenções e acompanhamento individualizado."
            )
        else:
            st.success(
                "O modelo **não identificou risco elevado de defasagem**.\n\n"
                "Mantenha o acompanhamento regular e monitore possíveis "
                "mudanças em engajamento, desempenho e indicadores socioemocionais."
            )

    st.markdown("### Recomendações pedagógicas automáticas")
    for rec in build_recommendations(input_df, proba_risco, threshold):
        st.write(f"- {rec}")

    st.markdown("### Explicabilidade (SHAP)")
    shap_df = get_shap_explanation(model, input_df)
    if shap_df.empty:
        st.info(
            "SHAP não disponível para este ambiente/modelo no momento "
            "(recurso opcional, exibido quando aplicável)."
        )
    else:
        st.caption("Top variáveis com maior impacto local na predição.")
        st.dataframe(shap_df[["feature", "shap_value"]], use_container_width=True, hide_index=True)

    st.caption(
        "A classificação é feita apenas comparando a probabilidade estimada "
        "com o threshold definido no treinamento do modelo. Nenhum ajuste "
        "adicional é realizado nesta aplicação."
    )

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

