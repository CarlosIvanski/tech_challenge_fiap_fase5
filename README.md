# Aplicação de risco de defasagem - Passos Mágicos

Esta aplicação em Streamlit disponibiliza o modelo preditivo de **risco de defasagem** desenvolvido para a Associação Passos Mágicos no contexto do Datathon.

O app utiliza um pipeline de machine learning já treinado (`modelo_risco_defasagem_pipeline.joblib`) e um arquivo de metadados (`threshold_risco_defasagem.json`) que contém o threshold de decisão e informações de avaliação. **Nenhum re-treinamento é feito na aplicação**.

## Como rodar localmente

1. Certifique-se de ter o Python 3.10+ instalado.
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Garanta que os arquivos abaixo estejam na raiz do projeto (mesmo diretório deste README):

- `app.py`
- `modelo_risco_defasagem_pipeline.joblib`
- `threshold_risco_defasagem.json`

4. Execute o Streamlit:

```bash
streamlit run app.py
```

## Uso da aplicação

- A interface está em **português (Brasil)**.
- O usuário preenche, via formulário, os indicadores do aluno (anos de 2022 e 2023).
- Ao clicar em **“Calcular risco”**, o app:
  - monta um `DataFrame` com as features esperadas pelo modelo;
  - calcula a probabilidade de risco usando `predict_proba`;
  - compara o valor com o `threshold` definido em `threshold_risco_defasagem.json`;
  - exibe a probabilidade estimada e a classificação (**em risco** / **sem risco identificado**).

Os resultados são **estimativas** baseadas em histórico e devem apoiar, não substituir, a análise pedagógica.

## Deploy na Streamlit Community Cloud

1. Suba este projeto para um repositório público no GitHub contendo, na raiz:
   - `app.py`
   - `modelo_risco_defasagem_pipeline.joblib`
   - `threshold_risco_defasagem.json`
   - `requirements.txt`
   - `README.md`
2. No site da Streamlit Community Cloud:
   - Crie um novo app apontando para o repositório.
   - Selecione `app.py` como arquivo principal.
   - Confirme que as dependências serão instaladas a partir de `requirements.txt`.

Após o deploy, teste o app com alguns exemplos de alunos (dados de 2022 e 2023) para verificar se as probabilidades e decisões de risco estão coerentes com as obtidas no notebook de machine learning.

