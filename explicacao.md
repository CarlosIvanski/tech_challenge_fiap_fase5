# Frontend da Solução Preditiva — Passos Mágicos

## 1) Papel do frontend no projeto

O frontend em Streamlit é a camada que transforma o modelo preditivo em uma ferramenta de uso prático para a equipe pedagógica da Associação Passos Mágicos.

Ele foi desenvolvido com três objetivos centrais:

- **Operacionalizar o modelo de ML** no dia a dia da instituição.
- **Preservar a fidelidade metodológica** do treinamento (sem re-treinamento e sem alterar threshold).
- **Entregar ação orientada por dados** (probabilidade de risco + classificação + recomendações).

Em termos simples:  
**o notebook valida a ciência de dados; o frontend entrega aplicação real de negócio.**

---

## 2) Arquitetura conceitual da solução

Fluxo de ponta a ponta:

1. Usuário preenche os indicadores do aluno no app.
2. O app monta a entrada no formato esperado pelo modelo.
3. O pipeline treinado retorna a probabilidade de risco (`predict_proba`).
4. O app compara com o threshold otimizado (salvo no JSON).
5. O app exibe:
   - probabilidade,
   - classificação (em risco / sem risco),
   - recomendações pedagógicas automáticas,
   - explicabilidade (SHAP, quando aplicável).

---

## 3) Como o frontend se conecta ao Machine Learning

### 3.1 Carregamento de artefatos (sem re-treinamento)

A aplicação consome artefatos já gerados no notebook:

- `modelo_risco_defasagem_pipeline.joblib` (pipeline completo)
- `threshold_risco_defasagem.json` (threshold, metadados e lista de features)

Isso garante que o frontend **não modifica os resultados do ML**.

### 3.2 Consistência de schema (colunas)

O app utiliza `feature_columns` do JSON para montar o `DataFrame` de entrada com o mesmo schema do treinamento.

Campos vazios viram `NaN` para o pipeline tratar internamente com os mesmos passos de pré-processamento usados no treino.

### 3.3 Regra de decisão

A saída do modelo é probabilística e a decisão final usa o threshold otimizado:

- Se `probabilidade >= threshold` → **Em risco de defasagem**
- Se `probabilidade < threshold` → **Sem risco identificado**

No seu projeto, o threshold salvo é `0.356208...` (arquivo `threshold_risco_defasagem.json`).

---

## 4) Estratégia de UX/UI para contexto educacional

A UI foi desenhada para ser simples e objetiva, sem perder robustez técnica:

- **Campos essenciais de 2023 em destaque** (preenchimento rápido).
- **Histórico 2022 e campos avançados em expansores** (opcional).
- **Select box para variáveis categóricas-chave** (`Gênero` e `Fase Ideal`) para reduzir erro de digitação.
- Mensagens claras de interpretação para público não técnico.
- Resultado com foco em decisão pedagógica, não apenas número técnico.

---

## 5) Funcionalidades implementadas no frontend

- Inserção manual dos indicadores do aluno.
- Cálculo automático de `IDA` (quando notas existem e `IDA` está vazio).
- Probabilidade de risco de defasagem via pipeline treinado.
- Classificação por threshold otimizado do projeto.
- Explicabilidade com SHAP quando o ambiente/modelo permite.
- Recomendações pedagógicas automáticas baseadas no perfil de risco e indicadores.

---

## 6) Governança e segurança metodológica

Para manter integridade do projeto:

- Não há `fit` no app.
- Não há recalibração de threshold no app.
- Não há alteração do pipeline treinado.
- Toda decisão é reproduzível a partir dos artefatos versionados.

Assim, o frontend respeita o princípio:  
**“produção consome modelo; produção não reescreve modelo.”**

---

## 7) Valor para o negócio (Passos Mágicos)

A solução entrega impacto em três frentes:

1. **Prevenção**: identifica risco antes de queda acentuada.
2. **Priorização**: ajuda a equipe a focar onde há maior urgência.
3. **Ação**: converte score em recomendação pedagógica concreta.

Na prática, o app conecta analytics + ML + intervenção pedagógica.

---

## 8) Limitações e uso responsável

- A predição é **estimativa**, não diagnóstico.
- Deve ser usada em conjunto com avaliação pedagógica e psicopedagógica.
- Qualidade de entrada influencia qualidade de saída (“garbage in, garbage out”).
- SHAP é opcional e depende de compatibilidade de ambiente.

---

## 9) Mensagem final para apresentação

> “Nossa entrega não ficou só no modelo. Nós transformamos o modelo em produto utilizável.  
> O frontend em Streamlit mantém fidelidade total ao pipeline treinado, aplica o threshold otimizado e gera recomendações pedagógicas acionáveis.  
> Assim, a Passos Mágicos ganha uma ferramenta prática para identificar risco de defasagem com antecedência e apoiar decisões educacionais com dados.”

---

## 10) Script curto (60–90 segundos)

“Este frontend é a camada operacional do nosso modelo preditivo de risco de defasagem.  
Ele carrega o pipeline treinado e o threshold definido na etapa de machine learning, sem re-treinar ou alterar parâmetros.  
A interface foi simplificada para o uso real da equipe: campos essenciais visíveis, histórico opcional e entradas categóricas por seleção.  
Ao calcular, o sistema retorna probabilidade, classifica o risco e sugere recomendações pedagógicas automáticas.  
Com isso, a análise de dados sai do notebook e vira ação prática para intervenção precoce com foco em impacto educacional.”

---

## 11) Exemplo de Input — Aluno com Baixo Risco

### Dados informados no formulário (2023)

- `Idade_2023`: **12**
- `Gênero_2023`: **Feminino**
- `Fase_Ideal_2023`: **Ágata**
- `INDE_2023`: **8.4**
- `IAA_2023`: **8.1**
- `IEG_2023`: **8.5**
- `IPS_2023`: **8.0**
- `IPP_2023`: **8.2**
- `Mat_2023`: **8.0**
- `Por_2023`: **8.3**
- `Ing_2023`: **8.1**

### Como o app processa esse caso

1. O frontend monta o `DataFrame` no formato esperado pelo pipeline.
2. Se necessario, calcula `IDA_2023` automaticamente a partir de `Mat_2023`, `Por_2023` e `Ing_2023`.
3. O modelo retorna a **probabilidade de risco** com `predict_proba`.
4. O sistema compara a probabilidade com o `threshold` otimizado (`0.3562`).
5. Exibe a classificacao final.

### Resultado esperado na demonstracao

- **Probabilidade de risco**: tende a ficar **abaixo do threshold**.
- **Classificacao**: **Sem risco identificado**.
- **Leitura pedagogica**: o aluno apresenta bons sinais em desempenho, engajamento e aspectos psicossociais, entao a recomendacao e manter acompanhamento regular.