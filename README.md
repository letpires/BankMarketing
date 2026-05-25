# Bank Marketing — Predição de conversão em campanhas de telemarketing

Projeto de **Machine Learning** sobre o dataset [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (UCI): um banco português realizou campanhas para oferecer **depósito a prazo**. O objetivo é identificar clientes com maior probabilidade de aceitar a oferta, reduzindo custo de ligações e priorizando quem mais converte.

O fluxo completo está no notebook **`BancoMarketing.ipynb`** (EDA, pré-processamento, balanceamento, modelagem com **LightGBM** e exportação do modelo). A interface web em **Streamlit** (`app.py`) consome dados e artefatos pelo **Amazon S3**.

Projeto Integrador do Nível 1 da formação em Data Science da **Alura**, com foco em análise de dados, machine learning e computação em nuvem.

---

## Estrutura do repositório

| Arquivo / pasta | Descrição |
|-----------------|-----------|
| `BancoMarketing.ipynb` | Pipeline completo: EDA, feature engineering, balanceamento, LazyPredict, LightGBM, upload S3 |
| `app.py` | App Streamlit (EDA, simulador de campanha, inferência por cliente) |
| `requirements.txt` | Dependências Python |
| `novos_clientes_simulacao.csv` | Exemplos de clientes para testar inferência (mesmo schema do dataset) |
| `imagens_ilustrativas/` | Diagramas usados no notebook (balanceamento, LightGBM, threshold) |
| `dados/` | Dados disponíveis da página UCI |

---

## Pré-requisitos

| Ferramenta | Uso |
|------------|-----|
| **Python 3.10+** (recomendado) | Ambiente de execução |
| **AWS CLI** | Configurar credenciais e acesso ao S3 |
| **Conta AWS** | Bucket S3 com os dados e modelos |

### Instalar AWS CLI

- **macOS (Homebrew):** `brew install awscli`
- **Windows:** [Instalador oficial](https://aws.amazon.com/cli/) ou Chocolatey
- **Linux:** siga a [documentação da AWS](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

### Configurar credenciais AWS

1. No console AWS: **IAM** → usuário → **Security credentials** → **Create access key** (guarde Access Key e Secret com segurança).
2. No terminal:

```bash
aws configure
```

Informe:

- **AWS Access Key ID** e **Secret Access Key**
- **Default region name** (ex.: `us-east-1` ou a região do seu bucket)
- **Default output format** (pode deixar `json`)

O `boto3` usado no notebook e no `app.py` utiliza essas credenciais (ou variáveis de ambiente equivalentes).

### Bucket S3 esperado

O código assume o bucket **`bank-marketing-alura`** com esta estrutura:

| Caminho no bucket | Conteúdo |
|-------------------|----------|
| `raw/bank-full.csv` | Dataset bruto (`sep=';'`) |
| `modelos/modelo_lightgbm.pkl` | Modelo LightGBM treinado |
| `modelos/X_test_pre.pkl` | Features de teste pré-processadas |
| `modelos/y_test.pkl` | Rótulos do conjunto de teste |
| `modelos/ohe.pkl` | One-Hot Encoder (categóricas) |
| `modelos/oe.pkl` | Ordinal Encoder (`education`) |
| `modelos/scaler_age.pkl` | StandardScaler da idade |
| `modelos/scaler_balance.pkl` | StandardScaler do saldo (após log) |

Se usar outro nome de bucket ou prefixos, ajuste a constante `bucket` em `app.py` (e no notebook, onde houver upload/leitura do S3).

---

## Ambiente virtual (venv)

Na pasta do projeto:

```bash
cd /caminho/para/BankMarketing

python3 -m venv .venv
```

Ativar o ambiente:

- **macOS / Linux:** `source .venv/bin/activate`
- **Windows (cmd):** `.venv\Scripts\activate.bat`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`

Instalar dependências:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Dependências principais (`requirements.txt`)

`boto3`, `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `joblib`, `streamlit`, `plotly`, `imbalanced-learn`, `lazypredict`, `matplotlib`, `seaborn`, `ipykernel`.

---

## Notebook `BancoMarketing.ipynb`

Resumo do que o notebook cobre:

1. **Setup e EDA** — Problema de negócio, AWS/S3, carregamento do CSV, exploração das variáveis e desbalanceamento de classes (~88% `no` / ~12% `yes`).
2. **Preparação para ML** — `train_test_split`, engenharia de features (One-Hot, Ordinal, StandardScaler, bins de `campaign`, tratamento de `balance`), balanceamento com **RandomOverSampler** + **RandomUnderSampler** (`imblearn`).
3. **Modelagem** — **LazyPredict** para varredura inicial; comparação manual (LightGBM, XGBoost, regressão logística); escolha do **LightGBM**; validação cruzada e matriz de confusão; persistência com `joblib` e upload para o S3.
4. **Streamlit** — Ideia e implementação do app: EDA + simulador de campanha com threshold e impacto em ligações/lucro + inferência individual.

Execute o notebook no Jupyter, VS Code ou Cursor com o kernel apontando para o mesmo ambiente onde instalou o `requirements.txt`.

> **Nota:** a variável `duration` (duração da ligação) é removida no pipeline de inferência do app porque só é conhecida **depois** do contato — usá-la na predição causaria *data leakage*.

---

## Aplicação Streamlit (`app.py`)

Com o venv ativado e o `aws configure` válido para uma conta que tenha **leitura** no bucket:

```bash
streamlit run app.py
```

Abra o endereço que o Streamlit mostrar no navegador (geralmente `http://localhost:8501`).

### Funcionalidades do app

| Página | O que faz |
|--------|-----------|
| **Análise exploratória** | Métricas resumidas, amostra dos dados, histograma de idade por conversão, taxa por profissão e série mensal |
| **Simulador de campanha** | Threshold sobre probabilidades, métricas de ligações/conversões, simulação financeira (receita/custo/lucro), curva de lucro por threshold, comparação com “ligar para todos”, matriz de confusão |
| **Inferência cliente** | Formulário com features do cliente; `predict_proba` + decisão “Ligar” / “Não ligar” conforme o threshold |

---

## Leitura complementar

Diagramas no repositório: `imagens_ilustrativas/` (LightGBM, threshold, balanceamento, pipeline de modelagem).

| Tema | Links |
|------|-------|
| **LightGBM** | [Documentação](https://lightgbm.readthedocs.io/en/stable/) · [GeeksForGeeks](https://www.geeksforgeeks.org/machine-learning/lightgbm-light-gradient-boosting-machine/) |
| **LazyPredict** | [GitHub](https://github.com/shankarpandala/lazypredict) · [Documentação](https://lazypredict.readthedocs.io/en/latest/) |
| **Threshold e métricas** | [Threhsold and confusion matrix](https://developers.google.com/machine-learning/crash-course/classification/thresholding) · [Classification Thresholds in ML](https://medium.com/@yasvanthkohli/classification-thresholds-in-machine-learning-6d7cf3cf38a4) · [Otimização de threshold - exemplo prático](https://medium.com/codex/optimize-your-decision-threshold-or-why-ml-model-metrics-are-not-always-helpful-to-build-a-b84821c51e2) |
| **Balanceamento de classes** | [imbalanced-learn — guia](https://imbalanced-learn.org/stable/) · [Combining Oversampling and Undersampling](https://machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification/) |
| **Data leakage** | [Encoding Before vs After Train_Test_Split?](https://www.geeksforgeeks.org/machine-learning/encoding-before-vs-after-train_test_split/) |

---

## Melhorias futuras (roadmap)

O projeto é um ponto de partida sólido para decisão baseada em dados. Caminhos sugeridos para aumentar eficácia, interpretabilidade e impacto operacional:

### Pipeline de modelagem em múltiplas etapas

Hoje perfil do cliente e estratégia de campanha entram na mesma predição. Evolução natural: dois modelos.

- [ ] **Modelo 1 — Perfil do cliente:** probabilidade de conversão
- [ ] **Modelo 2 — Estratégia de campanha:** como, quando e com qual abordagem contatar

Benefícios esperados: maior interpretabilidade, controle das decisões de negócio e otimização independente de cada etapa.

### Otimização de hiperparâmetros

O LightGBM foi treinado com parâmetros padrão. Próximo passo: `GridSearchCV`, Random Search ou **Optuna** (recomendado para produção), ajustando por exemplo `learning_rate`, `num_leaves` e `max_depth`.

- [ ] Busca sistemática com validação cruzada (foco em ROC AUC ou F1 da classe positiva)
- [ ] Comparar modelo tunado vs baseline atual

Resultado esperado: melhor capacidade preditiva e maior estabilidade.

### Engenharia de features temporais

Explorar melhor `day` e `month`:

- [ ] Agrupar dias em faixas (início, meio e final do mês)
- [ ] Identificar sazonalidade (meses com maior conversão, períodos de maior engajamento)
- [ ] Validação temporal (split por `month`/`year`) para evitar vazamento entre campanhas

### Otimização do threshold (ponto de corte)

O threshold não é só decisão técnica — depende de valor médio por conversão, custo por ligação e capacidade do time.

- [ ] Aprofundar simulação de cenários no app (já há base no Simulador de campanha)
- [ ] Documentar política de corte como decisão estratégica, não fixa em 0,50

### Segmentação e clusterização

- [ ] Aplicar clusterização (ex.: K-Means) para perfis distintos (engajados, alto saldo, resistentes à conversão)
- [ ] Campanhas personalizadas por segmento

### Integração com o CRM

- [ ] Mapear fluxo atual: distribuição de leads, registro de resultados das ligações
- [ ] Priorização automática a partir do score do modelo
- [ ] API REST (**FastAPI**) ou endpoint para o CRM consumir predições

### Análise de duração das ligações

A variável `duration` foi removida do pipeline de inferência (evita leakage), mas pode ser estudada offline:

- [ ] Custo real por interação e ROI com duração média
- [ ] Relação entre tempo de ligação e conversão

### Simulação de cenários de negócio

Evoluir o simulador atual com visão financeira completa:

- [ ] Receita estimada, custo total da campanha e lucro por threshold
- [ ] Comparar políticas: menos clientes com alta chance vs muitos com chance moderada

### Visualização do fluxo real

- [ ] Diagrama do fluxo para stakeholders: base → score → threshold → lista priorizada → campanha → resultado financeiro

### Deploy em produção

- [ ] Streamlit na AWS, GCP ou Azure
- [ ] APIs para CRM; monitoramento de performance e drift do modelo
- [ ] Pipeline sklearn único (pré-processamento + modelo) para alinhar notebook e app
- [ ] Permissões S3 mínimas (IAM só leitura nos prefixos necessários)

Objetivo: disponibilidade, escalabilidade e uso contínuo pelo time operacional.

---

## Outros datasets para treinar e comparar

Úteis para generalizar o pipeline (telemarketing, crédito, churn, marketing direto):

| Dataset | Fonte | Por que usar |
|---------|-------|--------------|
| **Bank Marketing** (este) | [UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing) | Baseline do projeto; telemarketing bancário |
| **Telco Customer Churn** | [Kaggle / IBM](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | Churn em telecom; forte desbalanceamento; bom para threshold e custo de retenção |
| **Credit Card Fraud Detection** | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Extremo desbalanceamento; métricas de raro evento |
| **Give Me Some Credit** | [Kaggle](https://www.kaggle.com/competitions/GiveMeSomeCredit) | Default de crédito; probabilidade de inadimplência |
| **Online Shoppers Purchasing Intention** | [UCI](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) | Conversão em e-commerce; sessões como observações |


