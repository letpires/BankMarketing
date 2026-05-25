import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import boto3
import joblib

st.set_page_config(page_title="Bank Marketing", page_icon=":money_with_wings:", layout="wide")

s3 = boto3.client('s3')

s3.download_file('bank-marketing-alura', 'modelos/modelo_lightgbm.pkl', '/tmp/modelo_lightgbm.pkl')
modelo = joblib.load('/tmp/modelo_lightgbm.pkl')

s3.download_file('bank-marketing-alura', 'raw/bank-full.csv', '/tmp/bank-full.csv')
df = pd.read_csv('/tmp/bank-full.csv', sep=';')

def carregar_arquivo_s3(bucket, key, tipo='joblib', sep = ';'):
    caminho_tmp = f"/tmp/{key.split('/')[-1]}"


    s3.download_file(bucket, key, caminho_tmp)

    if tipo == 'csv':
        return pd.read_csv(caminho_tmp, sep=sep)

    return joblib.load(caminho_tmp)

bucket = 'bank-marketing-alura'

modelo = carregar_arquivo_s3(bucket, 'modelos/modelo_lightgbm.pkl')

df = carregar_arquivo_s3(bucket, 'raw/bank-full.csv', tipo='csv')

X_test = carregar_arquivo_s3(bucket, 'modelos/X_test_pre.pkl')
y_test = carregar_arquivo_s3(bucket, 'modelos/y_test.pkl')

oe = carregar_arquivo_s3(bucket, 'modelos/oe.pkl')
scaler_age = carregar_arquivo_s3(bucket, 'modelos/scaler_age.pkl')
scaler_balance = carregar_arquivo_s3(bucket, 'modelos/scaler_balance.pkl')
ohe = carregar_arquivo_s3(bucket, 'modelos/ohe.pkl')


# ── Previsão ──────────────────────────────────────────────────────────────
y_prob = modelo.predict_proba(X_test)[:, 1]

# ── Preparação dos dados ──────────────────────────────────────────────────────────────
def preparar_novos_clientes(novos_clientes, ohe, oe, scaler, scaler_balance, colunas_treino):
    X_new = novos_clientes.copy()

    cat_cols = ['month', 'job', 'marital', 'contact']

    new_encoded = pd.DataFrame(
        ohe.transform(X_new[cat_cols]),
        columns=ohe.get_feature_names_out(cat_cols),
        index=X_new.index
    )

    X_new = pd.concat([X_new.drop(columns=cat_cols), new_encoded], axis=1)

    X_new['education_ord'] = oe.transform(X_new[['education']])
    X_new.drop(columns=['education'], inplace=True)

    X_new['age_t'] = scaler.transform(X_new[['age']])
    X_new.drop(columns=['age'], inplace=True)

    X_new['balance_sign'] = np.sign(X_new['balance'])
    X_new['balance_log'] = np.log(X_new['balance'].abs() + 1)
    X_new['balance_log'] = scaler_balance.transform(X_new[['balance_log']])
    X_new.drop(columns=['balance'], inplace=True)

    bins = [0, 1, 3, 6, float('inf')]
    labels = ['1_contato', '2_3_contatos', '4_6_contatos', '7+_contatos']

    X_new['campaign_bin'] = pd.cut(X_new['campaign'], bins=bins, labels=labels)
    X_new.drop(columns=['campaign'], inplace=True)

    X_new = pd.get_dummies(X_new, columns=['campaign_bin'])

    binary_map = {'no': 0, 'yes': 1}

    for col in ['default', 'housing', 'loan']:
        X_new[f'{col}_bin'] = X_new[col].map(binary_map)

    X_new.drop(columns=['default', 'housing', 'loan'], inplace=True)

    if 'pdays' in X_new.columns:
        X_new.drop(columns=['pdays'], inplace=True)

    if 'duration' in X_new.columns:
        X_new.drop(columns=['duration'], inplace=True)

    X_new['previous'] = (X_new['previous'] > 0).astype(int)

    poutcome_map = {
        'unknown': 0,
        'failure': 1,
        'other': 1,
        'success': 2
    }

    X_new['poutcome'] = X_new['poutcome'].map(poutcome_map).fillna(0)

    X_new = X_new.reindex(columns=colunas_treino, fill_value=0)

    return X_new


# ---

pagina = st.sidebar.radio(
    "Navegação",
    ["Análise Exploratória", "Simulador de Campanha", "Inferência Cliente"]
)


if pagina == "Análise Exploratória":

    st.title("Análise Exploratória da Campanha de Marketing Bancário")

    total_clientes = len(df)

    total_aceitaram = (df['y'] == 'yes').sum()

    taxa_conversao = total_aceitaram / total_clientes * 100 if total_clientes > 0 else 0

    col1, col2, col3 = st.columns(3)

    col1.metric("Total de Clientes", f"{total_clientes:,}".replace(',', '.'))
    col2.metric("Aceitaram a oferta", f"{total_aceitaram:,}".replace(',', '.'))
    col3.metric("Taxa de Conversão", f"{taxa_conversao:.1f}%")


    with st.expander("Visualizar as primeiras linhas"):

        st.dataframe(df.head(10))

    
    st.markdown("---")

    st.subheader("Perfil do Cliente")

    col_a, col_b = st.columns(2)

    with col_a:

        st.markdown("**Distribuição de idade por conversão**")

        df_idade = df.copy()

        df_idade["Resposta"] = df_idade['y'].map({'yes': 'Aceitou', 'no': 'Não aceitou'})

        fig_age = px.histogram(
            df_idade,
            x="age",
            color="Resposta",
            nbins = 25,
            barmode = "overlay",
            histnorm = "percent",
            color_discrete_map = {"Aceitou": "#1565C0", "Não aceitou": "#FFCDD2"},
        )
        fig_age.update_layout(
            xaxis_title="Idade",
            yaxis_title="Percentual de Clientes",
            legend_title="Resposta",
            template = "simple_white"
        )


        st.plotly_chart(fig_age, use_container_width=True)


    with col_b:

        st.markdown("**Taxa de Conversão por Profissão**")

        grupo_prof = (
            (df["y"] == "yes")
            .groupby(df["job"])
            .mean()
            .reset_index(name="taxa_conversao")
        )

        grupo_prof = grupo_prof.sort_values("taxa_conversao", ascending=False)

        fig_prof = px.bar(
            grupo_prof,
            y="job",
            x="taxa_conversao",
            orientation="h",
            color="taxa_conversao",
            color_continuous_scale="Blues",
            text=grupo_prof["taxa_conversao"].apply(lambda x: f"{x*100:.1f}%"),
        )
        fig_prof.update_traces(textposition="outside")

        fig_prof.update_layout(
            xaxis_title="Taxa de Conversão (%)",
            yaxis_title="Profissão",
            coloraxis_showscale=False,
            template="simple_white",
            margin=dict(l=60))
        
        fig_prof.update_xaxes(tickformat=".0%")


        st.plotly_chart(fig_prof, use_container_width=True)

# ---

    st.subheader("Taxa de conversão por mês")
    
    ordem_meses = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    conv_mes = (df.groupby('month')['y'].agg(total="count", convertidos = lambda x: (x == "yes").sum())
                .reindex(ordem_meses).reset_index()
    )


    conv_mes['taxa'] = conv_mes["convertidos"] / conv_mes["total"] * 100

    fig_mes = px.line(
        conv_mes,
        x="month",
        y = "taxa",
        markers=True,
        labels = {"month": "Mês", "taxa": "Taxa de Conversão (%)"},
        color_discrete_sequence = ["#1a56db"]
    )

    fig_mes.update_traces(
        text = conv_mes["taxa"].apply(lambda x: f"{x:.1f}%"),
        textposition = "top center",
        mode= "lines+markers+text"
    )

    fig_mes.update_layout(
        xaxis_title = "Mês",
        yaxis_title = "Taxa de Conversão (%)",
        margin = dict(t=30, b=10))

    st.plotly_chart(fig_mes, use_container_width=True)


elif pagina == "Simulador de Campanha":

    st.title("Simulador de Campanha de Marketing Bancário")
    st.caption("Ajuste os parâmetros financeiros e o limiar de decisão para estimar o impacto da campanha.")

    st.subheader("Parâmetros da campanha")

    col_p1, col_p2 = st.columns(2)

    with col_p1:

        valor_conversao = st.number_input(
            "Valor médio por conversão (R$)",
            min_value=0.0,
            value=500.0,
            step=50.0
        )
    
    with col_p2:
        custo_ligacao = st.number_input(
            "Custo por ligação (R$)",
            min_value=0.0,
            value=45.0,
            step=5.0
        )
    
    st.markdown("---")
    st.subheader("Limiar de decisão")

    threshold = st.slider(
        "Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        format="%.2f"
    )


    y_pred = (y_prob >= threshold).astype(int)

    TP = int(((y_pred == 1) & (y_test == 1)).sum())
    FP = int(((y_pred == 1) & (y_test == 0)).sum())
    FN = int(((y_pred == 0) & (y_test == 1)).sum())
    TN = int(((y_pred == 0) & (y_test == 0)).sum())

    total = len(y_test)
    conversores = int(y_test.sum())
    ligacoes = TP + FP
    economizadas = total - ligacoes

    receita = TP * valor_conversao
    custo_campanha = ligacoes * custo_ligacao
    lucro = receita - custo_campanha


    st.markdown("---")
    
    k1, k2, k3 = st.columns(3)

    with k1:
        with st.container(border=True):
            st.metric("📞 Ligações necessárias", f"{ligacoes:,}", f"de {total:,} na base")


    with k2:
        with st.container(border=True):
            st.metric("✅ Conversores capturados", f"{TP:,}", f"{TP/conversores*100:.0f}% do total")


    with k3:
        with st.container(border=True):
            st.metric("💰 Ligações economizadas", f"{economizadas:,}", f"{economizadas/total*100:.0f}% da base")

    k4, k5, k6 = st.columns(3)

    with k4:
        with st.container(border=True):
            st.metric("💸 Custo total", f"R${custo_campanha:,.0f}")

    with k5:
        with st.container(border=True):
            st.metric("📈 Receita estimada", f"R${receita:,.0f}")

    with k6:
        with st.container(border=True):
            st.metric("💵 Lucro estimado", f"R${lucro:,.0f}")

    
    st.markdown("---")

    st.subheader("Lucro estimado por threshold")

    pontos = [round(x/100, 2) for x in range(10, 91, 5)]

    resumo_lucro = []

    for corte in pontos:

        y_pred_corte = (y_prob >= corte).astype(int)


        tp_c = int(((y_pred_corte == 1) & (y_test == 1)).sum())

        fp_c = int(((y_pred_corte == 1) & (y_test == 0)).sum())

        lig_c = tp_c + fp_c

        resumo_lucro.append({
            "Threshold": corte,
            "Ligações": lig_c,
            "Conversões": tp_c,
            "Lucro": tp_c * valor_conversao - lig_c * custo_ligacao
        })
    
    df_lucro = pd.DataFrame(resumo_lucro)


    melhor = df_lucro.loc[df_lucro["Lucro"].idxmax()]

    fig_lucro = px.line(
        df_lucro,
        x = "Threshold",
        y= "Lucro",
        markers=True,
        labels = {"Lucro": "Lucro estimado (R$)"},
        color_discrete_sequence = ["#1a56db"]
    )

    fig_lucro.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Atual {threshold:.2f}",
        annotation_position="top right",
        annotation_font_color="red"
    )


    fig_lucro.add_vline(
        x=melhor['Threshold'],
        line_dash="dash",
        line_color="green",
        annotation_text=f"Ótimo {melhor['Threshold']:.2f}",
        annotation_position="top left",
        annotation_font_color="green"
    )

    fig_lucro.update_layout(
        margin=dict(t=30, b=10)
    )

    st.plotly_chart(fig_lucro, use_container_width=True)



    st.markdown("---")
    st.subheader("Modelo atual vs ligar para todos")

    lucro_baseline = conversores * valor_conversao - total * custo_ligacao
    diferenca_lucro = lucro - lucro_baseline

    col_b1, col_b2, col_b3 = st.columns(3)

    col_b1.metric("Lucro com modelo", f"R${lucro:,.0f}")
    col_b2.metric("Lucro ligando para todos", f"R${lucro_baseline:,.0f}")
    col_b3.metric("Diferença", f"R${diferenca_lucro:,.0f}")


    st.markdown("---")
    st.subheader("Matriz de confusão")

    col_cm, col_leg = st.columns([3, 2])

    with col_cm:

        text_matrix = [
            [f"TN<br>{TN:,}", f"FP<br>{FP:,}"],
            [f"FN<br>{FN:,}", f"TP<br>{TP:,}"],
        ]

        fig_cm = go.Figure(go.Heatmap(

            z = [[TN, FP], [FN, TP]],

            x = ["Previsto: Não", "Previsto: Sim"],
            y = ["Real: Não", "Real: Sim"],

            text = text_matrix,
            texttemplate = "%{text}",
            textfont = dict(color="black", size=18),
            colorscale = [[0, "#eef2ff"], [1, "#1a56db"]],
            showscale = False,
        ))

        fig_cm.update_layout(
            xaxis = dict(side="top"),
            height = 500,
            margin = dict(t=60, b=10, l=10, r=10),
        )

        st.plotly_chart(fig_cm, use_container_width=True)
    

    with col_leg:

        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        st.markdown(f"""
        **TN {TN:,}**
        Não ia converter e modelo disse não.
        Nenhum custo, nenhuma perda.

        ---

        **FP {FP:,}**
        Não ia converter mas modelo disse sim.

        ---

        **FN {FN:,}**
        Ia converter mas modelo disse não.

        ---

        **TP {TP:,}**
        Ia converter e modelo disse sim.
        """)























elif pagina == "Inferência Cliente":

    st.title("Inferência de Cliente")
    st.caption("Preencha os dados do cliente para estimar a probabilidade de conversão.")


    threshold_cliente = st.slider(
        "Threshold para decisão",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        format="%.2f"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input("Idade", min_value=18, max_value=100, value=35)
        job = st.selectbox("Profissão", sorted(df["job"].unique()))
        marital = st.selectbox("Estado Civil", sorted(df["marital"].unique()))
        education = st.selectbox("Educação", sorted(df["education"].unique()))
        default = st.selectbox("Inadimplente?", ["no", "yes"])

    with col2:

        balance = st.number_input("Saldo médio (R$)", value=1000.0)
        housing = st.selectbox("Financiamento habitacional?", ["no", "yes"])
        loan = st.selectbox("Empréstimo pessoal?", ["no", "yes"])
        contact = st.selectbox("Canal de contato", sorted(df["contact"].unique()))
        day = st.number_input("Dia do mês", min_value=1, max_value=31, value=15)
    
    with col3:

        month = st.selectbox("Mês", sorted(df["month"].unique()))
        campaign = st.number_input("Número de contatos", min_value=0, value=1)
        previous = st.number_input("Número de contatos anteriores", min_value=0, value=0)
        poutcome = st.selectbox("Resultado da última campanha", sorted(df["poutcome"].unique()))
    

    novo_cliente = pd.DataFrame([{
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day,
        'month': month,
        'campaign': campaign,
        'previous': previous,
        'poutcome': poutcome
    }])

    if st.button("Rodar inferência"):

        X_novo = preparar_novos_clientes(
            novo_cliente,
            ohe = ohe,
            oe = oe,
            scaler = scaler_age,
            scaler_balance = scaler_balance,
            colunas_treino = X_test.columns
        )

        prob = modelo.predict_proba(X_novo)[:, 1][0]

        decisao = "Ligar" if prob >= threshold_cliente else "Não ligar"

        st.subheader("Resultado da inferência")

        col_a, col_b = st.columns(2)

        col_a.metric("Probabilidade conversão", f"{prob:.2%}")
        col_b.metric("Decisão", decisao)

        if decisao == "Ligar":
            st.success("Vale a pena ligar para este cliente com o threshold atual.")
        
        else:
            st.warning("Com esse threshold, este cliente não seria priorizado para a campanha.")
    
    with st.expander("Visualizar dados do cliente"):
        st.dataframe(novo_cliente, use_container_width=True)