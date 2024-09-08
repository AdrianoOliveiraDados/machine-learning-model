import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from datetime import datetime, timedelta

# Carregar os dados com o engine 'openpyxl'
df = pd.read_excel('Dados-Ans-Modelo.xlsx', sheet_name='Dados', engine='openpyxl')

# Pré-processamento dos dados
df['Sinistralidade'] = df['Despesa_assistencial'] / df['Receita']  # Calcula a sinistralidade como um percentual
label_encoder = LabelEncoder()
df['Operadora_encoded'] = label_encoder.fit_transform(df['Operadora'])

# Filtrando valores inválidos ou fora do intervalo esperado
df = df[(df['Sinistralidade'] >= 0) & (df['Sinistralidade'] <= 1)]

# Criando o modelo de regressão linear
X = df[['Despesa_assistencial', 'Operadora_encoded']]
y = df['Sinistralidade']
model = LinearRegression()
model.fit(X, y)

# Título da aplicação
st.title('Previsão de Sinistralidade')

# Entrada do usuário para despesa e operadora
despesa_input = st.number_input('Digite o valor da Despesa Assistencial:')
operadora_input = st.selectbox('Selecione a Operadora:', df['Operadora'].unique())

# Encontrar a última data disponível nos dados
ultima_data = df['Data'].max()

# Entrada do usuário para data futura
data_input = st.date_input(
    'Selecione a Data para previsão da Sinistralidade (deve ser após a última data disponível nos dados)',
    min_value=ultima_data + timedelta(days=1)  # Data mínima é o dia após a última data
)

# Codificação da operadora selecionada
operadora_encoded = label_encoder.transform([operadora_input])[0]

# Prever a sinistralidade
if st.button('Prever Sinistralidade'):
    sinistralidade_prevista = model.predict([[despesa_input, operadora_encoded]])
    sinistralidade_prevista = np.clip(sinistralidade_prevista, 0, 1)  # Garantir que o resultado esteja entre 0 e 1
    st.write(f'Sinistralidade Prevista {sinistralidade_prevista[0] * 100:.2f}%')

# Filtro dos dados pela operadora selecionada
df_filtrado = df[df['Operadora'] == operadora_input]

# Gráfico de linha para receita e despesa ao longo do tempo
fig = px.line(
    df_filtrado, 
    x='Data', 
    y=['Despesa_assistencial', 'Receita'], 
    labels={'value': 'Valor', 'Data': 'Data', 'variable': 'Tipo'},
    title=f'Linha do Tempo de Receita e Despesa - {operadora_input}',
    color_discrete_map={
        'Receita': 'green',  # Cor verde para Receita
        'Despesa_assistencial': 'red'  # Cor vermelha para Despesa
    }
)
st.plotly_chart(fig)


logo_path = "logo.png"  # Substitua pelo caminho do seu arquivo de logo
st.sidebar.image(logo_path, use_column_width=True)