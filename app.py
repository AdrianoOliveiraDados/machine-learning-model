import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# 1. Preparação dos Dados (atualizado)

# Criação de um dataset sintético mais variado
data = {
    'sexo': ['Masculino', 'Feminino', 'Masculino', 'Feminino', 'Masculino', 'Feminino', 'Masculino', 'Feminino', 'Feminino', 'Masculino', 'Masculino', 'Feminino'],
    'procedimento': ['Cesária', 'Consulta Geral', 'Ortopedia', 'Cesária', 'Consulta Geral', 'Ortopedia', 'Cesária', 'Consulta Geral', 'Cesária', 'Ortopedia', 'Consulta Geral', 'Ortopedia'],
    'idade': [25, 30, 40, 28, 35, 50, 27, 45, 32, 60, 33, 37],
    'fraude': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]  # 1 indica fraude, 0 não é fraude
}

df = pd.DataFrame(data)

# Transformação de dados categóricos em numéricos
df['sexo'] = df['sexo'].map({'Masculino': 0, 'Feminino': 1})
df['procedimento'] = df['procedimento'].map({'Cesária': 0, 'Consulta Geral': 1, 'Ortopedia': 2})

# Separação de variáveis preditoras e alvo
X = df[['sexo', 'procedimento', 'idade']]
y = df['fraude']

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinamento do modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliação do modelo
y_pred = model.predict(X_test)
print(f"Acurácia do modelo: {accuracy_score(y_test, y_pred):.2f}")

# 2. Criação da Interface Streamlit

# Adicionar o logo no lado superior direito
logo_path = "logo.png"  # Substitua pelo caminho do seu arquivo de logo
st.sidebar.image(logo_path, use_column_width=True)

# Função para prever fraude
def prever_fraude(sexo, procedimento, idade):
    # Converter entrada do usuário para o formato esperado pelo modelo
    sexo_map = {'Masculino': 0, 'Feminino': 1}
    procedimento_map = {'Cesária': 0, 'Consulta Geral': 1, 'Ortopedia': 2}
    
    # Regras lógicas para detectar fraudes
    if sexo == 'Masculino' and procedimento == 'Cesária':
        return 'Fraude'
    if sexo == 'Feminino' and idade < 12 and procedimento == 'Cesária':
        return 'Fraude'
    
    # Preparação da entrada para o modelo
    entrada = [[sexo_map[sexo], procedimento_map[procedimento], idade]]
    
    # Previsão usando o modelo treinado
    previsao = model.predict(entrada)
    return 'Fraude' if previsao[0] == 1 else 'Não é Fraude'

# Interface do Streamlit
st.title("Prevenção de Fraude")

# Input de dados do usuário
sexo = st.selectbox("Selecione o sexo do paciente:", ['Masculino', 'Feminino'])
procedimento = st.selectbox("Selecione o tipo de procedimento:", ['Cesária', 'Consulta Geral', 'Ortopedia'])
idade = st.slider("Idade do paciente:", 0, 100, 25)

# Botão de previsão
if st.button("Verificar Fraude"):
    resultado = prever_fraude(sexo, procedimento, idade)
    st.write(f"Resultado da Previsão: {resultado}")
    
    # Exibir mensagem explicativa quando idade for menor que 12 e sexo for feminino
    if sexo == 'Feminino' and idade < 12 and procedimento == 'Cesária':
        st.warning("Não foi identificado nenhum registro de uma mulher abaixo de 12 anos que realizou cesária.")

    

    # Botões de "Like" e "Dislike"
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Like"):
            st.write("Você gostou do resultado!")
    with col2:
        if st.button("👎 Dislike"):
            st.write("Você não gostou do resultado.")
