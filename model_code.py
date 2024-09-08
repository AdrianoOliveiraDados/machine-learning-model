import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Carregar o modelo MobileNetV2 pré-treinado
model = MobileNetV2(weights='imagenet')

# Função para fazer a previsão
def predict_image(img):
    # Converter para RGB se a imagem tiver um canal alfa (RGBA)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize((224, 224))  # MobileNetV2 requer tamanho 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Fazer a previsão
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    predicted_label = decoded_predictions[0][1]  # Retorna o rótulo da previsão

    # Verificar se o modelo detectou algo relacionado a câncer
    if "tumor" in predicted_label.lower() or "cancer" in predicted_label.lower():
        return "Câncer Detectado"
    else:
        return "Sem Sinais de Câncer"

# Interface do Streamlit
st.title('Previsão de Câncer em Imagens de Raio-X')
uploaded_file = st.file_uploader("Faça o upload de uma imagem de raio-X", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Exibir a imagem carregada
    img = Image.open(uploaded_file)
    st.image(img, caption='Imagem carregada', use_column_width=True)

    # Fazer a previsão
    prediction = predict_image(img)
    st.write(f'Previsão: {prediction}')


# Adicionar o logo no lado superior direito
logo_path = "logo.png"  # Substitua pelo caminho do seu arquivo de logo
st.sidebar.image(logo_path, use_column_width=True)
