import streamlit as st
import pickle
import os

st.set_page_config(
    page_title="Classificador de Textos com IA",
    page_icon="🤖",
    layout="centered"
)

# Carregar modelo
@st.cache_resource
def carregar_modelo():
    """Carrega o modelo treinado (com cache para performance)"""
    with open("model/modelo.pkl", "rb") as f:
        return pickle.load(f)

modelo = carregar_modelo()

st.title("🤖 Classificador de Textos com IA")
st.write("Esta aplicação classifica textos usando um modelo de IA treinado.")

st.write("Insira o texto abaixo para classificar:")
texto = st.text_area("", height=200, placeholder="Digite ou cole seu texto aqui...")

if st.button("Classificar", type="primary"):
    if texto.strip() == "":
        st.warning("⚠️ Por favor, insira um texto para classificar.")
    else:
        with st.spinner("Classificando..."):
            try:
                resultado = modelo.predict([texto])[0]
                
                if resultado == "spam":
                    st.error(f"🚨 Classificação: **{resultado.upper()}**")
                else:
                    st.success(f"✅ Classificação: **{resultado}**")
                    
            except Exception as e:
                st.error(f"❌ Erro ao classificar: {e}")

# Botão para recarregar modelo
if st.button("🔄 Recarregar Modelo"):
    st.cache_resource.clear()
    modelo = carregar_modelo()
    st.info("Modelo recarregado com sucesso!")

st.divider()
st.write("💡 **Sobre:** A IA é treinada para classificar textos em diferentes categorias com base em um conjunto de dados prévio. A classificação realizada pelo modelo pode ajudar a identificar o conteúdo e o contexto dos textos. Lembre-se: a precisão depende da qualidade e diversidade dos dados de treinamento.")


