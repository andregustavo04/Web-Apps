import streamlit as st 
import pickle
import numpy as np
import sklearn
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.badges import badge
import time

st.set_page_config(page_title="ML - CÂNCER DE PULMÃO", layout="centered")


with st.sidebar:
	st.markdown("<h1 style='color:50px'>INFORMAÇÕES</h1>", unsafe_allow_html=True)
	st.markdown("<h5 style='color:orange; font-size:26px'>Modelo</h5>", unsafe_allow_html=True)
	st.markdown("<p style='font-size:20px'>Logistic Regression</p>", unsafe_allow_html=True)
	st.markdown(" <h5 style='color:orange; font-size:26px'>Acurácia</h5>", unsafe_allow_html=True)
	st.markdown("<p style='font-size:20px'>96,08%</p>", unsafe_allow_html=True)
	st.markdown(" <h5 style='color:orange; font-size:26px'>F1 Score</h5>", unsafe_allow_html=True)
	st.markdown("<p style='font-size:20px'>90,56%</p>", unsafe_allow_html=True)

	add_vertical_space(2)
	st.markdown("<h1>Para ver o código, clique no botão abaixo.</h1>", unsafe_allow_html=True)
	st.markdown("<a href='https://github.com/andregustavo04'><button style='width:150px;height:40px;border-radius:10px; background-color:orange; font-size:20px'>GitHub<button></a>", unsafe_allow_html=True)


# Função para pegar os objetos criados na construção do modelo
data = None
def load_data(file):
	with open(file, 'rb') as f:
		data = pickle.load(f)

	return data

# Carregando os dados
data = load_data("logistic_regression_model.pkl")


# Atribuindo os objetos a novas variáveis
modelo = data["model"]
scaler = data["scaler"]

st.header("Diagnóstico de Câncer de Pulmão com Machine Learning🤖")

with st.form("diagnóstico"):
	st.subheader("Preencha as informações abaixo para obter o diagnóstico.")
	col1, col2, col3 = st.columns([2, 2, 2])
	add_vertical_space(1)

	with col1:
		st.write("""##### Sexo""")
		sexo = st.radio("", options=["Homem", "Mulher"], index=0, horizontal=True, label_visibility="collapsed")

	with col2:
		st.write("""##### Idade""")
		idade = st.number_input("", format="%d", min_value=20, max_value=100, step=1, label_visibility="collapsed")

	with col3:
		st.write("""##### Fumante?""")
		fumante = st.radio("Fumante?", options=["Sim", "Não"], index=1, horizontal=True, label_visibility="collapsed")

	col4, col5, col6 = st.columns([2, 2, 2])
	add_vertical_space(1)

	with col4:
		st.write("##### Apresenta dedos amarelos?")
		dedos_amarelos = st.radio("Apresenta dedos amarelados?", options=["Sim", "Não"], index=1, horizontal=True, label_visibility="collapsed")

	with col5:
		st.write("""##### Tem ansiedade?""")
		ansiedade = st.radio("Tem ansiedade?", options=["Sim", "Não"], index=1, horizontal=True, label_visibility="collapsed")

	with col6:
		st.write("""##### Tem pressão alta?""")
		pressao = st.radio("Tem pressão alta?", options=["Sim", "Não"], index=1, horizontal=True, label_visibility="collapsed")

	col7, col8, col9 = st.columns([2, 2, 2])

	with col7:
		st.write("""##### Apresenta alguma doença crônica?""")
		doenca_cronica = st.radio("Apresenta alguma doença crônica?", options=["Sim", "Não"], index=1, horizontal=True, label_visibility="collapsed")

	with col8:
		st.write("""##### Apresenta Fadiga?""")
		fadiga = st.radio("Apresenta Fadiga?", options=["Sim", "Não"], index=1, horizontal=True, label_visibility="collapsed")

	with col9:
		st.write("""##### Tem alergia?""")
		alergia = st.radio("Tem alergia?", options=["Sim", "Não"], index=1, horizontal=True, label_visibility="collapsed")

	col10, col11, col12 = st.columns([2, 2, 2])

	with col10:
		st.write("""##### Apresenta respiração ofegante?""")
		respiracao_ofegante = st.radio("Apresenta respiração ofegante?", options=["Sim", "Não"], index=1, horizontal=True, label_visibility="collapsed")

	with col11:
		st.write("""##### Bebe bebidas alcoólicas?""")
		alcool = st.radio("Bebe bebidas alcoólicas?", options=["Sim", "Não"], index=1, horizontal=True, label_visibility="collapsed")

	with col12:
		st.write("""##### Apresenta tosse?""")
		tosse = st.radio("Apresenta tosse?", options=["Sim", "Não"], index=1, horizontal=True, label_visibility="collapsed")

	col13, col14, col15 = st.columns([2, 2, 2])

	with col13:
		st.write("""##### Tem falta de ar?""")
		falta_de_ar = st.radio("Tem falta de ar?", options=["Sim", "Não"], index=1, horizontal=True, label_visibility="collapsed")

	with col14:
		st.write("""##### Tem dificuldade para engolir?""")
		dificuldade_engolir = st.radio("Tem dificuldade para engolir?", options=["Sim", "Não"], index=1, horizontal=True, label_visibility="collapsed")

	with col15:
		st.write("""##### Apresenta dor no peito?""")
		dor_peito = st.radio("Apresenta dor no peito?", options=["Sim", "Não"], index=1, horizontal=True, label_visibility="collapsed")


	add_vertical_space(1)
	verificar = st.form_submit_button("VER DIAGNÓSTICO✅")

	if verificar:
		if sexo == "Homem":
			sexo = 1
		else:
			sexo = 0

		if fumante == "Sim":
			fumante = 2
		else:
			fumante = 1

		if dedos_amarelos == "Sim":
			dedos_amarelos = 2
		else:
			dedos_amarelos = 1

		if ansiedade == "Sim":
			ansiedade = 2
		else:
			ansiedade = 1

		if pressao == "Sim":
			pressao = 2
		else:
			pressao = 1

		if doenca_cronica == "Sim":
			doenca_cronica = 2
		else:
			doenca_cronica = 1

		if fadiga == "Sim":
			fadiga = 2
		else:
			fadiga = 1

		if alergia == "Sim":
			alergia = 2
		else:
			alergia = 1

		if respiracao_ofegante == "Sim":
			respiracao_ofegante = 2
		else:
			respiracao_ofegante = 1

		if alcool == "Sim":
			alcool = 2
		else:
			alcool = 1

		if tosse == "Sim":
			tosse = 2
		else:
			tosse = 1

		if falta_de_ar == "Sim":
			falta_de_ar = 2
		else:
			falta_de_ar = 1

		if dificuldade_engolir == "Sim":
			dificuldade_engolir = 2
		else:
			dificuldade_engolir = 1

		if dor_peito == "Sim":
			dor_peito = 2
		else:
			dor_peito = 1

		valores = np.array([[sexo, idade, fumante, dedos_amarelos, ansiedade, pressao, doenca_cronica, fadiga, alergia, 
							respiracao_ofegante, alcool, tosse, falta_de_ar, dificuldade_engolir, dor_peito]])

		valores = scaler.transform(valores)

		diagnostico = modelo.predict(valores)

		if diagnostico == "YES":
			st.write("#### :red[DIAGNÓSTICO:] Câncer de pulmão detectado.")
		else:
			st.write("#### :blue[DIAGNÓSTICO:] Não há câncer de pulmão.")



st.warning("Informações sobre o modelo de Machine Learning disponíveis na sidebar. (Clique na setinha no canto superior esquerdo da página caso a sidebar não esteja aberta😉)")
st.error("ATENÇÃO: Este é um modelo construído para aprendizado, portanto nenhuma previsão feita aqui deve ser levada em consideração. Além disso, há outros aspectos que devem ser considerados para fazer este tipo de agnóstico, e a consulta a um profissional da saúde não deve, jamais, ser deixada em segundo plano.")