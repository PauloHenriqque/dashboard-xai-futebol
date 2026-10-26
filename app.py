import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import io

matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="SAD Tático - XAI no Futebol",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚽ Sistema de Apoio à Decisão Tática com IA Explicável (XAI)")
st.markdown("---")

def gerar_grafico_estatico(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    st.image(buf)
    plt.close(fig)

# ==========================================
# TRADUTOR DE VARIÁVEIS PARA A LEITURA TÁTICA (EXPANDIDO)
# ==========================================
dicionario_traducao = {
    'distancia_gol': 'Distância até o gol',
    'X': 'Posicionamento no avanço (Eixo X)',
    'Y': 'Posicionamento lateral (Eixo Y)',
    'under_pressure': 'Pressão da marcação',
    'shot_first_time': 'Chute de primeira',
    'shot_body_part_Left Foot': 'Uso do Pé Esquerdo',
    'shot_body_part_Right Foot': 'Uso do Pé Direito',
    'shot_body_part_Other': 'Finalização com outra parte do corpo',
    'play_pattern_From Corner': 'Jogada originada de Escanteio',
    'play_pattern_From Counter': 'Jogada originada de Contra-ataque',
    'play_pattern_From Free Kick': 'Jogada originada de Tiro Livre',
    'play_pattern_From Throw In': 'Jogada originada de Arremesso Lateral',
    'play_pattern_From Goal Kick': 'Jogada originada de Tiro de Meta',
    'play_pattern_From Keeper': 'Jogada iniciada pelo Goleiro',
    'play_pattern_From Kick Off': 'Saída de Bola do Meio-Campo',
    'play_pattern_Other': 'Origem da jogada atípica',
    'play_pattern_Regular Play': 'Ataque Posicional (Bola rolando)',
    'shot_type_Free Kick': 'Cobrança de Falta Direta',
    'shot_type_Open Play': 'Chute com Bola Rolando',
    'shot_type_Penalty': 'Cobrança de Pênalti',
    'pass_length': 'Distância percorrida pelo passe',
    'pass_angle': 'Ângulo de visão do passe',
    'X_inicio': 'Origem do passe (Eixo X)',
    'Y_inicio': 'Origem do passe (Eixo Y)',
    'pass_cross': 'Tentativa de cruzamento',
    'pass_height_High Pass': 'Passe Alto',
    'pass_height_Low Pass': 'Passe Baixo'
}

def gerar_leitura_tatica(shap_vals, colunas, valores_input):
    """
    Lê a matriz SHAP e traduz os impactos para português.
    FILTRO INTELIGENTE: Detecta dinamicamente variáveis One-Hot Encoding e oculta se valor == 0.
    """
    impactos_validos = []
    
    for col, shap_v, val in zip(colunas, shap_vals, valores_input):
        # Regra de negócio dinâmica: se a coluna for dummy/booleana
        eh_binaria = (
            col.startswith('play_pattern_') or 
            col.startswith('shot_body_part_') or 
            col.startswith('shot_type_') or 
            col.startswith('pass_height_') or 
            col in ['under_pressure', 'shot_first_time', 'pass_cross']
        )
        
        # Se for uma coluna de "Sim/Não" e NÃO aconteceu (valor == 0), ignoramos na tela
        if eh_binaria and val == 0:
            continue
            
        impactos_validos.append((col, shap_v))
        
    # Ordena os impactos
    impactos = sorted(impactos_validos, key=lambda x: x[1], reverse=True)
    
    positivos = [(dicionario_traducao.get(f, f), v) for f, v in impactos if v > 0.03]
    negativos = [(dicionario_traducao.get(f, f), v) for f, v in impactos if v < -0.03]
    
    resumo = ""
    if positivos:
        fatores_pos = ", ".join([f"**{p[0]}**" for p in positivos[:2]])
        resumo += f"✅ **A favor:** O que mais elevou a chance de sucesso foi: {fatores_pos}.\n\n"
    if negativos:
        negativos_ord = sorted(negativos, key=lambda x: x[1])
        fatores_neg = ", ".join([f"**{n[0]}**" for n in negativos_ord[:2]])
        resumo += f"⚠️ **Contra:** O que mais reduziu a probabilidade (risco) foi: {fatores_neg}."
        
    if not positivos and not negativos:
        resumo = "⚖️ **Equilíbrio:** O modelo avaliou a jogada de forma neutra em relação à média, dependendo do talento individual."
        
    return resumo

# ==========================================
# 1. CARREGAMENTO DOS MODELOS TREINADOS (.json)
# ==========================================
MODELO_XG_PATH = "modelo_xg_copa.json"
MODELO_PASSE_PATH = "modelo_passe_copa.json"

@st.cache_resource
def carregar_modelos():
    try:
        modelo_xg = xgb.XGBClassifier()
        modelo_xg.load_model(MODELO_XG_PATH)

        modelo_passe = xgb.XGBClassifier()
        modelo_passe.load_model(MODELO_PASSE_PATH)
    except Exception as e:
        st.error(f"❌ Erro ao carregar os modelos: {e}")
        st.stop()

    explainer_xg = shap.TreeExplainer(modelo_xg)
    explainer_passe = shap.TreeExplainer(modelo_passe)

    return modelo_xg, explainer_xg, modelo_passe, explainer_passe

modelo_xg, explainer_xg, modelo_passe, explainer_passe = carregar_modelos()
COLUNAS_MODELO_XG = modelo_xg.get_booster().feature_names
COLUNAS_MODELO_PASSE = modelo_passe.get_booster().feature_names

aba_xg, aba_passe = st.tabs(["🎯 Avaliação de Finalização (xG)", "👟 Risco de Passe (xP)"])

# ==========================================
# 2. ABA 1: SIMULADOR DE FINALIZAÇÕES (xG)
# ==========================================
with aba_xg:
    st.header("Simulador de Qualidade de Chute (Expected Goals - xG)")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Parâmetros do Chute")
        x_chute = st.slider("Posição X (Comprimento do Campo)", 0.0, 120.0, 105.0, key="xg_x")
        y_chute = st.slider("Posição Y (Largura do Campo)", 0.0, 80.0, 40.0, key="xg_y")
        pressao_xg = st.checkbox("Sofrendo Pressão Defensiva?", value=False, key="xg_p")

        st.write("---")
        pe_dominante = st.selectbox("Pé Dominante do Jogador", ["Destro", "Canhoto"], key="xg_dom")
        pe_chute = st.selectbox(
            "Mecânica (Pé/Parte do corpo de finalização)",
            ["Pé Direito", "Pé Esquerdo", "Cabeça", "Outra parte do corpo"],
            key="xg_body"
        )
        
        tipo_jogada = st.selectbox(
            "Origem Tática da Jogada (Play Pattern)",
            ["Ataque Posicional (Regular Play)", "Contra-ataque (From Counter)", 
             "Escanteio (From Corner)", "Tiro Livre (From Free Kick)", "Outro (Lateral, Tiro de Meta, etc.)"],
            key="xg_pattern"
        )

        chute_primeira = st.checkbox("Finalização de Primeira?", value=False, key="xg_first")

        dist_gol = np.sqrt((120 - x_chute) ** 2 + (40 - y_chute) ** 2)

        linha_input_xg = {
            'distancia_gol': dist_gol,
            'X': x_chute,
            'Y': y_chute,
            'under_pressure': 1 if pressao_xg else 0,
            'shot_first_time': 1 if chute_primeira else 0,
            'shot_body_part_Left Foot': 1 if pe_chute == "Pé Esquerdo" else 0,
            'shot_body_part_Other': 1 if pe_chute == "Outra parte do corpo" else 0,
            'shot_body_part_Right Foot': 1 if pe_chute == "Pé Direito" else 0,
            'shot_type_Free Kick': 0, 
            'shot_type_Open Play': 1, 
            'shot_type_Penalty': 0,
            'play_pattern_From Corner': 1 if tipo_jogada == "Escanteio (From Corner)" else 0,
            'play_pattern_From Counter': 1 if tipo_jogada == "Contra-ataque (From Counter)" else 0,
            'play_pattern_From Free Kick': 1 if tipo_jogada == "Tiro Livre (From Free Kick)" else 0,
            'play_pattern_Other': 1 if tipo_jogada == "Outro (Lateral, Tiro de Meta, etc.)" else 0,
            'play_pattern_Regular Play': 1 if tipo_jogada == "Ataque Posicional (Regular Play)" else 0
        }
        
        input_xg = pd.DataFrame([linha_input_xg]).reindex(columns=COLUNAS_MODELO_XG, fill_value=0)
        prob_xg_ai = modelo_xg.predict_proba(input_xg)[0][1]

        fator_penalidade = 1.0  
        usou_perna_ruim = False
        if pe_chute in ("Pé Direito", "Pé Esquerdo"):
            if (pe_dominante == "Destro" and pe_chute == "Pé Esquerdo") or \
               (pe_dominante == "Canhoto" and pe_chute == "Pé Direito"):
                fator_penalidade = 0.65  
                usou_perna_ruim = True

        prob_xg_final = prob_xg_ai * fator_penalidade

        st.metric(label="🎯 Probabilidade Final do Gol (xG)", value=f"{prob_xg_final * 100:.1f}%")
        
        if usou_perna_ruim:
            st.warning(
                f"⚠️ **Regra de Negócio Ativada:** Finalização com a perna não dominante. "
                f"A probabilidade bruta da IA era {prob_xg_ai * 100:.1f}%, mas foi penalizada em 35%."
            )
        else:
            if prob_xg_final >= 0.15:
                st.success("✅ **Veredito:** BOA CHANCE DE GOL")
            else:
                st.error("⚠️ **Veredito:** CHANCE CLARA DIFÍCIL")
            
        st.caption(f"Distância Euclidiana Calculada: **{dist_gol:.1f} m**")

    with col2:
        fig_pitch = plt.figure(figsize=(8, 4.5))
        ax_pitch = fig_pitch.add_subplot(1, 1, 1)
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
        pitch.draw(ax=ax_pitch)
        
        pitch.scatter(x_chute, y_chute, ax=ax_pitch, s=200, color='red', edgecolors='white', zorder=3)
        ax_pitch.plot([x_chute, 120], [y_chute, 40], color='red', linestyle='--', alpha=0.5)

        fig_pitch.patch.set_facecolor('#22312b')
        gerar_grafico_estatico(fig_pitch) 

        st.write("---")
        st.subheader("🧠 Interpretação da IA")
        
        shap_val = explainer_xg(input_xg)
        
        # Agora enviamos os valores de input reais (input_xg.iloc[0].values) para a função
        texto_explicativo = gerar_leitura_tatica(shap_val[0].values, COLUNAS_MODELO_XG, input_xg.iloc[0].values)
        st.info(texto_explicativo)

        with st.expander("📊 Ver Gráfico Técnico SHAP"):
            st.markdown("Decomposição matemática do impacto de cada variável:")
            fig_shap = plt.figure(figsize=(8, 4))
            try:
                shap.plots.waterfall(shap_val[0], show=False)
                gerar_grafico_estatico(fig_shap)
            except Exception as e:
                plt.close(fig_shap)
                fig_shap_fallback = plt.figure(figsize=(8, 4))
                shap.plots.bar(shap_val[0], show=False)
                gerar_grafico_estatico(fig_shap_fallback)

# ==========================================
# 3. ABA 2: SIMULADOR DE PASSES VETORIAL (xP)
# ==========================================
with aba_passe:
    st.header("Simulador de Risco de Passes (Trajetória Vetorial)")
    st.markdown("---")

    col1_p, col2_p = st.columns([1, 2])

    with col1_p:
        st.subheader("⚙️ Trajetória e Contexto")
        x_inicio = st.slider("Saiu de X (Origem)", 0.0, 120.0, 60.0, key="p_xi")
        y_inicio = st.slider("Saiu de Y (Origem)", 0.0, 80.0, 40.0, key="p_yi")
        
        st.markdown("👇")
        
        x_fim = st.slider("Chegou em X (Destino)", 0.0, 120.0, 80.0, key="p_xf")
        y_fim = st.slider("Chegou em Y (Destino)", 0.0, 80.0, 20.0, key="p_yf")
        
        st.write("---")
        altura_passe = st.selectbox("Altura da Bola", ["Rasteiro (Ground)", "Baixo (Low)", "Alto (High)"])
        pressao_p = st.checkbox("Passador Sob Pressão na Origem?", value=True)
        e_cruzamento = st.checkbox("A intenção é um Cruzamento?", value=False)

        dist_calculada = np.sqrt((x_fim - x_inicio)**2 + (y_fim - y_inicio)**2)
        angulo_calculado = np.arctan2(y_fim - y_inicio, x_fim - x_inicio)

        linha_input_p = {
            'pass_length': dist_calculada,
            'pass_angle': angulo_calculado,
            'X_inicio': x_inicio,
            'Y_inicio': y_inicio,
            'under_pressure': 1 if pressao_p else 0,
            'pass_cross': 1 if e_cruzamento else 0,
            'pass_height_High Pass': 1 if altura_passe == "Alto (High)" else 0,
            'pass_height_Low Pass': 1 if altura_passe == "Baixo (Low)" else 0
        }
        
        input_p = pd.DataFrame([linha_input_p]).reindex(columns=COLUNAS_MODELO_PASSE, fill_value=0)
        prob_p = modelo_passe.predict_proba(input_p)[0][1]

        st.metric(label="🎯 Probabilidade de Sucesso do Passe", value=f"{prob_p * 100:.1f}%")

        if prob_p >= 0.50:
            st.success("✅ **Veredito:** PASSE SEGURO")
        else:
            st.error("⚠️ **Veredito:** PASSE ARRISCADO")

        st.caption(f"Distância: **{dist_calculada:.1f}m** | Ângulo: **{angulo_calculado:.2f} rad**")

    with col2_p:
        fig_p = plt.figure(figsize=(8, 4.5))
        ax_p = fig_p.add_subplot(1, 1, 1)
        pitch_p = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
        pitch_p.draw(ax=ax_p)
        
        pitch_p.arrows(x_inicio, y_inicio, x_fim, y_fim, ax=ax_p, color='cyan', width=2, headwidth=4, headlength=4, alpha=0.8)
        pitch_p.scatter(x_inicio, y_inicio, ax=ax_p, s=150, color='blue', edgecolors='white', zorder=3, label='Origem')
        pitch_p.scatter(x_fim, y_fim, ax=ax_p, s=150, color='cyan', edgecolors='black', zorder=3, label='Destino')
        
        ax_p.legend(loc="upper left")
        fig_p.patch.set_facecolor('#22312b')
        gerar_grafico_estatico(fig_p)

        st.write("---")
        st.subheader("🧠 Interpretação da IA")
        
        shap_val_p = explainer_passe(input_p)
        
        texto_explicativo_p = gerar_leitura_tatica(shap_val_p[0].values, COLUNAS_MODELO_PASSE, input_p.iloc[0].values)
        st.info(texto_explicativo_p)

        with st.expander("📊 Ver Gráfico Técnico SHAP"):
            st.markdown("Impacto da geometria tridimensional na taxa de sucesso.")
            fig_shap_p = plt.figure(figsize=(8, 4))
            try:
                shap.plots.waterfall(shap_val_p[0], show=False)
                gerar_grafico_estatico(fig_shap_p)
            except Exception as e:
                plt.close(fig_shap_p)
                fig_shap_p_fallback = plt.figure(figsize=(8, 4))
                shap.plots.bar(shap_val_p[0], show=False)
                gerar_grafico_estatico(fig_shap_p_fallback)