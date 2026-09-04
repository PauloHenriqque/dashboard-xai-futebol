import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import warnings
import io

# Ocultando os avisos chatos do matplotlib
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="SAD Tático - XAI no Futebol",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚽ Sistema de Apoio à Decisão Tática com IA Explicável (XAI)")
st.markdown("---")

# ==========================================
# FUNÇÃO DE RENDERIZAÇÃO (CORREÇÃO DE BUG MATPLOTLIB)
# ==========================================
def render_matplotlib_fig(fig):
    """Salva a figura em memória e exibe como imagem, evitando o bug do st.pyplot() no Python 3.13"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    st.image(buf)
    plt.close(fig)

# ==========================================
# 1. DICIONÁRIO DE TRADUÇÃO PARA O USUÁRIO
# ==========================================
dicionario_traducao = {
    'distancia_gol': 'Distância até o gol',
    'under_pressure': 'Pressão da marcação adversária',
    'shot_body_part_Right Foot': 'Finalização com o pé direito',
    'shot_body_part_Left Foot': 'Finalização com o pé esquerdo',
    'shot_body_part_Other': 'Finalização com outra parte do corpo',
    'distancia_passe': 'Distância percorrida pelo passe',
    'X_inicio': 'Avanço no campo (Eixo X)',
    'Y_inicio': 'Posição lateral (Eixo Y)'
}

# ==========================================
# 2. CARREGAMENTO DOS MODELOS TREINADOS (.json)
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
        st.error(
            f"❌ Erro ao carregar os modelos. "
            f"Verifique se `{MODELO_XG_PATH}` e `{MODELO_PASSE_PATH}` "
            f"estão na mesma pasta do app.py.\n\nDetalhe técnico: {e}"
        )
        st.stop()

    explainer_xg = shap.TreeExplainer(modelo_xg)
    explainer_passe = shap.TreeExplainer(modelo_passe)

    return modelo_xg, explainer_xg, modelo_passe, explainer_passe

modelo_xg, explainer_xg, modelo_passe, explainer_passe = carregar_modelos()

COLUNAS_MODELO_XG = modelo_xg.get_booster().feature_names
COLUNAS_MODELO_PASSE = modelo_passe.get_booster().feature_names

# ==========================================
# 3. ABA 1: SIMULADOR DE FINALIZAÇÕES (xG)
# ==========================================
aba_xg, aba_passe = st.tabs(["🎯 Avaliação de Finalização (xG)", "👟 Risco de Passe"])

with aba_xg:
    st.header("Simulador de Qualidade de Chute (Expected Goals - xG)")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Parâmetros do Chute")
        x_chute = st.slider("Posição X (Comprimento do Campo)", 0.0, 120.0, 105.0, key="xg_x")
        y_chute = st.slider("Posição Y (Largura do Campo)", 0.0, 80.0, 40.0, key="xg_y")
        pressao_xg = st.checkbox("Sofrendo Pressão Defensiva?", value=False, key="xg_p")

        pe_dominante = st.selectbox("Pé Dominante do Jogador", ["Destro", "Canhoto"], key="xg_dom")
        pe_chute = st.selectbox(
            "Pé/Parte do corpo usado na Finalização",
            ["Pé Direito", "Pé Esquerdo", "Cabeça", "Outra parte do corpo"],
            key="xg_body"
        )

        dist_gol = np.sqrt((120 - x_chute) ** 2 + (40 - y_chute) ** 2)

        linha_input_xg = {
            'distancia_gol': dist_gol,
            'under_pressure': 1 if pressao_xg else 0,
            'shot_body_part_Right Foot': 1 if pe_chute == "Pé Direito" else 0,
            'shot_body_part_Left Foot': 1 if pe_chute == "Pé Esquerdo" else 0,
            'shot_body_part_Other': 1 if pe_chute == "Outra parte do corpo" else 0,
        }
        
        input_xg = pd.DataFrame([linha_input_xg])[COLUNAS_MODELO_XG]
        prob_xg_ai = modelo_xg.predict_proba(input_xg)[0][1]

        # REGRA DE NEGÓCIO ADICIONAL: FATOR DE PERNA RUIM
        fator_penalidade = 1.0  
        usou_perna_ruim = False

        if pe_chute in ("Pé Direito", "Pé Esquerdo"):
            if (pe_dominante == "Destro" and pe_chute == "Pé Esquerdo") or \
               (pe_dominante == "Canhoto" and pe_chute == "Pé Direito"):
                fator_penalidade = 0.65  
                usou_perna_ruim = True

        prob_xg_final = prob_xg_ai * fator_penalidade

        st.metric(label="🎯 Probabilidade Final Ajustada (xG)", value=f"{prob_xg_final * 100:.1f}%")
        st.caption(f"Probabilidade bruta do modelo (antes da heurística): {prob_xg_ai * 100:.1f}%")

        if usou_perna_ruim:
            st.warning(
                f"⚠️ **Heurística aplicada (fora do modelo):** finalização com a perna não "
                f"dominante. O xG do modelo ({prob_xg_ai * 100:.1f}%) foi reduzido em 35% "
                f"por essa regra de domínio."
            )

        st.caption(f"Distância Calculada: **{dist_gol:.1f} unidades**")

    with col2:
        fig_pitch, ax_pitch = plt.subplots(figsize=(8, 4.5))
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
        pitch.draw(ax=ax_pitch)
        
        ax_pitch.text(30, 40, "DEFESA", color='white', ha='center', va='center', alpha=0.15, fontsize=24, fontweight='bold')
        ax_pitch.text(90, 40, "ATAQUE", color='white', ha='center', va='center', alpha=0.15, fontsize=24, fontweight='bold')
        ax_pitch.annotate("", xy=(85, 75), xytext=(35, 75), arrowprops=dict(arrowstyle="->", color="white", alpha=0.4, lw=2))
        ax_pitch.text(60, 72, "Sentido do Jogo", color='white', ha='center', va='center', alpha=0.4, fontsize=10)
        
        pitch.scatter(x_chute, y_chute, ax=ax_pitch, s=200, color='red', edgecolors='black', zorder=3)
        fig_pitch.patch.set_facecolor('#22312b')
        render_matplotlib_fig(fig_pitch) 

        st.write("---")
        st.subheader("💡 Leitura Tática da Inteligência Artificial")

        shap_val = explainer_xg(input_xg)
        idx_dist = COLUNAS_MODELO_XG.index('distancia_gol')
        
        shap_dist = shap_val[0].values[idx_dist]
        if shap_dist > 0.15:
            st.success("✅ **Proximidade do Gol**: Excelente posicionamento, muito próximo da baliza!")
        elif shap_dist > 0.0:
            st.info("✔️ **Proximidade do Gol**: Distância favorável para a finalização.")
        elif shap_dist < -0.15:
            st.error("⚠️ **Distância do Gol**: Finalização de longe, reduzindo drasticamente a chance de gol.")
        else:
            st.warning("📉 **Distância do Gol**: Distância intermediária, exige alta precisão do chute.")

        if pressao_xg:
            st.error("⚠️ **Pressão Defensiva**: Marcador próximo reduz o tempo de reação e a precisão da jogada.")
        else:
            st.success("✅ **Espaço Livre**: Finalizador sem pressão defensiva direta.")

        if pe_chute == "Cabeça":
            st.info("ℹ️ **Tipo de Finalização**: Cabeceio.")
        elif pe_chute == "Outra parte do corpo":
            st.info("ℹ️ **Tipo de Finalização**: Outra parte do corpo (ex: peito, coxa).")
        else:
            if usou_perna_ruim:
                st.error(f"⚠️ **Biomecânica do Chute**: Jogador é **{pe_dominante}**, mas finalizou com o **{pe_chute.lower()}** (perna não dominante). Isso reduz a precisão e a força do impacto.")
            else:
                st.success(f"✅ **Biomecânica do Chute**: Finalização realizada com a perna dominante (**{pe_dominante}**), garantindo melhor controle e potência.")

        with st.expander("📊 Ver Gráfico Técnico SHAP (Modo Analista)"):
            st.markdown("Decomposição detalhada das contribuições marginais do modelo XGBoost:")
            fig_shap = plt.figure(figsize=(8, 4))
            shap.plots.waterfall(shap_val[0], show=False)
            render_matplotlib_fig(fig_shap) 

# ==========================================
# 4. ABA 2: SIMULADOR DE PASSES
# ==========================================
with aba_passe:
    st.header("Simulador de Risco e Sucesso de Passe")
    col1_p, col2_p = st.columns([1, 2])

    with col1_p:
        st.subheader("⚙️ Parâmetros do Passe")
        x_pass = st.slider("Posição X de Origem", 0.0, 120.0, 80.0, key="pass_x")
        y_pass = st.slider("Posição Y de Origem", 0.0, 80.0, 40.0, key="pass_y")
        dist_pass = st.slider("Distância Percorrida (m)", 2.0, 60.0, 15.0, key="pass_d")
        pressao_p = st.checkbox("Passador Sob Pressão?", value=True, key="pass_pr")

        linha_input_p = {
            'distancia_passe': dist_pass,
            'under_pressure': 1 if pressao_p else 0,
            'X_inicio': x_pass,
            'Y_inicio': y_pass,
        }
        input_p = pd.DataFrame([linha_input_p])[COLUNAS_MODELO_PASSE]

        prob_p = modelo_passe.predict_proba(input_p)[0][1]
        st.metric(label="🎯 Probabilidade de Sucesso", value=f"{prob_p * 100:.1f}%")

    with col2_p:
        fig_p, ax_p = plt.subplots(figsize=(8, 4.5))
        pitch_p = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
        pitch_p.draw(ax=ax_p)
        
        ax_p.text(30, 40, "DEFESA", color='white', ha='center', va='center', alpha=0.15, fontsize=24, fontweight='bold')
        ax_p.text(90, 40, "ATAQUE", color='white', ha='center', va='center', alpha=0.15, fontsize=24, fontweight='bold')
        ax_p.annotate("", xy=(85, 75), xytext=(35, 75), arrowprops=dict(arrowstyle="->", color="white", alpha=0.4, lw=2))
        ax_p.text(60, 72, "Sentido do Jogo", color='white', ha='center', va='center', alpha=0.4, fontsize=10)
        
        pitch_p.scatter(x_pass, y_pass, ax=ax_p, s=150, color='blue', edgecolors='white', zorder=3)
        fig_p.patch.set_facecolor('#22312b')
        render_matplotlib_fig(fig_p) 

        st.write("---")
        st.subheader("💡 Leitura Tática da Inteligência Artificial")

        shap_val_p = explainer_passe(input_p)

        for col, shap_v in zip(input_p.columns, shap_val_p[0].values):
            nome_amigavel = dicionario_traducao.get(col, col)
            if shap_v > 0.15:
                st.success(f"✅ **{nome_amigavel}**: Excelente. Trouxe muita segurança ao passe.")
            elif shap_v > 0.0:
                st.info(f"✔️ **{nome_amigavel}**: Ajudou a manter o passe seguro.")
            elif shap_v < -0.15:
                st.error(f"⚠️ **{nome_amigavel}**: Alto risco. Reduziu drasticamente a chance de acerto.")
            else:
                st.warning(f"📉 **{nome_amigavel}**: Aumentou levemente o risco de interceptação.")

        with st.expander("📊 Ver Gráfico Técnico SHAP (Modo Cientista de Dados)"):
            st.markdown("Análise marginal do impacto estatístico das coordenadas no sucesso do passe.")
            fig_shap_p = plt.figure(figsize=(8, 4))
            shap.plots.waterfall(shap_val_p[0], show=False)
            render_matplotlib_fig(fig_shap_p)