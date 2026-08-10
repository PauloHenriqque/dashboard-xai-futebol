import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import warnings

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
# 1. DICIONÁRIO DE TRADUÇÃO PARA O USUÁRIO
# ==========================================
# Isso ensina o sistema a falar "futebolês" em vez de "tecniquês"
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
# 2. CARREGAMENTO DOS MODELOS TREINADOS (dados reais - Copa do Mundo 2022)
# ==========================================
# Os modelos abaixo são os mesmos treinados no notebook (TCC__1_.ipynb),
# com dados reais de eventos da StatsBomb (Copa do Mundo 2022). Nenhum dado
# sintético é usado aqui: o app apenas carrega os modelos já treinados.
#
# Usamos o formato nativo do XGBoost (.json) em vez de pickle/.pkl.
# O pickle serializa o buffer binário interno do Booster, e a própria
# documentação do XGBoost não garante que isso seja portável entre
# instalações/sistemas operacionais diferentes (foi exatamente essa a causa
# do erro "input stream corrupted" ao mover os .pkl para outra máquina).
# save_model()/load_model() é o método oficialmente suportado para
# persistir modelos de forma portável.
MODELO_XG_PATH = "modelo_xg_copa.json"
MODELO_PASSE_PATH = "modelo_passe_copa.json"


@st.cache_resource
def carregar_modelos():
    try:
        modelo_xg = xgb.XGBClassifier()
        modelo_xg.load_model(MODELO_XG_PATH)

        modelo_passe = xgb.XGBClassifier()
        modelo_passe.load_model(MODELO_PASSE_PATH)
    except (FileNotFoundError, xgb.core.XGBoostError) as e:
        st.error(
            "❌ Não foi possível carregar os modelos treinados "
            f"(`{MODELO_XG_PATH}` / `{MODELO_PASSE_PATH}`). "
            "Verifique se os arquivos .json exportados pelo notebook estão "
            "na mesma pasta do app.py.\n\n"
            f"Detalhe técnico: {e}"
        )
        st.stop()

    explainer_xg = shap.TreeExplainer(modelo_xg)
    explainer_passe = shap.TreeExplainer(modelo_passe)

    return modelo_xg, explainer_xg, modelo_passe, explainer_passe


modelo_xg, explainer_xg, modelo_passe, explainer_passe = carregar_modelos()

# Colunas exatas esperadas por cada modelo, na ordem em que foram treinadas.
# Isso evita erros de shape/ordem quando o XGBoost valida os nomes das features.
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

        # Pé Dominante vs Pé do Chute
        pe_dominante = st.selectbox("Pé Dominante do Jogador", ["Destro", "Canhoto"], key="xg_dom")
        # As 4 categorias abaixo espelham exatamente as classes que o modelo
        # aprendeu a partir do dado real (shot_body_part): Right Foot, Left
        # Foot e Other são colunas explícitas do modelo; "Cabeça" é a
        # categoria de referência (baseline), representada quando todas as
        # dummies de pé ficam em zero.
        pe_chute = st.selectbox(
            "Pé/Parte do corpo usado na Finalização",
            ["Pé Direito", "Pé Esquerdo", "Cabeça", "Outra parte do corpo"],
            key="xg_body"
        )

        # Engenharia de Recursos Espaciais (mesma fórmula do notebook)
        dist_gol = np.sqrt((120 - x_chute) ** 2 + (40 - y_chute) ** 2)

        linha_input_xg = {
            'distancia_gol': dist_gol,
            'under_pressure': 1 if pressao_xg else 0,
            'shot_body_part_Right Foot': 1 if pe_chute == "Pé Direito" else 0,
            'shot_body_part_Left Foot': 1 if pe_chute == "Pé Esquerdo" else 0,
            'shot_body_part_Other': 1 if pe_chute == "Outra parte do corpo" else 0,
        }
        # Reordena as colunas exatamente como o modelo foi treinado
        input_xg = pd.DataFrame([linha_input_xg])[COLUNAS_MODELO_XG]

        # Predição da IA (modelo real, treinado com dados da Copa 2022)
        prob_xg_ai = modelo_xg.predict_proba(input_xg)[0][1]

        # ==========================================
        # REGRA DE NEGÓCIO ADICIONAL: FATOR DE PERNA RUIM
        # ==========================================
        # IMPORTANTE: esta penalidade NÃO faz parte do modelo estatístico nem
        # do SHAP. É uma heurística de domínio aplicada por fora da IA, para
        # simular o efeito conhecido de finalizar com a perna não dominante.
        # O SHAP explica apenas a saída do modelo (prob_xg_ai); a leitura
        # tática sobre "perna dominante x perna do chute" é textual e
        # separada, para não ser confundida com uma explicação do modelo.
        fator_penalidade = 1.0  # 1.0 = sem penalidade (100% do valor)
        usou_perna_ruim = False

        if pe_chute in ("Pé Direito", "Pé Esquerdo"):
            if (pe_dominante == "Destro" and pe_chute == "Pé Esquerdo") or \
               (pe_dominante == "Canhoto" and pe_chute == "Pé Direito"):
                fator_penalidade = 0.65  # reduz a chance de gol em 35%
                usou_perna_ruim = True

        prob_xg_final = prob_xg_ai * fator_penalidade

        st.metric(label="🎯 Probabilidade Final Ajustada (xG)", value=f"{prob_xg_final * 100:.1f}%")
        st.caption(f"Probabilidade bruta do modelo (antes da heurística): {prob_xg_ai * 100:.1f}%")

        if usou_perna_ruim:
            st.warning(
                f"⚠️ **Heurística aplicada (fora do modelo):** finalização com a perna não "
                f"dominante. O xG do modelo ({prob_xg_ai * 100:.1f}%) foi reduzido em 35% "
                f"por essa regra de domínio, não pelo SHAP."
            )

        st.caption(f"Distância Calculada: **{dist_gol:.1f} unidades**")

    with col2:
        # Campo de Futebol 2D
        fig_pitch, ax_pitch = plt.subplots(figsize=(8, 4.5))
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
        pitch.draw(ax=ax_pitch)
        pitch.scatter(x_chute, y_chute, ax=ax_pitch, s=200, color='red', edgecolors='black', zorder=3)
        fig_pitch.patch.set_facecolor('#22312b')
        st.pyplot(fig_pitch)

        st.write("---")
        st.subheader("💡 Leitura Tática da Inteligência Artificial")

        # Cálculo dos valores SHAP em tempo real (uma única vez)
        shap_val = explainer_xg(input_xg)
        idx_dist = COLUNAS_MODELO_XG.index('distancia_gol')
        idx_pressao = COLUNAS_MODELO_XG.index('under_pressure')

        # 1. Explicação da Distância
        # Limiares (+-0.15) são um recorte heurístico para narrativa textual,
        # baseado na ordem de grandeza observada dos valores SHAP na base de
        # teste do notebook; não são um critério estatístico formal.
        shap_dist = shap_val[0].values[idx_dist]
        if shap_dist > 0.15:
            st.success("✅ **Proximidade do Gol**: Excelente posicionamento, muito próximo da baliza!")
        elif shap_dist > 0.0:
            st.info("✔️ **Proximidade do Gol**: Distância favorável para a finalização.")
        elif shap_dist < -0.15:
            st.error("⚠️ **Distância do Gol**: Finalização de longe, reduzindo drasticamente a chance de gol.")
        else:
            st.warning("📉 **Distância do Gol**: Distância intermediária, exige alta precisão do chute.")

        # 2. Explicação da Pressão Defensiva
        if pressao_xg:
            st.error("⚠️ **Pressão Defensiva**: Marcador próximo reduz o tempo de reação e a precisão da jogada.")
        else:
            st.success("✅ **Espaço Livre**: Finalizador sem pressão defensiva direta.")

        # 3. Leitura textual sobre Pé Dominante vs Pé do Chute
        # (Nota: essa leitura descreve a heurística de negócio acima, não um
        # valor SHAP — o modelo em si só sabe se foi Right Foot/Left
        # Foot/Other/Cabeça, não sabe qual é o pé "dominante" do jogador.)
        if pe_chute == "Cabeça":
            st.info("ℹ️ **Tipo de Finalização**: Cabeceio.")
        elif pe_chute == "Outra parte do corpo":
            st.info("ℹ️ **Tipo de Finalização**: Outra parte do corpo (ex: peito, coxa).")
        else:
            if usou_perna_ruim:
                st.error(
                    f"⚠️ **Biomecânica do Chute**: Jogador é **{pe_dominante}**, mas finalizou "
                    f"com o **{pe_chute.lower()}** (perna não dominante). Isso reduz a precisão "
                    f"e a força do impacto."
                )
            else:
                st.success(
                    f"✅ **Biomecânica do Chute**: Finalização realizada com a perna dominante "
                    f"(**{pe_dominante}**), garantindo melhor controle e potência."
                )

        # Menu retrátil com o gráfico técnico do SHAP
        with st.expander("📊 Ver Gráfico Técnico SHAP (Modo Analista)"):
            st.markdown(
                "Decomposição detalhada das contribuições marginais do modelo XGBoost "
                "(valores SHAP calculados sobre a probabilidade bruta do modelo, **antes** "
                "da heurística de perna ruim, que não faz parte do modelo):"
            )
            plt.clf()
            shap.plots.waterfall(shap_val[0], show=False)
            fig_shap = plt.gcf()
            fig_shap.set_size_inches(8, 4)
            st.pyplot(fig_shap)

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
        # Campo de Futebol
        fig_p, ax_p = plt.subplots(figsize=(8, 5))
        pitch_p = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
        pitch_p.draw(ax=ax_p)
        pitch_p.scatter(x_pass, y_pass, ax=ax_p, s=150, color='blue', edgecolors='white', zorder=3)
        fig_p.patch.set_facecolor('#22312b')
        st.pyplot(fig_p)

        st.write("---")
        st.subheader("💡 Leitura Tática da Inteligência Artificial")

        # Pega os valores matemáticos do SHAP
        shap_val_p = explainer_passe(input_p)

        # Traduz os valores para mensagens visuais
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

        # Esconde o gráfico complexo
        with st.expander("📊 Ver Gráfico Técnico SHAP (Modo Cientista de Dados)"):
            st.markdown("Análise marginal do impacto estatístico das coordenadas no sucesso do passe.")
            plt.clf()
            shap.plots.waterfall(shap_val_p[0], show=False)
            fig_shap_p = plt.gcf()
            fig_shap_p.set_size_inches(8, 4)
            st.pyplot(fig_shap_p)