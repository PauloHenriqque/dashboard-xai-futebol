# Inteligência Artificial Explicável no Futebol ⚽🤖

**Um Sistema de Apoio à Decisão para Eventos Ofensivos (xG e xP)**

Este repositório contém o código-fonte do *dashboard* interativo, os modelos exportados e os *notebooks* de treinamento do Trabalho de Conclusão de Curso (TCC) apresentado ao curso de **Engenharia de Software da Universidade de Pernambuco (UPE) - Campus Garanhuns**.

## 🎯 Sobre o Projeto
O futebol moderno depende cada vez mais de dados empíricos, mas a adoção de modelos preditivos complexos por comissões técnicas frequentemente esbarra no paradigma da "caixa-preta". Este projeto propõe um Sistema de Apoio à Decisão (SAD) focado em romper essa barreira através da **Inteligência Artificial Explicável (XAI)**. 

O sistema avalia a qualidade de finalizações (*Expected Goals* - xG) e o risco de passes (*Expected Passes* - xP) e utiliza a teoria dos Valores de Shapley (SHAP) para traduzir a matemática do algoritmo em justificativas visuais claras para o usuário.

## 🛠️ Tecnologias e Ferramentas Utilizadas
* **Linguagem:** Python
* **Base de Dados:** StatsBomb Open Data (Mais de 728 mil eventos processados)
* **Machine Learning:** Scikit-Learn, XGBoost (Algoritmo selecionado no benchmarking)
* **Inteligência Explicável (XAI):** SHAP (SHapley Additive exPlanations)
* **Interface (Dashboard):** Streamlit

## 📂 Estrutura do Repositório
Este projeto adota uma arquitetura integrada, contendo tanto a aplicação *web* quanto os *notebooks* de pesquisa:

* `app.py`: Arquivo principal contendo o código-fonte do *dashboard* interativo da Prova de Conceito (PoC).
* `modelo_xg_copa.json` e `modelo_passe_copa.json`: Modelos preditivos finais treinados (XGBoost), serializados e prontos para a inferência em tempo real pela interface.
* `requirements.txt`: Lista de dependências Python necessárias para rodar o *dashboard* na nuvem (Streamlit Community Cloud).
* `notebooks/`: Diretório focado na experimentação e pesquisa de dados. Contém o arquivo `TCC_V2.ipynb` (Google Colab) documentando todo o *pipeline* de extração, pré-processamento, engenharia de recursos, balanceamento de dados e treinamento comparativo dos modelos algorítmicos.

## 👨‍🎓 Autoria
* **Autor:** Paulo Henrique Alves dos Santos
* **Orientador:** Prof. Dr. Elisson da Silva Rocha
* **Instituição:** Universidade de Pernambuco (UPE) - Campus Garanhuns
* **Ano:** 2026
