import xgboost as xgb

print("Tentando carregar modelo_xg_copa.json...")
m1 = xgb.XGBClassifier()
m1.load_model("modelo_xg_copa.json")
print("✅ modelo_xg_copa.json carregou OK!")

print("Tentando carregar modelo_passe_copa.json...")
m2 = xgb.XGBClassifier()
m2.load_model("modelo_passe_copa.json")
print("✅ modelo_passe_copa.json carregou OK!")