from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split


def carregar_dados(caminho="data/processed/dados_modelo.csv"):
    df = pd.read_csv(caminho)
    return df


def selecionar_features_e_target(df):
    features = [
        "temperatura_interna",
        "temperatura_ambiente",
        "corrente",
        "consumo_kw",
        "pressao_succao",
        "pressao_descarga",
        "tempo_ciclo_min",
        "partidas_ult_1h",
        "alarme_ativo",
        "media_temp_1h",
        "media_consumo_1h",
        "media_corrente_1h",
        "std_temp_1h",
        "tendencia_temp",
        "qtd_alarmes_24h"
    ]

    target = "falha_24h"

    X = df[features].copy()
    y = df[target].copy()

    return X, y, features, target


def dividir_treino_teste(X, y, test_size=0.2, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def treinar_modelo(X_train, y_train, seed=42):
    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1
    )

    modelo.fit(X_train, y_train)
    return modelo


def avaliar_modelo(modelo, X_test, y_test, features):
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    print("\n=== MATRIZ DE CONFUSÃO ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
    print(classification_report(y_test, y_pred, digits=4))

    auc = roc_auc_score(y_test, y_prob)
    print("\n=== ROC-AUC ===")
    print(round(auc, 4))

    importancia = pd.DataFrame({
        "feature": features,
        "importancia": modelo.feature_importances_
    }).sort_values("importancia", ascending=False)

    print("\n=== IMPORTÂNCIA DAS FEATURES ===")
    print(importancia)

    return y_pred, y_prob, importancia


def salvar_modelo(modelo, importancia):
    pasta_saida = Path("outputs/modelos")
    pasta_saida.mkdir(parents=True, exist_ok=True)

    modelo_path = pasta_saida / "random_forest_falha_24h.pkl"
    importancia_path = pasta_saida / "importancia_features.csv"

    joblib.dump(modelo, modelo_path)
    importancia.to_csv(importancia_path, index=False, encoding="utf-8")

    print("\nModelo salvo em:", modelo_path)
    print("Importância das features salva em:", importancia_path)


def gerar_grafico_matriz_confusao(y_test, y_pred):
    pasta_saida = Path("outputs/graficos")
    pasta_saida.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap="Blues",
        ax=ax
    )
    ax.set_title("Matriz de Confusão - Random Forest")
    fig.tight_layout()

    caminho = pasta_saida / "matriz_confusao.png"
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Gráfico salvo em:", caminho)


def gerar_grafico_importancia(importancia):
    pasta_saida = Path("outputs/graficos")
    pasta_saida.mkdir(parents=True, exist_ok=True)

    importancia_plot = importancia.sort_values("importancia", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importancia_plot["feature"], importancia_plot["importancia"])
    ax.set_title("Importância das Variáveis")
    ax.set_xlabel("Importância")
    ax.set_ylabel("Variável")
    fig.tight_layout()

    caminho = pasta_saida / "importancia_features.png"
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Gráfico salvo em:", caminho)


def gerar_curva_roc(y_test, y_prob):
    pasta_saida = Path("outputs/graficos")
    pasta_saida.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("Curva ROC")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.legend()
    fig.tight_layout()

    caminho = pasta_saida / "curva_roc.png"
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Gráfico salvo em:", caminho)

def main():
    print("Carregando dados...")
    df = carregar_dados()

    print("Selecionando features e target...")
    X, y, features, target = selecionar_features_e_target(df)

    print("Dividindo treino e teste...")
    X_train, X_test, y_train, y_test = dividir_treino_teste(X, y)

    print("Treinando Random Forest...")
    modelo = treinar_modelo(X_train, y_train)

    print("Avaliando modelo...")
    y_pred, y_prob, importancia = avaliar_modelo(modelo, X_test, y_test, features)

    print("Gerando gráficos...")
    gerar_grafico_matriz_confusao(y_test, y_pred)
    gerar_grafico_importancia(importancia)
    gerar_curva_roc(y_test, y_prob)

    print("Salvando artefatos...")
    salvar_modelo(modelo, importancia)

    print("\nConcluído com sucesso.")
    
if __name__ == "__main__":
    main()