from pathlib import Path
import pandas as pd


def carregar_dados():
    df = pd.read_csv("data/synthetic/telemetria_com_falhas.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def criar_features(df):
    df = df.sort_values(["equipamento_id", "timestamp"]).copy()

    # Agrupar por equipamento
    grupos = df.groupby("equipamento_id")

    # Média última 1h (4 pontos)
    df["media_temp_1h"] = grupos["temperatura_interna"].transform(lambda x: x.rolling(4).mean())
    df["media_consumo_1h"] = grupos["consumo_kw"].transform(lambda x: x.rolling(4).mean())
    df["media_corrente_1h"] = grupos["corrente"].transform(lambda x: x.rolling(4).mean())

    # Desvio padrão temperatura
    df["std_temp_1h"] = grupos["temperatura_interna"].transform(lambda x: x.rolling(4).std())

    # Tendência (diferença entre agora e 1h atrás)
    df["tendencia_temp"] = grupos["temperatura_interna"].transform(lambda x: x - x.shift(4))

    # Contagem de alarmes últimas 24h (96 pontos)
    df["qtd_alarmes_24h"] = grupos["alarme_ativo"].transform(lambda x: x.rolling(96).sum())

    return df


def limpar_dados(df):
    # remover linhas iniciais com NaN
    df = df.dropna().reset_index(drop=True)
    return df


def salvar(df):
    caminho = Path("data/processed/dados_modelo.csv")
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False)
    print("Arquivo salvo em:", caminho)


def main():
    df = carregar_dados()

    print("Criando features...")
    df = criar_features(df)

    print("Limpando dados...")
    df = limpar_dados(df)

    salvar(df)

    print("\nResumo:")
    print(df.head())
    print("\nColunas:")
    print(df.columns)


if __name__ == "__main__":
    main()