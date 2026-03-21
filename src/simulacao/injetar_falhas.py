from pathlib import Path
import numpy as np
import pandas as pd


def carregar_telemetria(caminho="data/synthetic/telemetria_normal.csv"):
    df = pd.read_csv(caminho)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def escolher_equipamentos_com_falha(df, percentual_falha=0.10, seed=42):
    """
    Escolhe uma porcentagem dos equipamentos para falhar.
    """
    rng = np.random.default_rng(seed)
    equipamentos = df["equipamento_id"].drop_duplicates().tolist()

    qtd_falhas = max(1, int(len(equipamentos) * percentual_falha))
    escolhidos = rng.choice(equipamentos, size=qtd_falhas, replace=False)

    return list(escolhidos)


def definir_instantes_de_falha(df, equipamentos_com_falha, janela_degradacao_horas=24, seed=42):
    """
    Para cada equipamento escolhido, define um instante de falha.
    A falha não pode ser muito no começo, para dar tempo da degradação acontecer.
    """
    rng = np.random.default_rng(seed)
    falhas = {}

    for equipamento in equipamentos_com_falha:
        df_eq = df[df["equipamento_id"] == equipamento].sort_values("timestamp").reset_index(drop=True)

        # Precisamos de espaço antes da falha para degradar
        min_idx = janela_degradacao_horas * 4   # 4 leituras por hora (15 min)
        max_idx = len(df_eq) - 1

        if min_idx >= max_idx:
            continue

        idx_falha = rng.integers(min_idx, max_idx)
        timestamp_falha = df_eq.loc[idx_falha, "timestamp"]

        falhas[equipamento] = {
            "timestamp_falha": timestamp_falha,
            "idx_falha": int(idx_falha)
        }

    return falhas


def aplicar_degradacao_e_falha(df, falhas, janela_degradacao_horas=24, seed=42):
    """
    Injeta degradação progressiva e marca falha.
    """
    rng = np.random.default_rng(seed)

    df = df.copy()
    df["em_degradacao"] = 0
    df["falha_ativa"] = 0
    df["falha_24h"] = 0

    eventos = []

    for equipamento, info in falhas.items():
        timestamp_falha = info["timestamp_falha"]

        mask_eq = df["equipamento_id"] == equipamento
        df_eq = df.loc[mask_eq].sort_values("timestamp").copy()

        inicio_degradacao = timestamp_falha - pd.Timedelta(hours=janela_degradacao_horas)

        mask_deg = (df_eq["timestamp"] >= inicio_degradacao) & (df_eq["timestamp"] < timestamp_falha)
        mask_falha = df_eq["timestamp"] == timestamp_falha
        mask_falha_24h = (df_eq["timestamp"] >= inicio_degradacao) & (df_eq["timestamp"] < timestamp_falha)

        idxs_deg = df_eq[mask_deg].index.tolist()
        idxs_falha = df_eq[mask_falha].index.tolist()
        idxs_falha_24h = df_eq[mask_falha_24h].index.tolist()

        # Marcar target
        df.loc[idxs_falha_24h, "falha_24h"] = 1

        # Aplicar degradação progressiva
        n = len(idxs_deg)
        for pos, idx in enumerate(idxs_deg):
            progresso = (pos + 1) / n if n > 0 else 0

            # Quanto mais perto da falha, maior a deterioração
            df.loc[idx, "temperatura_interna"] += round(0.5 + 6.0 * progresso + rng.normal(0, 0.2), 1)
            df.loc[idx, "corrente"] += round(0.3 + 4.0 * progresso + rng.normal(0, 0.2), 1)
            df.loc[idx, "consumo_kw"] += round(0.2 + 1.8 * progresso + rng.normal(0, 0.1), 1)
            df.loc[idx, "pressao_descarga"] += round(5.0 + 25.0 * progresso + rng.normal(0, 2.0), 1)
            df.loc[idx, "tempo_ciclo_min"] = int(df.loc[idx, "tempo_ciclo_min"] + round(2 + 12 * progresso))
            df.loc[idx, "partidas_ult_1h"] = int(max(1, df.loc[idx, "partidas_ult_1h"] + round(1 + 4 * progresso)))

            # Em degradação o alarme começa a aparecer mais
            if rng.random() < (0.10 + 0.50 * progresso):
                df.loc[idx, "alarme_ativo"] = 1

            df.loc[idx, "em_degradacao"] = 1

        # Aplicar ponto de falha
        for idx in idxs_falha:
            df.loc[idx, "temperatura_interna"] += round(8.0 + rng.normal(0, 0.5), 1)
            df.loc[idx, "corrente"] += round(5.0 + rng.normal(0, 0.4), 1)
            df.loc[idx, "consumo_kw"] += round(2.5 + rng.normal(0, 0.2), 1)
            df.loc[idx, "pressao_descarga"] += round(35.0 + rng.normal(0, 3.0), 1)
            df.loc[idx, "tempo_ciclo_min"] = int(df.loc[idx, "tempo_ciclo_min"] + 15)
            df.loc[idx, "partidas_ult_1h"] = int(df.loc[idx, "partidas_ult_1h"] + 5)
            df.loc[idx, "alarme_ativo"] = 1
            df.loc[idx, "falha_ativa"] = 1

        # Registrar evento
        loja_id = df_eq.iloc[0]["loja_id"]

        eventos.append({
            "evento_id": f"EVT_{equipamento}_{timestamp_falha.strftime('%Y%m%d%H%M')}",
            "loja_id": loja_id,
            "equipamento_id": equipamento,
            "timestamp_evento": timestamp_falha,
            "tipo_evento": "falha_compressor",
            "falha_confirmada": 1,
            "severidade": "alta",
            "descricao": "Falha simulada precedida por degradação progressiva"
        })

    # Garantir que não fique valor impossível
    df["tempo_ciclo_min"] = df["tempo_ciclo_min"].clip(lower=1)
    df["partidas_ult_1h"] = df["partidas_ult_1h"].clip(lower=1)

    # Arredondar algumas colunas
    colunas_float = [
        "temperatura_interna", "temperatura_ambiente", "corrente",
        "consumo_kw", "pressao_succao", "pressao_descarga"
    ]
    for col in colunas_float:
        df[col] = df[col].round(1)

    eventos_df = pd.DataFrame(eventos)
    if not eventos_df.empty:
        eventos_df["timestamp_evento"] = pd.to_datetime(eventos_df["timestamp_evento"])

    return df, eventos_df


def main():
    # 1. Carregar telemetria normal
    df = carregar_telemetria("data/synthetic/telemetria_normal.csv")

    # 2. Escolher equipamentos que vão falhar
    equipamentos_com_falha = escolher_equipamentos_com_falha(
        df,
        percentual_falha=0.10,  # 10% dos equipamentos
        seed=42
    )

    print("Equipamentos escolhidos para falha:")
    for eq in equipamentos_com_falha:
        print("-", eq)

    # 3. Definir instante de falha para cada um
    falhas = definir_instantes_de_falha(
        df,
        equipamentos_com_falha,
        janela_degradacao_horas=24,
        seed=42
    )

    # 4. Aplicar degradação e falha
    df_falhas, eventos_df = aplicar_degradacao_e_falha(
        df,
        falhas,
        janela_degradacao_horas=24,
        seed=42
    )

    # 5. Salvar resultados
    pasta_saida = Path("data/synthetic")
    pasta_saida.mkdir(parents=True, exist_ok=True)

    telemetria_saida = pasta_saida / "telemetria_com_falhas.csv"
    eventos_saida = pasta_saida / "eventos_falha.csv"

    df_falhas.to_csv(telemetria_saida, index=False, encoding="utf-8")
    eventos_df.to_csv(eventos_saida, index=False, encoding="utf-8")

    # 6. Mostrar resumo
    print("\nArquivos gerados:")
    print("-", telemetria_saida)
    print("-", eventos_saida)

    print("\nResumo:")
    print("Total de linhas telemetria:", len(df_falhas))
    print("Total de eventos de falha:", len(eventos_df))
    print("Total de linhas com falha_24h = 1:", int(df_falhas["falha_24h"].sum()))
    print("Total de linhas com em_degradacao = 1:", int(df_falhas["em_degradacao"].sum()))
    print("Total de linhas com falha_ativa = 1:", int(df_falhas["falha_ativa"].sum()))

    if not eventos_df.empty:
        print("\nPrimeiros eventos:")
        print(eventos_df.head())


if __name__ == "__main__":
    main()