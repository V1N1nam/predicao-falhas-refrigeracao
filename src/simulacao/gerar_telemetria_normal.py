from pathlib import Path
import numpy as np
import pandas as pd


def criar_equipamentos(qtd_lojas=10, compressores_por_loja=5):
    """
    Cria uma tabela de equipamentos.
    Cada loja terá vários compressores.
    """
    registros = []

    for loja in range(1, qtd_lojas + 1):
        loja_id = f"L{loja:02d}"

        for comp in range(1, compressores_por_loja + 1):
            equipamento_id = f"COMP_{loja_id}_{comp:02d}"

            registros.append({
                "loja_id": loja_id,
                "equipamento_id": equipamento_id,
                "tipo_equipamento": "compressor",
                "modelo": "Copeland_ZB",
                "criticidade": "alta",
                "setpoint_temperatura": -18.0
            })

    return pd.DataFrame(registros)


def criar_timestamps(data_inicio="2026-01-01 00:00:00", dias=30, freq="15min"):
    """
    Cria a sequência de tempo.
    Exemplo: de 15 em 15 minutos por 30 dias.
    """
    inicio = pd.Timestamp(data_inicio)
    fim = inicio + pd.Timedelta(days=dias) - pd.Timedelta(minutes=15)

    timestamps = pd.date_range(start=inicio, end=fim, freq=freq)
    return timestamps


def gerar_telemetria_normal(equipamentos_df, timestamps, seed=42):
    """
    Gera dados sintéticos normais para cada equipamento em cada timestamp.
    """
    rng = np.random.default_rng(seed)
    registros = []

    for _, eq in equipamentos_df.iterrows():
        loja_id = eq["loja_id"]
        equipamento_id = eq["equipamento_id"]
        setpoint = eq["setpoint_temperatura"]

        # Cada equipamento ganha um "perfil base" próprio.
        # Isso evita que todos tenham exatamente o mesmo comportamento.
        temp_base = setpoint + rng.uniform(-1.0, 1.0)
        ambiente_base = rng.uniform(22.0, 32.0)
        corrente_base = rng.uniform(8.5, 11.5)
        consumo_base = rng.uniform(2.8, 3.8)
        succao_base = rng.uniform(22.0, 28.0)
        descarga_base = rng.uniform(190.0, 210.0)
        ciclo_base = rng.integers(10, 15)
        partidas_base = rng.integers(1, 3)

        for ts in timestamps:
            # Oscilações pequenas em torno do comportamento normal
            temperatura_interna = temp_base + rng.normal(0, 0.6)
            temperatura_ambiente = ambiente_base + rng.normal(0, 1.5)
            corrente = corrente_base + rng.normal(0, 0.4)
            consumo_kw = consumo_base + rng.normal(0, 0.2)
            pressao_succao = succao_base + rng.normal(0, 1.0)
            pressao_descarga = descarga_base + rng.normal(0, 4.0)
            tempo_ciclo_min = ciclo_base + rng.integers(-1, 2)
            partidas_ult_1h = max(1, partidas_base + rng.integers(-1, 2))

            # Alarme raro em operação normal
            alarme_ativo = 1 if rng.random() < 0.03 else 0

            registros.append({
                "timestamp": ts,
                "loja_id": loja_id,
                "equipamento_id": equipamento_id,
                "temperatura_interna": round(temperatura_interna, 1),
                "temperatura_ambiente": round(temperatura_ambiente, 1),
                "corrente": round(corrente, 1),
                "consumo_kw": round(consumo_kw, 1),
                "pressao_succao": round(pressao_succao, 1),
                "pressao_descarga": round(pressao_descarga, 1),
                "tempo_ciclo_min": int(tempo_ciclo_min),
                "partidas_ult_1h": int(partidas_ult_1h),
                "alarme_ativo": int(alarme_ativo)
            })

    return pd.DataFrame(registros)


def main():
    # 1. Criar equipamentos
    equipamentos_df = criar_equipamentos(qtd_lojas=10, compressores_por_loja=5)

    # 2. Criar timestamps
    timestamps = criar_timestamps(
        data_inicio="2026-01-01 00:00:00",
        dias=30,
        freq="15min"
    )

    # 3. Gerar telemetria normal
    telemetria_df = gerar_telemetria_normal(equipamentos_df, timestamps, seed=42)

    # 4. Criar pastas de saída
    pasta_saida = Path("data/synthetic")
    pasta_saida.mkdir(parents=True, exist_ok=True)

    # 5. Salvar arquivos
    equipamentos_path = pasta_saida / "equipamentos.csv"
    telemetria_path = pasta_saida / "telemetria_normal.csv"

    equipamentos_df.to_csv(equipamentos_path, index=False, encoding="utf-8")
    telemetria_df.to_csv(telemetria_path, index=False, encoding="utf-8")

    # 6. Mostrar resumo
    print("Arquivos gerados com sucesso:")
    print(f"- {equipamentos_path}")
    print(f"- {telemetria_path}")

    print("\nResumo:")
    print(f"- Total de equipamentos: {len(equipamentos_df)}")
    print(f"- Total de timestamps por equipamento: {len(timestamps)}")
    print(f"- Total de linhas de telemetria: {len(telemetria_df)}")

    print("\nPrimeiras linhas da telemetria:")
    print(telemetria_df.head())


if __name__ == "__main__":
    main()