from __future__ import annotations

import json
import time
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

# 1. Asegurar que Python encuentre la carpeta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src/")))
# Importar el motor híbrido que ya tenemos depurado
from ejecucion_controlada.secop_engine import SecopHybridEngine, SearchFilter


ARTIFACTS_DIR = Path("artefactos")


def run_scientific_benchmark() -> None:
    console = Console()
    console.print(
        "[bold cyan]=== Benchmark Científico de Cierre: UNSPSC & Ablaciones Topológicas ===[/bold cyan]"
    )

    engine = SecopHybridEngine()

    # 1. Cargar una muestra de validación de textos reales sin inyección UNSPSC previa
    pq_path = ARTIFACTS_DIR / "secop_contracts_clean.parquet"
    import pyarrow.parquet as pq

    sample_table = pq.read_table(
        pq_path,
        columns=[
            "objeto_del_contrato",
            "codigo_de_categoria_principal",
            "departamento",
            "modalidad_de_contratacion",
        ],
    )

    # Seleccionar 100 consultas de prueba basadas en objetos contractuales reales
    df_sample = sample_table.to_pandas().dropna().sample(n=100, random_state=2026)

    unsp_seg_hits = 0
    unsp_fam_hits = 0
    ndcg_list = []
    latencies = []

    console.print(
        "[yellow]Evaluando 100 consultas de texto libre contra etiquetas taxonómicas UNSPSC...[/yellow]"
    )

    for idx, row in df_sample.iterrows():
        raw_obj = row["objeto_del_contrato"]
        true_unspsc = str(row["codigo_de_categoria_principal"])

        # Simular una consulta extrayendo las primeras 6 palabras o un fragmento representativo
        words = raw_obj.split()
        query_text = " ".join(words[: min(len(words), 8)])
        if len(query_text.strip()) < 5:
            continue

        t0 = time.perf_counter()
        res = engine.search(query_text=query_text, top_k=10, profile="forense")
        latencies.append((time.perf_counter() - t0) * 1000.0)

        results = res["results"]
        if not results:
            continue

        # Evaluar coincidencia taxonómica UNSPSC (Segmento = 2 dígitos, Familia = 4 dígitos)
        true_seg = true_unspsc[:2]
        true_fam = true_unspsc[:4] if len(true_unspsc) >= 4 else true_seg

        top1_row_id = results[0]["row_id"]
        top1_unspsc = str(
            sample_table["codigo_de_categoria_principal"][top1_row_id].as_py()
        )

        if top1_unspsc[:2] == true_seg:
            unsp_seg_hits += 1
        if len(top1_unspsc) >= 4 and top1_unspsc[:4] == true_fam:
            unsp_fam_hits += 1

        # Cálculo simplificado de DCG@10 por relevancia UNSPSC binaria (1 si coincide familia, 0 si no)
        dcg = sum(
            1.0 / np.log2(i + 2)
            for i, r in enumerate(results)
            if str(sample_table["codigo_de_categoria_principal"][r["row_id"]].as_py())[
                :4
            ]
            == true_fam
        )
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(10, len(results))))
        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)

    n_eval = len(ndcg_list)
    avg_seg_acc = (unsp_seg_hits / n_eval) * 100.0
    avg_fam_acc = (unsp_fam_hits / n_eval) * 100.0
    mean_ndcg = np.mean(ndicg_list := ndcg_list) * 100.0
    mean_lat = np.mean(latencies)

    table = Table(
        title="Resultados del Benchmark de Relevancia Taxonómica UNSPSC (N = 100 Consultas Reales)"
    )
    table.add_column("Métrica Evaluada", style="cyan")
    table.add_column("Valor Obtenido", justify="right", style="bold green")
    table.add_column("Criterio / Meta Propuesta", style="yellow")

    table.add_row(
        "Precisión Segmento UNSPSC (2 Dígitos)",
        f"{avg_seg_acc:.2f}%",
        "≥ 85.0% (Validación Semántica)",
    )
    table.add_row(
        "Precisión Familia UNSPSC (4 Dígitos)",
        f"{avg_fam_acc:.2f}%",
        "≥ 75.0% (Especificidad Técnica)",
    )
    table.add_row(
        "nDCG@10 Promedio (Relevancia Taxonómica)",
        f"{mean_ndcg:.2f}%",
        "≥ 80.0% (Calidad de Ranking)",
    )
    table.add_row(
        "Latencia Promedio End-to-End (Warm)",
        f"{mean_lat:.2f} ms",
        "≤ 120.0 ms (Restricción 8 GB)",
    )

    console.print(table)

    # Guardar reporte JSON para el anexo de la tesis
    report = {
        "n_queries": n_eval,
        "segment_precision_pct": round(avg_seg_acc, 2),
        "family_precision_pct": round(avg_fam_acc, 2),
        "ndcg_10_pct": round(mean_ndcg, 2),
        "mean_latency_ms": round(mean_lat, 2),
    }
    (ARTIFACTS_DIR / "unspsc_benchmark_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    console.print(
        "[bold green]✔ Reporte de benchmark UNSPSC guardado en artefactos/[/bold green]"
    )


if __name__ == "__main__":
    run_scientific_benchmark()
