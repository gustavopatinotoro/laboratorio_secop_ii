from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import duckdb
import psutil
import pyarrow.parquet as pq
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# =============================================================================
# CONFIGURACIÓN Y REGLAS DE CLASIFICACIÓN ARQUITECTÓNICA (FASE 1)
# =============================================================================
DATA_DIR = Path("datos")
ARTIFACTS_DIR = Path("artefactos")
TEMP_SPILL_DIR = ARTIFACTS_DIR / "duckdb_tmp"

PROFILE_CSV_PATH = ARTIFACTS_DIR / "01_column_profile_summary.csv"
DEDUP_JSON_PATH = ARTIFACTS_DIR / "01_corpus_dedup_stats.json"

COLUMN_BATCH_SIZE = 16  # Procesa de a 16 columnas por consulta para acotar RAM < 450 MB

# Falsos valores usados como comodines burocráticos en SECOP II
BUREAUCRATIC_NULLS = (
    "",
    "NO DEFINIDO",
    "NO DEFINIDA",
    "SIN DEFINIR",
    "SIN DESCRIPCION",
    "SIN DESCRIPCIÓN",
    "NO APLICA",
    "N/A",
    "NA",
    "NINGUNO",
    "NINGUNA",
    "NULL",
    "NONE",
)

SEMANTIC_TEXT_COLS = {
    "objeto_del_contrato",
    "descripcion_del_proceso",
}

CORE_IDENTIFIERS_AND_NUMERICS = {
    "id_contrato",
    "proceso_de_compra",
    "referencia_del_contrato",
    "documento_proveedor",
    "proveedor_adjudicado",
    "codigo_proveedor",
    "valor_del_contrato",
    "valor_pagado",
    "valor_facturado",
    "valor_pendiente_de_pago",
    "valor_amortizado",
}

HIGH_PRIORITY_FACET_COLS = {
    "nombre_entidad",
    "nit_entidad",
    "codigo_entidad",
    "departamento",
    "ciudad",
    "orden",
    "sector",
    "rama",
    "entidad_centralizada",
    "estado_contrato",
    "codigo_de_categoria_principal",
    "tipo_de_contrato",
    "modalidad_de_contratacion",
    "justificacion_modalidad_de",
    "condiciones_de_entrega",
    "es_pyme",
    "espostconflicto",
    "fecha_de_firma",
    "fecha_de_inicio_del_contrato",
    "fecha_de_fin_del_contrato",
}


def get_ram_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def recommend_column_action(
    col_name: str,
    effective_valid_pct: float,
    approx_cardinality: int,
    avg_char_len: float,
) -> str:
    """Clasifica cada una de las 95 columnas según su rol en el motor de 8 GB de RAM."""
    if col_name in SEMANTIC_TEXT_COLS:
        return "KEEP_SEMANTIC_TEXT"
    if col_name in CORE_IDENTIFIERS_AND_NUMERICS:
        return "NORMALIZE_OR_PACK_DISK"
    if col_name in HIGH_PRIORITY_FACET_COLS:
        return "INDEX_ROARING_BITMAP"
    if effective_valid_pct < 35.0 or approx_cardinality <= 1:
        return "DROP_LOW_VALIDITY_OR_CONSTANT"
    if "url" in col_name or avg_char_len > 120.0:
        return "DROP_HIGH_STORAGE_OVERHEAD"
    if approx_cardinality <= 15_000 and effective_valid_pct >= 65.0:
        return "OPTIONAL_FACET_CANDIDATE"
    return "DROP_NON_ESSENTIAL_METADATA"


def run_column_profiling() -> None:
    console = Console()
    t0 = time.perf_counter()

    console.print(
        "[bold cyan]=== Fase 1: Perfilado Estadístico y Auditoría de Redundancia 1:N (DuckDB Out-of-Core) ===[/bold cyan]"
    )

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        console.print(
            f"[bold red]Error:[/bold red] No se encontraron archivos Parquet en '{DATA_DIR}/'."
        )
        return

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_SPILL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Leer lista de columnas e información básica desde los footers Parquet
    first_meta = pq.read_metadata(parquet_files[0])
    col_names = [
        first_meta.schema.column(i).name for i in range(first_meta.num_columns)
    ]
    total_raw_rows = sum(pq.read_metadata(p).num_rows for p in parquet_files)

    console.print(
        f"• Archivos detectados: [bold]{len(parquet_files)}[/bold] | "
        f"Filas crudas totales: [bold green]{total_raw_rows:,}[/bold green] | "
        f"Columnas: [bold]{len(col_names)}[/bold]"
    )
    console.print(f"• RAM inicial del proceso: {get_ram_mb():.1f} MB\n")

    con = duckdb.connect()
    con.execute("SET memory_limit = '1.5GB';")
    con.execute("SET threads = 4;")
    con.execute("SET preserve_insertion_order = false;")
    con.execute(f"SET temp_directory = '{TEMP_SPILL_DIR}';")

    parquet_glob = str(DATA_DIR / "*.parquet")
    null_literals_sql = ", ".join(f"'{v}'" for v in BUREAUCRATIC_NULLS)

    profile_records: list[dict[str, Any]] = []
    num_batches = math.ceil(len(col_names) / COLUMN_BATCH_SIZE)

    # -------------------------------------------------------------------------
    # PASO 1: Perfilado por lotes de 16 columnas (Validez Efectiva + HLL + Longitud)
    # -------------------------------------------------------------------------
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} lotes"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Perfilando {len(col_names)} columnas en bloques...", total=num_batches
        )

        for b_idx in range(num_batches):
            batch_cols = col_names[
                b_idx * COLUMN_BATCH_SIZE : (b_idx + 1) * COLUMN_BATCH_SIZE
            ]
            select_exprs: list[str] = []

            for idx_c, col in enumerate(batch_cols):
                q_col = f'"{col}"'
                clean_expr = f"UPPER(TRIM({q_col}))"
                is_valid_cond = f"({q_col} IS NOT NULL AND {clean_expr} NOT IN ({null_literals_sql}))"
                select_exprs.extend(
                    [
                        f"COUNT({q_col}) AS nn_{idx_c}",
                        f"COUNT(*) FILTER (WHERE {is_valid_cond}) AS eff_{idx_c}",
                        f"APPROX_COUNT_DISTINCT({q_col}) FILTER (WHERE {is_valid_cond}) AS card_{idx_c}",
                        f"COALESCE(AVG(LENGTH(TRIM({q_col}))) FILTER (WHERE {is_valid_cond}), 0.0) AS len_{idx_c}",
                    ]
                )

            sql = (
                f"SELECT {', '.join(select_exprs)} FROM read_parquet('{parquet_glob}');"
            )
            row = con.execute(sql).fetchone()
            assert row is not None

            for idx_c, col in enumerate(batch_cols):
                nn_cnt = int(row[idx_c * 4 + 0])
                eff_cnt = int(row[idx_c * 4 + 1])
                approx_card = int(row[idx_c * 4 + 2])
                avg_len = float(row[idx_c * 4 + 3])

                non_null_pct = (nn_cnt / total_raw_rows) * 100.0
                eff_valid_pct = (eff_cnt / total_raw_rows) * 100.0
                bureaucratic_null_pct = max(0.0, non_null_pct - eff_valid_pct)

                action = recommend_column_action(
                    col, eff_valid_pct, approx_card, avg_len
                )

                profile_records.append(
                    {
                        "column_index": b_idx * COLUMN_BATCH_SIZE + idx_c,
                        "column_name": col,
                        "non_null_count": nn_cnt,
                        "non_null_pct": round(non_null_pct, 2),
                        "effective_valid_count": eff_cnt,
                        "effective_valid_pct": round(eff_valid_pct, 2),
                        "bureaucratic_placeholder_pct": round(bureaucratic_null_pct, 2),
                        "approx_cardinality": approx_card,
                        "avg_char_len": round(avg_len, 1),
                        "recommended_action": action,
                    }
                )

            progress.update(task, advance=1)

    # Guardar CSV de perfilado usando DuckDB o escritura directa
    import csv

    with PROFILE_CSV_PATH.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=list(profile_records[0].keys()))
        writer.writeheader()
        writer.writerows(profile_records)

    # -------------------------------------------------------------------------
    # PASO 2: Auditoría Exacta de Deduplicación 1:N (Contratos vs. Nodos Semánticos)
    # -------------------------------------------------------------------------
    console.print(
        "\n[bold yellow]Ejecutando auditoría exacta de deduplicación 1:N (Hashes de 64 bits sobre objeto_del_contrato)...[/bold yellow]"
    )
    dedup_row = con.execute(f"""
        SELECT
            COUNT(*) AS total_raw_rows,
            COUNT(DISTINCT TRIM(COALESCE("id_contrato", ''))) AS unique_id_contrato,
            APPROX_COUNT_DISTINCT(LOWER(TRIM(COALESCE("objeto_del_contrato", '')))) AS hll_unique_objeto,
            COUNT(DISTINCT hash(LOWER(TRIM(COALESCE("objeto_del_contrato", ''))))) AS exact_unique_semantic_nodes
        FROM read_parquet('{parquet_glob}');
    """).fetchone()
    assert dedup_row is not None

    n_raw = int(dedup_row[0])
    n_unique_contracts = int(dedup_row[1])
    n_hll_nodes = int(dedup_row[2])
    n_exact_nodes = int(dedup_row[3])

    duplicate_id_rows = n_raw - n_unique_contracts
    redundant_template_contracts = n_unique_contracts - n_exact_nodes
    exact_template_dup_pct = (
        redundant_template_contracts / max(n_unique_contracts, 1)
    ) * 100.0
    hll_template_dup_pct = (
        (n_unique_contracts - n_hll_nodes) / max(n_unique_contracts, 1)
    ) * 100.0

    dedup_stats = {
        "total_parquet_files": len(parquet_files),
        "total_raw_rows": n_raw,
        "unique_contracts_by_id": n_unique_contracts,
        "duplicate_contract_id_rows": duplicate_id_rows,
        "hll_estimated_unique_nodes": n_hll_nodes,
        "exact_64bit_unique_semantic_nodes": n_exact_nodes,
        "redundant_template_contracts_absorbed": redundant_template_contracts,
        "hll_template_duplication_pct": round(hll_template_dup_pct, 2),
        "exact_template_duplication_pct": round(exact_template_dup_pct, 2),
        "execution_time_sec": round(time.perf_counter() - t0, 2),
        "peak_ram_mb": round(get_ram_mb(), 1),
    }
    DEDUP_JSON_PATH.write_text(
        json.dumps(dedup_stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    con.close()

    # -------------------------------------------------------------------------
    # PASO 3: Impresión de Tablas de Resultados para la Tesis
    # -------------------------------------------------------------------------
    action_counts: dict[str, int] = {}
    for r in profile_records:
        act = r["recommended_action"]
        action_counts[act] = action_counts.get(act, 0) + 1

    t_summary = Table(
        title="Resumen de Clasificación Arquitectónica de las 95 Columnas"
    )
    t_summary.add_column("Acción Recomendada en el Motor Híbrido", style="cyan")
    t_summary.add_column("N.º de Columnas", justify="right", style="bold green")
    for act, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
        t_summary.add_row(act, str(cnt))
    console.print("\n", t_summary)

    t_core = Table(
        title="Perfil de Columnas Nucleares Seleccionadas (Texto, Facetas y Cuantías)"
    )
    t_core.add_column("Columna", style="cyan")
    t_core.add_column("Validez Efectiva", justify="right")
    t_core.add_column("Nulos Burocráticos", justify="right", style="yellow")
    t_core.add_column("Cardinalidad Aprox.", justify="right", style="bold green")
    t_core.add_column("Long. Media", justify="right")
    t_core.add_column("Rol Arquitectónico")

    for r in profile_records:
        if r["recommended_action"] in (
            "KEEP_SEMANTIC_TEXT",
            "INDEX_ROARING_BITMAP",
            "NORMALIZE_OR_PACK_DISK",
        ):
            t_core.add_row(
                r["column_name"],
                f"{r['effective_valid_pct']:.2f}%",
                f"{r['bureaucratic_placeholder_pct']:.2f}%",
                f"{r['approx_cardinality']:,}",
                f"{r['avg_char_len']:.1f} chars",
                r["recommended_action"],
            )
    console.print("\n", t_core)

    t_dedup = Table(
        title="Auditoría Forense de Redundancia Semántica 1:N (Grafo Cociente)"
    )
    t_dedup.add_column("Métrica del Corpus", style="cyan")
    t_dedup.add_column("Valor Exacto Auditado", justify="right", style="bold green")
    t_dedup.add_row("Filas crudas totales en los 189 Parquets (N_raw)", f"{n_raw:,}")
    t_dedup.add_row(
        "Contratos únicos por id_contrato (N_contracts)", f"{n_unique_contracts:,}"
    )
    t_dedup.add_row(
        "Filas con id_contrato duplicado en fuente", f"{duplicate_id_rows:,}"
    )
    t_dedup.add_row(
        "Nodos semánticos únicos (Estimación HyperLogLog)",
        f"{n_hll_nodes:,} ({hll_template_dup_pct:.2f}% dup)",
    )
    t_dedup.add_row(
        "Nodos semánticos únicos (Hash Exacto 64-bit UBIGINT)",
        f"{n_exact_nodes:,} ({exact_template_dup_pct:.2f}% dup)",
    )
    t_dedup.add_row(
        "Contratos clonados absorbidos por el Puente CSR 1:N",
        f"{redundant_template_contracts:,}",
    )
    t_dedup.add_row(
        "Tiempo total de ejecución / RAM final",
        f"{dedup_stats['execution_time_sec']:.1f} s | {dedup_stats['peak_ram_mb']:.1f} MB",
    )
    console.print("\n", t_dedup)
    console.print(
        f"\n[bold green]✔ Reportes guardados en '{PROFILE_CSV_PATH}' y '{DEDUP_JSON_PATH}'.[/bold green]"
    )


if __name__ == "__main__":
    run_column_profiling()
