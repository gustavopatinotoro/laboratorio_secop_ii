from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import psutil
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table

DATA_DIR = Path("datos")
ARTIFACTS_DIR = Path("artefactos")
TEMP_SPILL_DIR = ARTIFACTS_DIR / "duckdb_tmp"

CLEAN_CONTRACTS_PATH = ARTIFACTS_DIR / "secop_contracts_clean.parquet"
UNIQUE_TEXTS_PATH = ARTIFACTS_DIR / "secop_unique_texts.parquet"
ETL_REPORT_PATH = ARTIFACTS_DIR / "02_etl_compaction_report.json"


def get_ram_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def clean_num_sql(col_name: str) -> str:
    return f"COALESCE(TRY_CAST(REGEXP_REPLACE(TRIM(\"{col_name}\"), '[^0-9.-]', '', 'g') AS DOUBLE), 0.0)"


def clean_date_sql(col_name: str) -> str:
    """
    Soporta todos los formatos de fecha de Socrata Colombia:
    YYYY-MM-DD, YYYY/MM/DD, DD/MM/YYYY, MM/DD/YYYY y timestamps ISO.
    """
    s = f'SUBSTR(TRIM("{col_name}"), 1, 10)'
    return f"""COALESCE(
        TRY_CAST({s} AS DATE),
        TRY_STRPTIME({s}, '%Y/%m/%d')::DATE,
        TRY_STRPTIME({s}, '%d/%m/%Y')::DATE,
        TRY_STRPTIME({s}, '%m/%d/%Y')::DATE,
        TRY_STRPTIME({s}, '%d-%m-%Y')::DATE,
        TRY_STRPTIME({s}, '%m-%d-%Y')::DATE
    )"""


def clean_str_sql(col_name: str, default_val: str = "NO DEFINIDO") -> str:
    return f"COALESCE(NULLIF(TRIM(\"{col_name}\"), ''), '{default_val}')"


def run_etl_compaction() -> None:
    console = Console()
    t0 = time.perf_counter()

    console.print(
        "[bold cyan]=== Fase 2: Compactación Relacional con Parseo Multi-Formato de Fechas ===[/bold cyan]"
    )

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        console.print(
            f"[bold red]Error:[/bold red] No se encontraron archivos Parquet en '{DATA_DIR}/'."
        )
        return

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_SPILL_DIR.mkdir(parents=True, exist_ok=True)

    parquet_glob = str(DATA_DIR / "*.parquet")
    con = duckdb.connect()
    con.execute("SET memory_limit = '2.2GB';")
    con.execute("SET threads = 4;")
    con.execute("SET preserve_insertion_order = false;")
    con.execute(f"SET temp_directory = '{TEMP_SPILL_DIR}';")

    # Verificar muestra de fechas antes de compactar
    sample_dates = con.execute(f"""
        SELECT "fecha_de_firma", {clean_date_sql("fecha_de_firma")} AS parsed_date
        FROM read_parquet('{parquet_glob}')
        WHERE "fecha_de_firma" IS NOT NULL AND TRIM("fecha_de_firma") != ''
        LIMIT 3;
    """).fetchall()
    console.print(
        f"• Muestra de verificación de fechas (Cruda -> Parseada): {sample_dates}"
    )

    t1 = time.perf_counter()
    console.print(
        "[bold yellow]Paso 1/2:[/bold yellow] Aislando llaves id_contrato con duplicados en fuente..."
    )
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE dup_contract_ids AS
        SELECT TRIM(COALESCE("id_contrato", '')) AS id_clean
        FROM read_parquet('{parquet_glob}')
        GROUP BY id_clean
        HAVING COUNT(*) > 1;
    """)
    n_dup_keys = int(
        con.execute("SELECT COUNT(*) FROM dup_contract_ids;").fetchone()[0]
    )
    console.print(
        f"   ✔ {n_dup_keys:,} llaves duplicadas aisladas en {time.perf_counter() - t1:.1f}s"
    )

    canonical_select_sql = f"""
        TRIM(COALESCE("id_contrato", 'SIN_ID')) AS id_contrato,
        {clean_str_sql("proceso_de_compra")} AS proceso_de_compra,
        {clean_str_sql("referencia_del_contrato")} AS referencia_del_contrato,
        {clean_str_sql("nit_entidad")} AS nit_entidad,
        {clean_str_sql("nombre_entidad")} AS nombre_entidad,
        {clean_str_sql("codigo_entidad")} AS codigo_entidad,
        {clean_str_sql("documento_proveedor")} AS documento_proveedor,
        {clean_str_sql("proveedor_adjudicado")} AS proveedor_adjudicado,
        {clean_str_sql("codigo_proveedor")} AS codigo_proveedor,
        UPPER({clean_str_sql("departamento")}) AS departamento,
        UPPER({clean_str_sql("ciudad")}) AS ciudad,
        UPPER({clean_str_sql("orden")}) AS orden,
        UPPER({clean_str_sql("sector")}) AS sector,
        UPPER({clean_str_sql("rama")}) AS rama,
        UPPER({clean_str_sql("entidad_centralizada")}) AS entidad_centralizada,
        UPPER({clean_str_sql("estado_contrato")}) AS estado_contrato,
        UPPER({clean_str_sql("codigo_de_categoria_principal", "00000000")}) AS codigo_de_categoria_principal,
        UPPER({clean_str_sql("tipo_de_contrato")}) AS tipo_de_contrato,
        UPPER({clean_str_sql("modalidad_de_contratacion")}) AS modalidad_de_contratacion,
        UPPER({clean_str_sql("condiciones_de_entrega")}) AS condiciones_de_entrega,
        UPPER({clean_str_sql("es_pyme", "NO")}) AS es_pyme,
        UPPER({clean_str_sql("espostconflicto", "NO")}) AS espostconflicto,
        {clean_date_sql("fecha_de_firma")} AS fecha_de_firma,
        {clean_date_sql("fecha_de_inicio_del_contrato")} AS fecha_de_inicio_del_contrato,
        {clean_date_sql("fecha_de_fin_del_contrato")} AS fecha_de_fin_del_contrato,
        {clean_num_sql("valor_del_contrato")} AS valor_del_contrato,
        {clean_num_sql("valor_pagado")} AS valor_pagado,
        {clean_num_sql("valor_facturado")} AS valor_facturado,
        {clean_str_sql("objeto_del_contrato", "SIN OBJETO")} AS objeto_del_contrato,
        {clean_str_sql("descripcion_del_proceso", "SIN DESCRIPCION")} AS descripcion_del_proceso,
        CAST(hash(LOWER(TRIM(COALESCE("objeto_del_contrato", '')))) AS UBIGINT) AS text_node_hash
    """

    t2 = time.perf_counter()
    console.print(
        "[bold yellow]Paso 2/2:[/bold yellow] Regenerando 'secop_contracts_clean.parquet' con fechas verificadas..."
    )
    clean_out_str = str(CLEAN_CONTRACTS_PATH)

    con.execute(f"""
        COPY (
            SELECT {canonical_select_sql}
            FROM read_parquet('{parquet_glob}')
            WHERE TRIM(COALESCE("id_contrato", '')) NOT IN (SELECT id_clean FROM dup_contract_ids)
            UNION ALL
            SELECT DISTINCT ON (TRIM(COALESCE("id_contrato", '')))
                {canonical_select_sql}
            FROM read_parquet('{parquet_glob}')
            WHERE TRIM(COALESCE("id_contrato", '')) IN (SELECT id_clean FROM dup_contract_ids)
        ) TO '{clean_out_str}' (FORMAT PARQUET, COMPRESSION 'ZSTD', COMPRESSION_LEVEL 3, ROW_GROUP_SIZE 100000);
    """)

    valid_dates_cnt = int(
        con.execute(
            f"SELECT COUNT(fecha_de_firma) FROM read_parquet('{clean_out_str}');"
        ).fetchone()[0]
    )
    meta_clean = pq.read_metadata(CLEAN_CONTRACTS_PATH)
    clean_rows = int(meta_clean.num_rows)
    console.print(
        f"   ✔ 'secop_contracts_clean.parquet' listo: [bold green]{clean_rows:,}[/bold green] contratos | "
        f"Fechas de firma válidas: [bold green]{valid_dates_cnt:,} ({valid_dates_cnt / clean_rows * 100:.2f}%)[/bold green] "
        f"en {time.perf_counter() - t2:.1f}s"
    )
    con.close()


if __name__ == "__main__":
    run_etl_compaction()
