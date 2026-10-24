from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import psutil
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table

DATA_DIR = Path("datos")
ARTIFACTS_DIR = Path("artefactos")
OUTPUT_JSON = ARTIFACTS_DIR / "00_schema_audit_report.json"


def get_ram_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def run_footer_audit() -> None:
    console = Console()
    t0 = time.perf_counter()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    console.print(
        "[bold cyan]=== Fase 0: Auditoría Forense de Footers Parquet y Verificación de Esquema ===[/bold cyan]"
    )

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        console.print(
            f"[bold red]Error:[/bold red] No se encontraron archivos .parquet en '{DATA_DIR}/'."
        )
        return

    total_rows = 0
    total_bytes = 0
    total_row_groups = 0
    row_counts: list[int] = []

    reference_schema: list[tuple[str, str]] | None = None
    schema_mismatches: list[dict[str, Any]] = []
    type_counter: Counter[str] = Counter()

    for idx, fpath in enumerate(parquet_files):
        fsize = fpath.stat().st_size
        meta = pq.read_metadata(fpath)
        schema = pq.read_schema(fpath)

        num_rows = int(meta.num_rows)
        num_rgs = int(meta.num_row_groups)

        total_rows += num_rows
        total_bytes += fsize
        total_row_groups += num_rgs
        row_counts.append(num_rows)

        current_schema = [(field.name, str(field.type)) for field in schema]

        if reference_schema is None:
            reference_schema = current_schema
            for _, col_type in reference_schema:
                type_counter[col_type] += 1
        elif current_schema != reference_schema:
            # Detectar exactamente qué columnas cambiaron de nombre o formato físico
            ref_dict = dict(reference_schema)
            cur_dict = dict(current_schema)
            diff_cols = {
                k: {"esperado": ref_dict.get(k), "encontrado": cur_dict.get(k)}
                for k in set(ref_dict) | set(cur_dict)
                if ref_dict.get(k) != cur_dict.get(k)
            }
            schema_mismatches.append(
                {
                    "file": fpath.name,
                    "num_columns": len(current_schema),
                    "differences": diff_cols,
                }
            )

    assert reference_schema is not None
    schema_hash = hashlib.sha256(
        json.dumps(reference_schema, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    report = {
        "total_parquet_files": len(parquet_files),
        "total_raw_rows": total_rows,
        "total_row_groups": total_row_groups,
        "total_size_bytes": total_bytes,
        "total_size_gb": round(total_bytes / (1024**3), 4),
        "num_columns": len(reference_schema),
        "schema_consistent_across_all_files": len(schema_mismatches) == 0,
        "schema_mismatches_count": len(schema_mismatches),
        "schema_sha256_fingerprint": schema_hash,
        "physical_type_distribution": dict(type_counter),
        "rows_per_file_stats": {
            "min_rows": min(row_counts),
            "max_rows": max(row_counts),
            "files_with_30k_rows": sum(1 for r in row_counts if r == 30_000),
            "last_file_rows": row_counts[-1],
        },
        "columns": [
            {"index": i, "name": name, "arrow_physical_type": dtype}
            for i, (name, dtype) in enumerate(reference_schema)
        ],
        "schema_mismatches_detail": schema_mismatches[:10],
    }

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Tabla resumen de integridad física
    t_summary = Table(
        title="Resumen Físico del Corpus Crudo SECOP II (Lectura O(1) de Footers)"
    )
    t_summary.add_column("Métrica de Verificación", style="cyan")
    t_summary.add_column("Resultado Auditado", justify="right", style="bold green")

    t_summary.add_row("Archivos Parquet analizados", f"{len(parquet_files):,}")
    t_summary.add_row("Filas crudas totales (N_raw)", f"{total_rows:,}")
    t_summary.add_row("Row-Groups totales", f"{total_row_groups:,}")
    t_summary.add_row(
        "Tamaño físico en disco",
        f"{report['total_size_gb']:.2f} GB ({total_bytes / (1024**2):,.1f} MB)",
    )
    t_summary.add_row("Número de columnas por archivo", f"{len(reference_schema)}")
    t_summary.add_row(
        "Consistencia de esquema (0 Schema Drift)",
        "100% CONSISTENTE"
        if len(schema_mismatches) == 0
        else f"ALERTA ({len(schema_mismatches)} desajustes)",
    )
    t_summary.add_row(
        "Distribución de formatos físicos Arrow", json.dumps(dict(type_counter))
    )
    t_summary.add_row("Huella SHA-256 del esquema", f"{schema_hash[:24]}...")
    t_summary.add_row(
        "Tiempo de ejecución / Pico RAM",
        f"{time.perf_counter() - t0:.2f} s | {get_ram_mb():.1f} MB",
    )

    console.print("\n", t_summary)
    console.print(
        f"[bold green]✔ Reporte de esquema guardado en:[/bold green] {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    run_footer_audit()
