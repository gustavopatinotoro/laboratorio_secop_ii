from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# =============================================================================
# CONSTANTES CANÓNICAS DEL PROYECTO SECOP II (Socrata Dataset ID: jbjy-vk9h)
# =============================================================================
SOCRATA_DATASET_ID = "jbjy-vk9h"
SOCRATA_BULK_CSV_URL = f"https://www.datos.gov.co/api/views/{SOCRATA_DATASET_ID}/rows.csv?accessType=DOWNLOAD"
SOCRATA_SODA_CSV_URL = f"https://www.datos.gov.co/resource/{SOCRATA_DATASET_ID}.csv"

CHUNK_ROWS = 30_000  # 5,633,184 filas / 30,000 = 188 archivos Parquet exactos
EXPECTED_TOTAL_FILES = 188
DATA_DIR = Path("datos")
ARTIFACTS_DIR = Path("artifactos")
MANIFEST_JSON_PATH = ARTIFACTS_DIR / "00_sha256_manifest.json"
SHA256SUMS_TXT_PATH = DATA_DIR / "SHA256SUMS.txt"


def get_ram_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def compute_file_sha256(filepath: Path) -> str:
    """Calcula el hash criptográfico SHA-256 leyendo en bloques de 1 MB (0 impacto en RAM)."""
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def normalize_column_name(col_name: str) -> str:
    """Normaliza encabezados de Socrata a snake_case limpio."""
    clean = col_name.strip().lower()
    for src, dst in [
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
        ("ñ", "n"),
    ]:
        clean = clean.replace(src, dst)
    for ch in [" ", "-", "/", ".", "(", ")", ":"]:
        clean = clean.replace(ch, "_")
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_")


def write_parquet_part_and_hash(
    table: pa.Table,
    part_idx: int,
    out_dir: Path,
) -> dict[str, Any]:
    """
    Escribe un bloque de hasta 30,000 filas como archivo Parquet ZSTD de forma atómica
    y calcula inmediatamente su firma SHA-256.
    """
    part_name = f"secop_part_{part_idx:05d}.parquet"
    final_path = out_dir / part_name
    tmp_path = out_dir / f".tmp_{part_name}"

    pq.write_table(
        table,
        tmp_path,
        compression="ZSTD",
        compression_level=3,
        row_group_size=CHUNK_ROWS,
        use_dictionary=True,
    )
    tmp_path.replace(final_path)

    sha256_hex = compute_file_sha256(final_path)
    size_bytes = final_path.stat().st_size

    return {
        "part_index": part_idx,
        "filename": part_name,
        "num_rows": int(table.num_rows),
        "num_columns": int(table.num_columns),
        "size_bytes": int(size_bytes),
        "sha256": sha256_hex,
    }


def save_manifest(parts_metadata: list[dict[str, Any]], mode: str) -> None:
    """Guarda el manifiesto JSON en artifacts/ y el archivo estándar SHA256SUMS.txt en datos/."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    parts_sorted = sorted(parts_metadata, key=lambda x: x["part_index"])
    total_rows = sum(p["num_rows"] for p in parts_sorted)
    total_bytes = sum(p["size_bytes"] for p in parts_sorted)

    # Hash global del corpus (Merkle-like root sobre la concatenación de los SHA-256 individuales)
    corpus_hasher = hashlib.sha256()
    for p in parts_sorted:
        corpus_hasher.update(p["sha256"].encode("ascii"))

    manifest = {
        "dataset_id": SOCRATA_DATASET_ID,
        "ingestion_mode": mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chunk_row_target": CHUNK_ROWS,
        "total_files": len(parts_sorted),
        "total_rows": total_rows,
        "total_size_bytes": total_bytes,
        "total_size_gb": round(total_bytes / (1024**3), 4),
        "corpus_merkle_sha256": corpus_hasher.hexdigest(),
        "files": parts_sorted,
    }
    MANIFEST_JSON_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Escribir formato estándar GNU coreutils (compatible con `shasum -a 256 -c SHA256SUMS.txt`)
    lines = [f"{p['sha256']}  {p['filename']}" for p in parts_sorted]
    SHA256SUMS_TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# MODO 1: INGESTA PAGINADA DETERMINISTA VÍA SODA API (REANUDABLE POR PARTES)
# =============================================================================
def ingest_paginated_soda_api(
    app_token: str | None = None,
    max_parts: int | None = None,
) -> None:
    console = Console()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    console.print(
        "[bold cyan]=== Etapa 00a: Ingesta Paginada Determinista desde Socrata (SODA API) ===[/bold cyan]"
    )
    console.print(
        f"• Dataset ID: [bold]{SOCRATA_DATASET_ID}[/bold] | Lote fijo: [bold]{CHUNK_ROWS:,}[/bold] filas/Parquet"
    )

    existing_parts: dict[int, dict[str, Any]] = {}
    if MANIFEST_JSON_PATH.exists():
        old_man = json.loads(MANIFEST_JSON_PATH.read_text(encoding="utf-8"))
        for item in old_man.get("files", []):
            p_path = DATA_DIR / item["filename"]
            if p_path.exists() and compute_file_sha256(p_path) == item["sha256"]:
                existing_parts[int(item["part_index"])] = item

    if existing_parts:
        console.print(
            f"✔ Detectados [bold green]{len(existing_parts)}[/bold green] archivos Parquet íntegros previos (reanudación activa)."
        )

    part_idx = 0
    canonical_names: list[str] | None = None
    canonical_schema: pa.Schema | None = None

    while True:
        if max_parts is not None and part_idx >= max_parts:
            break

        if part_idx in existing_parts:
            part_idx += 1
            continue

        offset = part_idx * CHUNK_ROWS
        # Orden determinista por :id en Socrata evita filas duplicadas o saltadas entre lotes
        query_url = f"{SOCRATA_SODA_CSV_URL}?$order=:id%20ASC&$limit={CHUNK_ROWS}&$offset={offset}"

        headers = {"User-Agent": "SECOP-Hybrid-Thesis-Engine/1.0"}
        if app_token:
            headers["X-App-Token"] = app_token

        raw_csv_bytes: bytes | None = None
        for attempt in range(1, 6):
            try:
                req = urllib.request.Request(query_url, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    raw_csv_bytes = resp.read()
                break
            except Exception as exc:
                wait_s = 2**attempt
                console.print(
                    f"[yellow]Reintento {attempt}/5 en lote {part_idx} (offset={offset:,}): {exc}. Esperando {wait_s}s...[/yellow]"
                )
                time.sleep(wait_s)

        if raw_csv_bytes is None:
            raise RuntimeError(
                f"Fallo definitivo descargando el lote {part_idx} (offset={offset:,})."
            )

        # Leer CSV en memoria con PyArrow forzando strings para evitar Schema Drift entre los 188 archivos
        read_opts = pacsv.ReadOptions(use_threads=True, block_size=16 * 1024 * 1024)
        parse_opts = pacsv.ParseOptions(newlines_in_values=True)

        # Primera pasada rápida para obtener nombres de columna si aún no están fijados
        temp_table = pacsv.read_csv(
            io.BytesIO(raw_csv_bytes), read_options=read_opts, parse_options=parse_opts
        )
        if temp_table.num_rows == 0:
            console.print(f"✔ Fin del dataset alcanzado en el archivo #{part_idx}.")
            break

        if canonical_names is None:
            canonical_names = [
                normalize_column_name(c) for c in temp_table.column_names
            ]
            canonical_schema = pa.schema(
                [(name, pa.string()) for name in canonical_names]
            )

        # Re-leer forzando todas las columnas como string crudo uniforme para garantizar esquema idéntico en los 188 Parquets
        string_types = {col: pa.string() for col in temp_table.column_names}
        conv_opts = pacsv.ConvertOptions(
            column_types=string_types, strings_can_be_null=True
        )
        table = pacsv.read_csv(
            io.BytesIO(raw_csv_bytes),
            read_options=read_opts,
            parse_options=parse_opts,
            convert_options=conv_opts,
        )
        table = table.rename_columns(canonical_names).cast(canonical_schema)

        meta = write_parquet_part_and_hash(table, part_idx, DATA_DIR)
        existing_parts[part_idx] = meta
        save_manifest(list(existing_parts.values()), mode="paginated_soda_api")

        console.print(
            f"  • [{part_idx + 1:03d}] {meta['filename']} | "
            f"Filas: {meta['num_rows']:,} | "
            f"Tamaño: {meta['size_bytes'] / (1024**2):.2f} MB | "
            f"SHA-256: {meta['sha256'][:16]}... | "
            f"RAM: {get_ram_mb():.1f} MB"
        )

        n_rows_batch = table.num_rows
        del table, temp_table, raw_csv_bytes
        pa.default_memory_pool().release_unused()
        gc.collect()

        if n_rows_batch < CHUNK_ROWS:
            console.print(
                f"✔ Último lote parcial detectado ({n_rows_batch:,} filas). Ingesta completada."
            )
            break

        part_idx += 1


# =============================================================================
# MODO 2: STREAMING DIRECTO DESDE EL VOLCADO MASIVO DE SOCRATA (SIN CSV EN DISCO)
# =============================================================================
def ingest_bulk_stream() -> None:
    console = Console()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    console.print(
        "[bold cyan]=== Etapa 00a: Streaming Masivo Out-of-Core desde Socrata -> 188 Parquets ===[/bold cyan]"
    )
    req = urllib.request.Request(
        SOCRATA_BULK_CSV_URL,
        headers={"User-Agent": "SECOP-Hybrid-Thesis-Engine/1.0"},
    )

    parts_metadata: list[dict[str, Any]] = []
    part_idx = 0
    batch_buffer: list[pa.RecordBatch] = []
    buffered_rows = 0
    canonical_names: list[str] | None = None

    with urllib.request.urlopen(req, timeout=300) as response:
        read_opts = pacsv.ReadOptions(block_size=32 * 1024 * 1024)
        parse_opts = pacsv.ParseOptions(newlines_in_values=True)
        csv_reader = pacsv.open_csv(
            response, read_options=read_opts, parse_options=parse_opts
        )

        for batch in csv_reader:
            if canonical_names is None:
                canonical_names = [normalize_column_name(f.name) for f in batch.schema]

            # Convertir cada columna del RecordBatch a string uniforme para evitar choques de tipos entre lotes
            str_arrays = [
                pc_cast_to_str(batch.column(i)) for i in range(batch.num_columns)
            ]
            clean_batch = pa.RecordBatch.from_arrays(str_arrays, names=canonical_names)

            batch_buffer.append(clean_batch)
            buffered_rows += clean_batch.num_rows

            while buffered_rows >= CHUNK_ROWS:
                combined = pa.Table.from_batches(batch_buffer)
                slice_30k = combined.slice(0, CHUNK_ROWS)
                remainder = combined.slice(CHUNK_ROWS)

                meta = write_parquet_part_and_hash(slice_30k, part_idx, DATA_DIR)
                parts_metadata.append(meta)
                console.print(
                    f"  • [{part_idx + 1:03d}] {meta['filename']} | "
                    f"Filas: {meta['num_rows']:,} | "
                    f"{meta['size_bytes'] / (1024**2):.2f} MB | "
                    f"SHA-256: {meta['sha256'][:16]}... | RAM: {get_ram_mb():.1f} MB"
                )
                part_idx += 1

                if remainder.num_rows > 0:
                    batch_buffer = remainder.to_batches()
                    buffered_rows = remainder.num_rows
                else:
                    batch_buffer = []
                    buffered_rows = 0

                del combined, slice_30k, remainder
                pa.default_memory_pool().release_unused()

        # Escribir el lote final remanente (ej. part_00187.parquet con las últimas 23,184 filas)
        if buffered_rows > 0:
            final_table = pa.Table.from_batches(batch_buffer)
            meta = write_parquet_part_and_hash(final_table, part_idx, DATA_DIR)
            parts_metadata.append(meta)
            console.print(
                f"  • [{part_idx + 1:03d}] {meta['filename']} (Lote de cierre) | "
                f"Filas: {meta['num_rows']:,} | "
                f"{meta['size_bytes'] / (1024**2):.2f} MB | "
                f"SHA-256: {meta['sha256'][:16]}..."
            )

    save_manifest(parts_metadata, mode="bulk_http_stream")
    console.print(
        f"\n[bold green]✔ Ingesta finalizada: {len(parts_metadata)} archivos Parquet y manifiesto SHA-256 creados.[/bold green]"
    )


def pc_cast_to_str(arr: pa.Array) -> pa.Array:
    import pyarrow.compute as pc

    if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
        return arr.cast(pa.string())
    return pc.cast(arr, pa.string(), safe=False)


# =============================================================================
# MODO 3: VERIFICACIÓN CRIPTOGRÁFICA SHA-256 (POST-DESCARGA DESDE GOOGLE DRIVE)
# =============================================================================
def verify_local_corpus_integrity() -> None:
    console = Console()
    t0 = time.perf_counter()
    console.print(
        "[bold cyan]=== Auditoría Criptográfica SHA-256 de los 188 Archivos Parquet Locales ===[/bold cyan]"
    )

    if not MANIFEST_JSON_PATH.exists():
        # Si aún no existe el manifiesto pero ya tienes los 188 archivos en datos/, podemos sellarlos ahora mismo
        parquet_files = sorted(DATA_DIR.glob("*.parquet"))
        if not parquet_files:
            console.print(
                f"[bold red]Error:[/bold red] No existe {MANIFEST_JSON_PATH} ni archivos Parquet en {DATA_DIR}/."
            )
            return
        console.print(
            f"[yellow]No se encontró {MANIFEST_JSON_PATH}. Generando manifiesto SHA-256 inicial sobre los {len(parquet_files)} archivos existentes en '{DATA_DIR}/'...[/yellow]"
        )
        parts_meta = []
        for idx, p_path in enumerate(parquet_files):
            meta_pq = pq.read_metadata(p_path)
            parts_meta.append(
                {
                    "part_index": idx,
                    "filename": p_path.name,
                    "num_rows": int(meta_pq.num_rows),
                    "num_columns": int(meta_pq.num_columns),
                    "size_bytes": int(p_path.stat().st_size),
                    "sha256": compute_file_sha256(p_path),
                }
            )
        save_manifest(parts_meta, mode="sealed_from_existing_local_parquets")
        console.print(
            f"✔ Manifiesto sellado en {MANIFEST_JSON_PATH} y {SHA256SUMS_TXT_PATH}.\n"
        )

    manifest = json.loads(MANIFEST_JSON_PATH.read_text(encoding="utf-8"))
    expected_files = manifest["files"]

    ok_count = 0
    corrupted: list[str] = []
    missing: list[str] = []
    total_verified_rows = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} archivos"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Verificando firmas SHA-256 y footers Parquet...", total=len(expected_files)
        )
        for item in expected_files:
            fpath = DATA_DIR / item["filename"]
            if not fpath.exists():
                missing.append(item["filename"])
                progress.update(task, advance=1)
                continue

            actual_sha = compute_file_sha256(fpath)
            meta_pq = pq.read_metadata(fpath)

            if actual_sha != item["sha256"] or meta_pq.num_rows != item["num_rows"]:
                corrupted.append(item["filename"])
            else:
                ok_count += 1
                total_verified_rows += int(meta_pq.num_rows)
            progress.update(task, advance=1)

    table = Table(title="Resultado de Verificación Criptográfica del Corpus SECOP II")
    table.add_column("Métrica de Custodia", style="cyan")
    table.add_column("Valor Verificado", justify="right", style="bold green")

    table.add_row(
        "Archivos Parquet esperados / verificados OK",
        f"{ok_count} / {len(expected_files)}",
    )
    table.add_row(
        "Archivos faltantes / corruptos", f"{len(missing)} / {len(corrupted)}"
    )
    table.add_row("Total de filas crudas verificadas", f"{total_verified_rows:,}")
    table.add_row("Tamaño físico total en disco", f"{manifest['total_size_gb']:.2f} GB")
    table.add_row(
        "Raíz Merkle SHA-256 del Corpus", f"{manifest['corpus_merkle_sha256'][:32]}..."
    )
    table.add_row("Tiempo de auditoría", f"{time.perf_counter() - t0:.2f} s")

    console.print("\n", table)
    if missing or corrupted:
        raise SystemExit(
            f"¡FALLO DE INTEGRIDAD! Faltantes: {missing} | Corruptos: {corrupted}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingesta Socrata -> 188 Parquets + Custodia SHA-256"
    )
    parser.add_argument(
        "--mode",
        choices=["verify", "paginated-api", "stream-bulk"],
        default="verify",
        help="Modo: 'verify' sella/verifica los 188 Parquets locales; 'paginated-api' o 'stream-bulk' descargan de Socrata.",
    )
    parser.add_argument(
        "--app-token", type=str, default=os.environ.get("SOCRATA_APP_TOKEN")
    )
    args = parser.parse_args()

    if args.mode == "verify":
        verify_local_corpus_integrity()
    elif args.mode == "paginated-api":
        ingest_paginated_soda_api(app_token=args.app_token)
    elif args.mode == "stream-bulk":
        ingest_bulk_stream()
