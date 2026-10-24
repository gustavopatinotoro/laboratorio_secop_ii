from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.table import Table

ARTIFACTS_DIR = Path("artefactos")

REQUIRED_FILES = [
    "secop_contracts_clean.parquet",
    "node_hashes_uint64.bin",
    "vectors_rabitq_72b.bin",
    "csr_graph_r16.bin",
    "entry_medoids_k2048.npz",
    "projection_matrix.npz",
    "vectors_rerank_int8.bin",
]


def get_ram_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def normalize_str_array(col: pa.Array | pa.ChunkedArray) -> list[str]:
    filled = pc.fill_null(col, "NO DEFINIDO")
    upper = pc.utf8_upper(pc.utf8_trim_whitespace(filled))
    return upper.to_pylist()


def build_hybrid_bridge() -> None:
    console = Console()
    t0 = time.perf_counter()

    console.print(
        "[bold cyan]=== Fase 4 Local (Alineación Física 1:1 por Row-Groups): Puente CSR Nodos <-> Contratos ===[/bold cyan]"
    )

    missing = [f for f in REQUIRED_FILES if not (ARTIFACTS_DIR / f).exists()]
    if missing:
        console.print(
            f"[bold red]Error:[/bold red] Faltan los siguientes archivos en '{ARTIFACTS_DIR}/': {missing}"
        )
        return

    hashes_path = ARTIFACTS_DIR / "node_hashes_uint64.bin"
    rabitq_path = ARTIFACTS_DIR / "vectors_rabitq_72b.bin"
    csr_path = ARTIFACTS_DIR / "csr_graph_r16.bin"
    int8_path = ARTIFACTS_DIR / "vectors_rerank_int8.bin"
    clean_parquet_path = ARTIFACTS_DIR / "secop_contracts_clean.parquet"

    n_nodes = hashes_path.stat().st_size // 8
    assert rabitq_path.stat().st_size == n_nodes * 72, (
        "Desajuste en vectors_rabitq_72b.bin"
    )
    assert csr_path.stat().st_size == n_nodes * 16 * 4, "Desajuste en csr_graph_r16.bin"
    assert int8_path.stat().st_size == n_nodes * 512, (
        "Desajuste en vectors_rerank_int8.bin"
    )

    pq_file = pq.ParquetFile(clean_parquet_path)
    n_contracts = pq_file.metadata.num_rows
    num_rgs = pq_file.num_row_groups

    console.print(
        f"✔ Integridad verificada: [bold green]{n_nodes:,}[/bold green] nodos | [bold green]{n_contracts:,}[/bold green] contratos en {num_rgs} Row-Groups."
    )
    console.print(f"• RAM inicial del proceso: {get_ram_mb():.1f} MB\n")

    # -------------------------------------------------------------------------
    # PASO 1: Lectura Secuencial Exacta por Row-Groups con PyArrow (Cero SQL Joins)
    # Garantiza que el índice c_id (0..5,632,885) coincida 1:1 con el archivo Parquet físico
    # -------------------------------------------------------------------------
    console.print(
        "[bold yellow]Paso 1/3:[/bold yellow] Escaneando Row-Groups en orden físico estricto (0 desordenamiento)..."
    )

    FILTER_DTYPE = np.dtype(
        [
            ("valor_cop", np.float32),
            ("anio_firma", np.uint16),
            ("depto_id", np.uint8),
            ("modalidad_id", np.uint8),
            ("tipo_id", np.uint8),
            ("estado_id", np.uint8),
        ]
    )

    filter_arr = np.empty(n_contracts, dtype=FILTER_DTYPE)
    contract_hashes = np.empty(n_contracts, dtype=np.uint64)

    depto_map: dict[str, int] = {}
    mod_map: dict[str, int] = {}
    tipo_map: dict[str, int] = {}
    est_map: dict[str, int] = {}

    def map_codes(values: list[str], vocab_dict: dict[str, int]) -> np.ndarray:
        out = np.empty(len(values), dtype=np.uint8)
        for i, v in enumerate(values):
            code = vocab_dict.get(v)
            if code is None:
                code = len(vocab_dict)
                vocab_dict[v] = code
            out[i] = code
        return out

    cols_to_read = [
        "text_node_hash",
        "valor_del_contrato",
        "fecha_de_firma",
        "departamento",
        "modalidad_de_contratacion",
        "tipo_de_contrato",
        "estado_contrato",
    ]

    row_cursor = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} Row-Groups"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Empaquetando filtros en orden físico...", total=num_rgs
        )
        for rg_idx in range(num_rgs):
            rg_table = pq_file.read_row_group(rg_idx, columns=cols_to_read)
            rg_len = rg_table.num_rows
            s, e = row_cursor, row_cursor + rg_len

            h_np = (
                rg_table["text_node_hash"]
                .to_numpy(zero_copy_only=False)
                .view(np.uint64)
            )
            contract_hashes[s:e] = h_np

            val_col = pc.fill_null(rg_table["valor_del_contrato"], 0.0)
            filter_arr["valor_cop"][s:e] = val_col.to_numpy(
                zero_copy_only=False
            ).astype(np.float32)

            yr_col = pc.fill_null(pc.year(rg_table["fecha_de_firma"]), 0)
            filter_arr["anio_firma"][s:e] = yr_col.to_numpy(
                zero_copy_only=False
            ).astype(np.uint16)

            filter_arr["depto_id"][s:e] = map_codes(
                normalize_str_array(rg_table["departamento"]), depto_map
            )
            filter_arr["modalidad_id"][s:e] = map_codes(
                normalize_str_array(rg_table["modalidad_de_contratacion"]), mod_map
            )
            filter_arr["tipo_id"][s:e] = map_codes(
                normalize_str_array(rg_table["tipo_de_contrato"]), tipo_map
            )
            filter_arr["estado_id"][s:e] = map_codes(
                normalize_str_array(rg_table["estado_contrato"]), est_map
            )

            row_cursor = e
            del rg_table
            pa.default_memory_pool().release_unused()
            progress.update(task, advance=1)

    filter_bin_path = ARTIFACTS_DIR / "contract_filters_10b.bin"
    filter_arr.tofile(filter_bin_path)
    del filter_arr
    gc.collect()
    console.print(
        f"   ✔ Filtros físicos guardados ({filter_bin_path.stat().st_size / (1024**2):.2f} MB) | RAM: {get_ram_mb():.1f} MB"
    )

    # -------------------------------------------------------------------------
    # PASO 2: Enlace Exacto Contrato -> Nodo Semántico vía Búsqueda Binaria NumPy
    # -------------------------------------------------------------------------
    console.print(
        "[bold yellow]Paso 2/3:[/bold yellow] Enlazando 5.63M contratos con sus 3.23M nodos vía búsqueda binaria vectorizada..."
    )
    node_hashes = np.fromfile(hashes_path, dtype=np.uint64)
    sort_perm = np.argsort(node_hashes, kind="stable").astype(np.uint32)
    sorted_node_hashes = node_hashes[sort_perm]
    del node_hashes
    gc.collect()

    pos = np.searchsorted(sorted_node_hashes, contract_hashes)
    pos = np.clip(pos, 0, n_nodes - 1)
    matched = int(np.sum(sorted_node_hashes[pos] == contract_hashes))
    console.print(
        f"   ✔ Coincidencia exacta de hashes verificada: [bold green]{matched:,} / {n_contracts:,} ({matched / n_contracts * 100:.2f}%)[/bold green]"
    )
    assert matched == n_contracts, (
        f"Error de integridad: solo coincidieron {matched} de {n_contracts} hashes."
    )

    contract_to_node = sort_perm[pos].astype(np.uint32)
    del sorted_node_hashes, sort_perm, pos, contract_hashes
    gc.collect()

    c2n_path = ARTIFACTS_DIR / "contract_to_node_uint32.bin"
    contract_to_node.tofile(c2n_path)

    # -------------------------------------------------------------------------
    # PASO 3: Construcción del Puente CSR Invertido (Nodo -> Filas Físicas Parquet)
    # -------------------------------------------------------------------------
    console.print(
        "[bold yellow]Paso 3/3:[/bold yellow] Construyendo índice CSR invertido (node_indptr + node_contract_indices)..."
    )
    counts = np.bincount(contract_to_node, minlength=n_nodes)
    node_indptr = np.zeros(n_nodes + 1, dtype=np.uint32)
    np.cumsum(counts, out=node_indptr[1:])
    del counts

    node_contract_indices = np.argsort(contract_to_node, kind="stable").astype(
        np.uint32
    )
    del contract_to_node
    gc.collect()

    indptr_path = ARTIFACTS_DIR / "node_indptr_uint32.bin"
    indices_path = ARTIFACTS_DIR / "node_contract_indices_uint32.bin"
    node_indptr.tofile(indptr_path)
    node_contract_indices.tofile(indices_path)
    del node_indptr, node_contract_indices
    gc.collect()

    def dict_to_list(d: dict[str, int]) -> list[str]:
        res = [""] * len(d)
        for k, v in d.items():
            res[v] = k
        return res

    vocab_path = ARTIFACTS_DIR / "facet_vocabularies.json"
    vocab_payload = {
        "n_contracts": int(n_contracts),
        "n_nodes": int(n_nodes),
        "departamento": dict_to_list(depto_map),
        "modalidad_de_contratacion": dict_to_list(mod_map),
        "tipo_de_contrato": dict_to_list(tipo_map),
        "estado_contrato": dict_to_list(est_map),
    }
    vocab_path.write_text(
        json.dumps(vocab_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    table = Table(title="Artefactos del Puente Híbrido Local (Alineación Física 1:1)")
    table.add_column("Archivo", style="cyan")
    table.add_column("Tamaño", justify="right", style="bold green")
    table.add_column("Rol en el Motor de 8 GB RAM")

    for fname, role in [
        (
            "node_indptr_uint32.bin",
            "Punteros CSR Nodo -> Inicio/Fin de sus contratos (RAM)",
        ),
        (
            "node_contract_indices_uint32.bin",
            "IDs físicos exactos en secop_contracts_clean.parquet (RAM)",
        ),
        (
            "contract_to_node_uint32.bin",
            "Mapeo directo Fila Física -> Nodo semántico (RAM)",
        ),
        (
            "contract_filters_10b.bin",
            "Arreglo columnar de 10 B/contrato alineado físicamente (RAM)",
        ),
        (
            "facet_vocabularies.json",
            "Diccionarios de facetas (Departamento, Modalidad, Tipo, Estado)",
        ),
    ]:
        p = ARTIFACTS_DIR / fname
        table.add_row(fname, f"{p.stat().st_size / (1024**2):.2f} MB", role)

    console.print("\n", table)
    console.print(
        f"[bold green]✔ Puente Híbrido reconstruido con alineación física 1:1 en {time.perf_counter() - t0:.1f}s | RAM final: {get_ram_mb():.1f} MB[/bold green]"
    )


if __name__ == "__main__":
    build_hybrid_bridge()
