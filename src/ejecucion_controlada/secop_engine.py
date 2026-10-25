from __future__ import annotations

import gc
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from rich.console import Console
from rich.table import Table
from transformers import AutoModel, AutoTokenizer

ARTIFACTS_DIR = Path("artefactos")
MODEL_NAME = "intfloat/multilingual-e5-base"
D_OUT = 512
R_DEG = 16
INV_SQRT_D = float(1.0 / np.sqrt(D_OUT))

# Umbral optimizado para conmutación a Régimen 1 (250k nodos ocupan solo ~17.1 MB en RAM)
REGIME1_MAX_VALID_NODES = 250_000

RABITQ_DTYPE = np.dtype(
    [
        ("bits", np.uint8, (64,)),
        ("o_dot_x", np.float32),
        ("int8_scale", np.float32),
    ]
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

BYTE_SIGNS = np.array(
    [[1.0 if (b >> (7 - bit)) & 1 else -1.0 for bit in range(8)] for b in range(256)],
    dtype=np.float32,
)
ARANGE_64 = np.arange(64)

STOP_AND_BOILERPLATE = {
    "de",
    "la",
    "el",
    "en",
    "y",
    "a",
    "los",
    "del",
    "las",
    "un",
    "por",
    "con",
    "no",
    "una",
    "su",
    "para",
    "es",
    "al",
    "lo",
    "como",
    "mas",
    "o",
    "pero",
    "sus",
    "le",
    "ha",
    "me",
    "si",
    "sin",
    "sobre",
    "este",
    "ya",
    "entre",
    "cuando",
    "todo",
    "esta",
    "ser",
    "son",
    "dos",
    "tambien",
    "fue",
    "habia",
    "era",
    "muy",
    "anos",
    "hasta",
    "desde",
    "contrato",
    "contratos",
    "contratar",
    "contratacion",
    "prestacion",
    "servicios",
    "servicio",
    "suministro",
    "suministrar",
    "adquisicion",
    "adquirir",
    "compra",
    "comprar",
    "compraventa",
    "elementos",
    "equipos",
    "apoyo",
    "gestion",
    "profesionales",
    "realizar",
    "municipio",
    "distrito",
    "entidad",
    "proyecto",
    "marco",
    "objeto",
    "proceso",
    "mediante",
    "caracter",
    "temporal",
    "dotacion",
    "fortalecimiento",
    "atencion",
    "situacion",
    "emergencia",
}


def strip_accents(text: str) -> str:
    """Elimina tildes/diacríticos (NFD) y normaliza a minúsculas para comparaciones inmunes a ortografía."""
    return "".join(
        c
        for c in unicodedata.normalize("NFD", text.lower().strip())
        if unicodedata.category(c) != "Mn"
    )


def _spanish_stem(word: str) -> str:
    """Recorta sufijos plurales y de género en español (ej. ambulancias -> ambulancia)."""
    if len(word) > 6 and word.endswith("ciones"):
        return word[:-6] + "cion"
    if len(word) > 5 and word.endswith("es") and word[-3] not in "aeiou":
        return word[:-2]
    if len(word) > 4 and word.endswith("s"):
        w = word[:-1]
        if len(w) > 5 and w[-1] in ("a", "o"):
            return w[:-1]
        return w
    if len(word) > 5 and word[-1] in ("a", "o"):
        return word[:-1]
    return word


def extract_discriminative_stems(query_text: str) -> list[str]:
    """Extrae raíces léxicas discriminativas (>= 4 letras) excluyendo jerga genérica y normalizando plurales."""
    norm = strip_accents(query_text)
    tokens = re.findall(r"[a-z0-9]{4,}", norm)
    stems = []
    for t in tokens:
        if t not in STOP_AND_BOILERPLATE:
            stem = _spanish_stem(t)
            if len(stem) >= 4 and stem not in stems:
                stems.append(stem)
    return list(dict.fromkeys(stems))


@dataclass
class SearchFilter:
    departamento: str | None = None
    modalidad: str | None = None
    tipo_contrato: str | None = None
    anio_min: int | None = None
    anio_max: int | None = None
    valor_min_cop: float | None = None
    valor_max_cop: float | None = None


class SecopHybridEngine:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        self.console = Console()
        self.artifacts_dir = artifacts_dir

        vocab_path = artifacts_dir / "facet_vocabularies.json"
        self.vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.n_contracts = int(self.vocab["n_contracts"])
        self.n_nodes = int(self.vocab["n_nodes"])

        self.rabitq = np.memmap(
            artifacts_dir / "vectors_rabitq_72b.bin",
            dtype=RABITQ_DTYPE,
            mode="r",
            shape=(self.n_nodes,),
        )
        self.csr_graph = np.memmap(
            artifacts_dir / "csr_graph_r16.bin",
            dtype=np.uint32,
            mode="r",
            shape=(self.n_nodes, R_DEG),
        )
        self.int8_ssd = np.memmap(
            artifacts_dir / "vectors_rerank_int8.bin",
            dtype=np.int8,
            mode="r",
            shape=(self.n_nodes, D_OUT),
        )

        medoids = np.load(artifacts_dir / "entry_medoids_k2048.npz")
        self.centroids_int8 = medoids["centroids_int8"]
        self.medoid_ids = medoids["medoid_node_ids"]

        proj = np.load(artifacts_dir / "projection_matrix.npz")
        self.mean_vec = proj["mean_vec"].squeeze(0).astype(np.float32)
        self.w_fused = proj["w_fused"].astype(np.float32)

        self.node_indptr = np.fromfile(
            artifacts_dir / "node_indptr_uint32.bin", dtype=np.uint32
        )
        self.node_contract_indices = np.fromfile(
            artifacts_dir / "node_contract_indices_uint32.bin", dtype=np.uint32
        )
        self.contract_to_node = np.fromfile(
            artifacts_dir / "contract_to_node_uint32.bin", dtype=np.uint32
        )
        self.filters = np.fromfile(
            artifacts_dir / "contract_filters_10b.bin", dtype=FILTER_DTYPE
        )

        self.pq_file = pq.ParquetFile(artifacts_dir / "secop_contracts_clean.parquet")
        rg_counts = [
            self.pq_file.metadata.row_group(i).num_rows
            for i in range(self.pq_file.num_row_groups)
        ]
        self.rg_offsets = np.zeros(len(rg_counts) + 1, dtype=np.int64)
        np.cumsum(rg_counts, out=self.rg_offsets[1:])
        self._row_cache: dict[int, dict[str, Any]] = {}

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_query(self, query_text: str) -> np.ndarray:
        clean_q = query_text.strip()
        formatted = f"query: {clean_q}"
        enc = self.tokenizer(
            [formatted],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**enc)[0].float().cpu()
        mask = enc["attention_mask"].cpu().unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        normed_768 = (
            torch.nn.functional.normalize(pooled, p=2, dim=1).squeeze(0).numpy()
        )

        proj_512 = (normed_768 - self.mean_vec) @ self.w_fused
        q_rot = proj_512 / (np.linalg.norm(proj_512) + 1e-9)
        return q_rot.astype(np.float32)

    def evaluate_structured_filter(
        self, f: SearchFilter | None
    ) -> tuple[np.ndarray | None, np.ndarray | None, int]:
        if f is None or all(v is None for v in f.__dict__.values()):
            return None, None, self.n_nodes

        mask = np.ones(self.n_contracts, dtype=bool)

        if f.departamento is not None:
            target = strip_accents(f.departamento)
            matches = [
                i
                for i, name in enumerate(self.vocab["departamento"])
                if target in strip_accents(name)
            ]
            if matches:
                mask &= np.isin(self.filters["depto_id"], matches)
            else:
                mask[:] = False

        if f.modalidad is not None:
            target = strip_accents(f.modalidad)
            matches = [
                i
                for i, name in enumerate(self.vocab["modalidad_de_contratacion"])
                if target in strip_accents(name)
            ]
            if matches:
                mask &= np.isin(self.filters["modalidad_id"], matches)
            else:
                mask[:] = False

        if f.tipo_contrato is not None:
            target = strip_accents(f.tipo_contrato)
            matches = [
                i
                for i, name in enumerate(self.vocab["tipo_de_contrato"])
                if target in strip_accents(name)
            ]
            if matches:
                mask &= np.isin(self.filters["tipo_id"], matches)
            else:
                mask[:] = False

        if f.anio_min is not None:
            mask &= self.filters["anio_firma"] >= f.anio_min
        if f.anio_max is not None:
            mask &= self.filters["anio_firma"] <= f.anio_max
        if f.valor_min_cop is not None:
            mask &= self.filters["valor_cop"] >= f.valor_min_cop
        if f.valor_max_cop is not None:
            mask &= self.filters["valor_cop"] <= f.valor_max_cop

        valid_node_bool = np.zeros(self.n_nodes, dtype=bool)
        valid_node_bool[self.contract_to_node[mask]] = True
        n_valid_nodes = int(np.count_nonzero(valid_node_bool))
        return mask, valid_node_bool, n_valid_nodes

    def hydrate_and_lexical_rerank(
        self,
        candidate_rows_and_scores: list[tuple[int, float]],
        query_text: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        cols = [
            "id_contrato",
            "nombre_entidad",
            "departamento",
            "modalidad_de_contratacion",
            "fecha_de_firma",
            "valor_del_contrato",
            "objeto_del_contrato",
        ]
        stems = extract_discriminative_stems(query_text)
        n_stems = max(len(stems), 1)

        rg_cache: dict[int, pa.Table] = {}
        enriched_candidates: list[dict[str, Any]] = []

        for row_id, cos_score in candidate_rows_and_scores:
            if row_id in self._row_cache:
                base_dict = dict(self._row_cache[row_id])
            else:
                rg_idx = int(np.searchsorted(self.rg_offsets[1:], row_id, side="right"))
                local_offset = int(row_id - self.rg_offsets[rg_idx])
                if rg_idx not in rg_cache:
                    rg_cache[rg_idx] = self.pq_file.read_row_group(rg_idx, columns=cols)
                rg_slice = rg_cache[rg_idx].slice(local_offset, 1)
                base_dict = {k: rg_slice[k][0].as_py() for k in cols}
                if len(self._row_cache) < 2048:
                    self._row_cache[row_id] = dict(base_dict)

            norm_obj = strip_accents(base_dict.get("objeto_del_contrato") or "")
            lex_hits = sum(1 for st in stems if st in norm_obj)
            lex_coverage = lex_hits / n_stems
            hybrid_score = cos_score + (0.18 * lex_coverage if stems else 0.0)

            base_dict["cosine_score"] = round(cos_score, 4)
            base_dict["lexical_hits"] = f"{lex_hits}/{len(stems)}"
            base_dict["hybrid_score"] = round(hybrid_score, 4)
            base_dict["row_id"] = row_id
            enriched_candidates.append(base_dict)

        rg_cache.clear()
        pa.default_memory_pool().release_unused()
        gc.collect()

        enriched_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return enriched_candidates[:top_k]

    def search(
        self,
        query_text: str,
        search_filter: SearchFilter | None = None,
        top_k: int = 5,
        profile: str = "forense",
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        q_rot = self.encode_query(query_text)
        t_encode_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        contract_mask, valid_node_bool, n_valid_nodes = self.evaluate_structured_filter(
            search_filter
        )
        t_filter_ms = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        q_LUT = (q_rot.reshape(64, 8) @ BYTE_SIGNS.T) * INV_SQRT_D

        if n_valid_nodes == 0:
            return {
                "results": [],
                "telemetry": {
                    "regime": "0 contratos cumplen el filtro",
                    "device_encoder": self.device,
                    "nodes_visited_in_ram": 0,
                    "reranked_ssd_candidates": 0,
                    "encode_query_ms": round(t_encode_ms, 2),
                    "filter_eval_ms": round(t_filter_ms, 2),
                    "vector_search_rerank_ms": 0.0,
                    "hydrate_parquet_ms": 0.0,
                    "total_ram_rss_mb": round(
                        psutil.Process().memory_info().rss / (1024 * 1024), 1
                    ),
                },
            }

        visited_count = 0
        if valid_node_bool is not None and n_valid_nodes <= REGIME1_MAX_VALID_NODES:
            valid_nodes = np.flatnonzero(valid_node_bool).astype(np.uint32)
            visited_count = len(valid_nodes)
            regime = f"Régimen 1 (Barrido Directo RaBitQ sobre {visited_count:,} nodos filtrados)"

            sub_rows = self.rabitq[valid_nodes]
            nbits = sub_rows["bits"]
            nodot = sub_rows["o_dot_x"]
            est_cos = q_LUT[ARANGE_64, nbits].sum(axis=1) / np.clip(nodot, 1e-4, None)

            rerank_m = min(350, visited_count)
            if visited_count <= rerank_m:
                top_node_candidates = valid_nodes
            else:
                top_pos = np.argpartition(-est_cos, rerank_m)[:rerank_m]
                top_node_candidates = valid_nodes[top_pos]
        else:
            regime = f"Régimen 2 (Grafo CSR Vamana + Medoides sobre {n_valid_nodes:,} nodos activos)"
            n_medoids = 64 if profile == "forense" else 48
            expand_k = 48 if profile == "forense" else 40
            max_hops = 15 if profile == "forense" else 12
            rerank_m = 500 if profile == "forense" else 350

            c_scores = self.centroids_int8.astype(np.float32) @ q_rot
            top_c = np.argpartition(-c_scores, n_medoids)[:n_medoids]
            seeds = np.unique(self.medoid_ids[top_c]).tolist()

            if valid_node_bool is not None:
                valid_medoids = [int(m) for m in self.medoid_ids if valid_node_bool[m]]
                if len(valid_medoids) > 0:
                    seeds = list(dict.fromkeys(seeds + valid_medoids[:32]))

            visited = set(int(n) for n in seeds)
            expanded: set[int] = set()
            next_batch = list(visited)
            nav_pool: list[tuple[float, int]] = []
            valid_pool: list[tuple[float, int]] = []

            for hop in range(max_hops + 1):
                if next_batch:
                    idx_arr = np.array(next_batch, dtype=np.uint32)
                    sub_rows = self.rabitq[idx_arr]
                    nbits = sub_rows["bits"]
                    nodot = sub_rows["o_dot_x"]
                    est_cos = q_LUT[ARANGE_64, nbits].sum(axis=1) / np.clip(
                        nodot, 1e-4, None
                    )

                    if valid_node_bool is not None:
                        is_val = valid_node_bool[idx_arr]
                        for sc, nid, v_ok in zip(
                            est_cos.tolist(), next_batch, is_val.tolist()
                        ):
                            if v_ok:
                                valid_pool.append((sc, nid))
                                nav_pool.append((sc + 0.04, nid))
                            else:
                                nav_pool.append((sc, nid))
                    else:
                        pairs = list(zip(est_cos.tolist(), next_batch))
                        nav_pool.extend(pairs)
                        valid_pool.extend(pairs)

                    nav_pool.sort(reverse=True)

                if hop < max_hops:
                    to_expand = []
                    for _, u in nav_pool:
                        if u not in expanded:
                            expanded.add(u)
                            to_expand.append(u)
                            if len(to_expand) >= expand_k:
                                break
                    if not to_expand:
                        break
                    cand_neighbors = self.csr_graph[to_expand].ravel()
                    next_batch = [
                        int(v)
                        for v in np.unique(cand_neighbors)
                        if int(v) not in visited
                    ]
                    visited.update(next_batch)

            visited_count = len(visited)
            valid_pool.sort(reverse=True)
            top_node_candidates = np.array(
                [nid for _, nid in valid_pool[:rerank_m]], dtype=np.uint32
            )

        cand_int8 = self.int8_ssd[top_node_candidates].astype(np.float32)
        cand_norms = np.linalg.norm(cand_int8, axis=1) + 1e-8
        exact_cos = (cand_int8 @ q_rot) / cand_norms
        sorted_order = np.argsort(-exact_cos)

        pool_for_lexical = max(top_k * 3, 15)
        selected_contracts: list[tuple[int, float]] = []
        for idx_ord in sorted_order:
            u_node = int(top_node_candidates[idx_ord])
            score = float(exact_cos[idx_ord])
            p_start = int(self.node_indptr[u_node])
            p_end = int(self.node_indptr[u_node + 1])
            contract_rows = self.node_contract_indices[p_start:p_end]

            for c_id in contract_rows:
                c_int = int(c_id)
                if contract_mask is None or contract_mask[c_int]:
                    selected_contracts.append((c_int, score))
                    break
            if len(selected_contracts) >= pool_for_lexical:
                break

        t_index_ms = (time.perf_counter() - t2) * 1000.0
        t3 = time.perf_counter()
        hydrated = self.hydrate_and_lexical_rerank(
            selected_contracts, query_text, top_k=top_k
        )
        t_hydrate_ms = (time.perf_counter() - t3) * 1000.0
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)

        return {
            "results": hydrated,
            "telemetry": {
                "regime": regime,
                "device_encoder": self.device,
                "nodes_visited_in_ram": visited_count,
                "reranked_ssd_candidates": len(top_node_candidates),
                "encode_query_ms": round(t_encode_ms, 2),
                "filter_eval_ms": round(t_filter_ms, 2),
                "vector_search_rerank_ms": round(t_index_ms, 2),
                "hydrate_parquet_ms": round(t_hydrate_ms, 2),
                "total_ram_rss_mb": round(rss_mb, 1),
            },
        }


def print_test_block(
    console: Console, title: str, query: str, res_cold: dict, res_warm: dict
) -> None:
    console.print(
        f"\n[bold magenta]================================================================================[/bold magenta]"
    )
    console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print(f"• Consulta: [italic]'{query}'[/italic]")

    t_table = Table(
        title="Comparativa de Telemetría: Caché Fría (1.ª Consulta) vs. Caché Caliente (2.ª Consulta)"
    )
    t_table.add_column("Métrica", style="cyan")
    t_table.add_column("Pasada 1 (Cold SSD)", justify="right", style="yellow")
    t_table.add_column("Pasada 2 (Warm RAM)", justify="right", style="bold green")

    c_tel = res_cold["telemetry"]
    w_tel = res_warm["telemetry"]
    for k in [
        "regime",
        "device_encoder",
        "nodes_visited_in_ram",
        "reranked_ssd_candidates",
        "encode_query_ms",
        "filter_eval_ms",
        "vector_search_rerank_ms",
        "hydrate_parquet_ms",
        "total_end_to_end_ms",
        "total_ram_rss_mb",
    ]:
        t_table.add_row(k, str(c_tel.get(k)), str(w_tel.get(k)))
    console.print(t_table)

    r_table = Table(
        title="Top-5 Contratos Recuperados (Fusión Híbrida Densa + Cobertura Léxica)"
    )
    r_table.add_column("ID Contrato", style="cyan")
    r_table.add_column(
        "Score Híbrido\n(Cos | Lex)", justify="right", style="bold green"
    )
    r_table.add_column("Entidad / Depto")
    r_table.add_column("Modalidad / Fecha", justify="center")
    r_table.add_column("Valor COP", justify="right")
    r_table.add_column("Objeto del Contrato")

    for r in res_warm["results"]:
        val_str = (
            f"${r['valor_del_contrato']:,.0f}" if r["valor_del_contrato"] else "$0"
        )
        z_val = r["cosine_score"] * np.sqrt(512)
        obj_short = (r["objeto_del_contrato"] or "")[:85] + "..."
        r_table.add_row(
            str(r["id_contrato"]),
            f"{r['hybrid_score']:.4f} (+{z_val:.1f}σ)\n[{r['cosine_score']:.4f} | {r['lexical_hits']}]",
            f"{r['nombre_entidad']} ({r['departamento']})",
            f"{r['modalidad_de_contratacion']}\n[{r['fecha_de_firma']}]",
            val_str,
            obj_short,
        )
    console.print(r_table)


if __name__ == "__main__":
    console = Console()
    engine = SecopHybridEngine()

    q_2a = "interventoría técnica administrativa y financiera de obras de infraestructura vial y pavimentación"
    f_2a = SearchFilter(
        modalidad="CONCURSO DE MERITOS",
        anio_min=2021,
        valor_min_cop=500_000_000,
    )

    t0 = time.perf_counter()
    res_2a_cold = engine.search(
        query_text=q_2a, search_filter=f_2a, top_k=5, profile="forense"
    )
    res_2a_cold["telemetry"]["total_end_to_end_ms"] = round(
        (time.perf_counter() - t0) * 1000.0, 2
    )

    t0 = time.perf_counter()
    res_2a_warm = engine.search(
        query_text=q_2a, search_filter=f_2a, top_k=5, profile="forense"
    )
    res_2a_warm["telemetry"]["total_end_to_end_ms"] = round(
        (time.perf_counter() - t0) * 1000.0, 2
    )

    print_test_block(
        console,
        "PRUEBA 2A — RÉGIMEN 2 HÍBRIDO NACIONAL (Concurso de Méritos | Año >= 2021 | >= $500M COP)",
        q_2a,
        res_2a_cold,
        res_2a_warm,
    )

    q_2b = "adquisición de ambulancias medicalizadas y equipos biomédicos para urgencias hospitalarias"

    t0 = time.perf_counter()
    res_2b_cold = engine.search(
        query_text=q_2b, search_filter=None, top_k=5, profile="forense"
    )
    res_2b_cold["telemetry"]["total_end_to_end_ms"] = round(
        (time.perf_counter() - t0) * 1000.0, 2
    )

    t0 = time.perf_counter()
    res_2b_warm = engine.search(
        query_text=q_2b, search_filter=None, top_k=5, profile="forense"
    )
    res_2b_warm["telemetry"]["total_end_to_end_ms"] = round(
        (time.perf_counter() - t0) * 1000.0, 2
    )

    print_test_block(
        console,
        "PRUEBA 2B — RÉGIMEN 2 GLOBAL PURO (Sin Filtros sobre los 5,632,886 Contratos de Colombia)",
        q_2b,
        res_2b_cold,
        res_2b_warm,
    )
