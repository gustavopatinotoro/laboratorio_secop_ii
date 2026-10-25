from __future__ import annotations

from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# CONFIGURACIÓN ESTÉTICA DE PUBLICACIÓN CIENTÍFICA (IEEE / ACM / TESIS)
# =============================================================================
OUT_DIR = Path("docs/figures")

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
    }
)

PALETTE = {
    "navy": "#1B365D",
    "teal": "#008080",
    "amber": "#D97706",
    "crimson": "#B91C1C",
    "emerald": "#047857",
    "slate": "#475569",
    "light_slate": "#94A3B8",
    "purple": "#6D28D9",
}


def save_fig(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{name}.png"
    pdf_path = OUT_DIR / f"{name}.pdf"
    fig.savefig(png_path, format="png", dpi=300)
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)
    print(f"✔ Generada: {png_path} y {pdf_path}")


# =============================================================================
# FIGURA 1: COLAPSO AL GRAFO COCIENTE 1:N Y NULOS BUROCRÁTICOS (FASE 1 Y 2)
# =============================================================================
def plot_fig01_corpus_quotient_and_validity() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Panel A: Colapso del Corpus a Grafo Cociente
    stages = [
        "Filas Crudas\n(189 Parquets)",
        "Contratos Únicos\n(id_contrato)",
        "Nodos Únicos\n(Hash 64-bit Exacto)",
        "Estimación HLL\n(HyperLogLog)",
    ]
    counts_m = [5.648071, 5.647774, 3.240802, 3.093505]
    colors_a = [PALETTE["slate"], PALETTE["navy"], PALETTE["emerald"], PALETTE["teal"]]

    bars = ax1.bar(
        stages, counts_m, color=colors_a, width=0.58, edgecolor="black", linewidth=0.8
    )
    ax1.set_ylabel("Millones de Registros / Vértices")
    ax1.set_title("(a) Reducción Topológica al Grafo Cociente 1:N (42.62% Redundancia)")
    ax1.set_ylim(0, 6.6)

    for bar, val in zip(bars, counts_m):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.12,
            f"{val * 1e6:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    # Anotación de absorción 1:N
    ax1.annotate(
        "Absorción CSR 1:N:\n-2,406,972 contratos\nclonados (-42.62%)",
        xy=(2, 3.240802),
        xytext=(1.15, 4.45),
        arrowprops=dict(arrowstyle="->", color=PALETTE["crimson"], lw=1.5),
        fontsize=8.5,
        fontweight="bold",
        color=PALETTE["crimson"],
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="#FEE2E2", edgecolor=PALETTE["crimson"]
        ),
    )

    # Panel B: Validez Efectiva vs Nulos Burocráticos
    cols = [
        "condiciones_de_entrega",
        "ciudad",
        "fecha_inicio_contrato",
        "departamento",
        "fecha_de_firma",
        "descripcion_proceso",
        "objeto_del_contrato",
    ]
    valid_pct = np.array([69.35, 78.71, 97.83, 98.05, 98.82, 99.82, 99.83])
    buro_pct = 100.0 - valid_pct

    y_pos = np.arange(len(cols))
    ax2.barh(
        y_pos,
        valid_pct,
        color=PALETTE["navy"],
        height=0.6,
        label="Validez Efectiva Real (%)",
    )
    ax2.barh(
        y_pos,
        buro_pct,
        left=valid_pct,
        color=PALETTE["amber"],
        height=0.6,
        label="Nulos Burocráticos ('NO DEFINIDO', 'N/A')",
    )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(cols)
    ax2.set_xlabel("Porcentaje del Corpus (%)")
    ax2.set_title("(b) Validez Efectiva vs. Nulos Burocráticos en Atributos Clave")
    ax2.set_xlim(0, 108)
    ax2.legend(loc="lower right", frameon=True)

    for i, (v, b) in enumerate(zip(valid_pct, buro_pct)):
        if b >= 1.5:
            ax2.text(
                100.8,
                i,
                f"-{b:.2f}% nulo",
                va="center",
                fontsize=8,
                color=PALETTE["crimson"],
                fontweight="bold",
            )
        else:
            ax2.text(
                100.8,
                i,
                f"{v:.2f}% OK",
                va="center",
                fontsize=8,
                color=PALETTE["emerald"],
                fontweight="bold",
            )

    fig.tight_layout()
    save_fig(fig, "fig01_corpus_quotient_and_validity")


# =============================================================================
# FIGURA 2: EL MURO DE MEMORIA EN 8 GB RAM (COMPARATIVA ARQUITECTÓNICA)
# =============================================================================
def plot_fig02_memory_wall_8gb_comparison() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    configs = [
        "1. HNSW float32 Estándar\n(1:1 sobre 5.65M contratos, D=768)",
        "2. HNSW float32 en Cociente\n(Sobre 3.24M nodos únicos, D=768)",
        "3. HNSW float16 Estándar\n(1:1 sobre 5.65M contratos, D=768)",
        "4. HNSW int8 Escalar\n(1:1 sobre 5.65M contratos, D=768)",
        "5. Arquitectura Propuesta (Tesis)\n(RaBitQ 72B + CSR R=16 + Puente 1:N)",
    ]

    # Huella en GB para vectores + grafo + metadatos en RAM
    vec_gb = np.array([16.16, 9.27, 8.08, 4.04, 0.217])  # 222.53 MB = 0.217 GB
    graph_gb = np.array([0.68, 0.39, 0.68, 0.68, 0.193])  # 197.80 MB = 0.193 GB
    bridge_gb = np.array(
        [0.00, 0.054, 0.00, 0.00, 0.109]
    )  # Puente 55.44 MB + Filtros 53.86 MB = 0.107 GB

    y = np.arange(len(configs))
    ax.barh(y, vec_gb, height=0.55, color=PALETTE["navy"], label="Vectores en RAM")
    ax.barh(
        y,
        graph_gb,
        left=vec_gb,
        height=0.55,
        color=PALETTE["teal"],
        label="Grafo de Vecinos en RAM",
    )
    ax.barh(
        y,
        bridge_gb,
        left=vec_gb + graph_gb,
        height=0.55,
        color=PALETTE["amber"],
        label="Puente CSR 1:N + Filtros 10B",
    )

    # Líneas de límite de hardware
    ax.axvline(
        8.0,
        color=PALETTE["crimson"],
        linestyle="-",
        linewidth=2.0,
        label="RAM Física Total del Equipo (8.0 GB)",
    )
    ax.axvline(
        5.5,
        color=PALETTE["amber"],
        linestyle="--",
        linewidth=1.8,
        label="Presupuesto Máximo Disponible (~5.5 GB tras SO)",
    )

    totals = vec_gb + graph_gb + bridge_gb
    for i, tot in enumerate(totals):
        status = (
            "DESBORDA (Swap/OOM)"
            if tot > 5.5
            else "ÓPTIMO EN 8 GB (0.52 GB índice | ~0.88 GB RSS)"
        )
        col = PALETTE["crimson"] if tot > 5.5 else PALETTE["emerald"]
        ax.text(
            tot + 0.25,
            i,
            f"{tot:.2f} GB — {status}",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color=col,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(configs)
    ax.invert_yaxis()
    ax.set_xlabel("Memoria Principal Requerida en RAM (Gigabytes)")
    ax.set_title(
        "Superación del Muro de Memoria en 8 GB RAM sobre el Corpus Completo SECOP II (5.65M Contratos)"
    )
    ax.set_xlim(0, 21.5)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    save_fig(fig, "fig02_memory_wall_8gb_comparison")


# =============================================================================
# FIGURA 3: ESQUEMA FÍSICO DE BYTES EN RAM (RABITQ 72B, CSR 64B, FILTRO 10B)
# =============================================================================
def plot_fig03_byte_layout_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    def draw_block(
        x: float, y: float, w: float, h: float, color: str, title: str, subtitle: str
    ) -> None:
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.3",
            facecolor=color,
            edgecolor="black",
            linewidth=1.1,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h * 0.62,
            title,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=9,
        )
        ax.text(
            x + w / 2,
            y + h * 0.28,
            subtitle,
            ha="center",
            va="center",
            color="#F1F5F9",
            fontsize=8,
        )

    # Fila 1: Registro RaBitQ (72 bytes por nodo -> 222.53 MB)
    ax.text(
        2,
        88,
        "1. Registro Vectorial Asimétrico RaBitQ (`vectors_rabitq_72b.bin` — 72 Bytes/Nodo = 222.53 MB en RAM)",
        fontweight="bold",
        fontsize=10,
        color=PALETTE["navy"],
    )
    draw_block(
        2,
        68,
        62,
        16,
        PALETTE["navy"],
        "bits: uint8[64] (Bytes 0..63 = 512 bits de signo)",
        "Evaluado vía tabla q_LUT (64 KB) residente en Caché L1 de CPU",
    )
    draw_block(
        65, 68, 16, 16, PALETTE["teal"], "o_dot_x: float32", "Bytes 64..67 (4 B)"
    )
    draw_block(
        82, 68, 16, 16, PALETTE["slate"], "int8_scale: float32", "Bytes 68..71 (4 B)"
    )

    # Fila 2: Registro Grafo CSR Vamana R=16 (64 bytes por nodo -> 197.80 MB)
    ax.text(
        2,
        57,
        "2. Grafo Plano R-Regular sin `indptr` (`csr_graph_r16.bin` — 16 × uint32 = 64 Bytes/Nodo = 197.80 MB en RAM)",
        fontweight="bold",
        fontsize=10,
        color=PALETTE["emerald"],
    )
    draw_block(
        2,
        37,
        46,
        16,
        PALETTE["emerald"],
        "8 Aristas Primarias Intra-Clúster (32 B)",
        "Poda Angular Vamana (α = 1.15) + 1-Hop NN-Descent",
    )
    draw_block(
        49,
        37,
        24,
        16,
        PALETTE["teal"],
        "4 Aristas Puente (16 B)",
        "2.º Clúster Voronoï",
    )
    draw_block(
        74,
        37,
        24,
        16,
        PALETTE["purple"],
        "4 Aristas Inversas (16 B)",
        "Garantía de Conexidad",
    )

    # Fila 3: Registro de Filtros Empaquetados (10 bytes por contrato -> 53.86 MB)
    ax.text(
        2,
        26,
        "3. Tupla Columnar de Predicados (`contract_filters_10b.bin` — 10 Bytes/Contrato = 53.86 MB para 5,647,774 Contratos)",
        fontweight="bold",
        fontsize=10,
        color=PALETTE["amber"],
    )
    draw_block(
        2,
        6,
        34,
        16,
        PALETTE["amber"],
        "valor_cop: float32 (4 B)",
        "Bytes 0..3 (Cuantía en COP)",
    )
    draw_block(
        37,
        6,
        20,
        16,
        PALETTE["crimson"],
        "anio_firma: uint16 (2 B)",
        "Bytes 4..5 (Año)",
    )
    draw_block(58, 6, 10, 16, PALETTE["slate"], "depto_id", "uint8 (1B)")
    draw_block(69, 6, 10, 16, PALETTE["slate"], "modalidad", "uint8 (1B)")
    draw_block(80, 6, 9, 16, PALETTE["slate"], "tipo_id", "uint8 (1B)")
    draw_block(90, 6, 8, 16, PALETTE["slate"], "estado", "uint8 (1B)")

    fig.suptitle(
        "Diseño Físico de Estructuras Binarias Alineadas en Memoria Principal (Cero Punteros Dinámicos)",
        y=0.98,
    )
    fig.tight_layout()
    save_fig(fig, "fig03_byte_layout_architecture")


# =============================================================================
# FIGURA 4: FRONTERA DE PARETO AUDITADA (RECALL@10 VS LATENCIA Y VISITADOS)
# =============================================================================
def plot_fig04_pareto_frontier_recall_latency() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    profiles = [
        "1. Rápido\n(32/32/10/200)",
        "2. Balanceado\n(48/40/12/350)",
        "3. Forense\n(64/48/15/500)",
    ]
    p50_ms = np.array([7.48, 10.20, 14.15])
    p95_ms = np.array([10.27, 14.93, 18.71])
    recall_final = np.array([81.60, 90.60, 94.00])
    recall_graph = np.array([81.80, 90.80, 94.20])
    visited_nodes = np.array([1639, 2301, 3233])
    visited_pct = (visited_nodes / 3_240_802) * 100.0

    # Panel A: Recall@10 vs Latencia P50 y P95
    ax1.plot(
        p50_ms,
        recall_graph,
        "o--",
        color=PALETTE["teal"],
        lw=2,
        markersize=8,
        label="Recall@10 Grafo (Descubrimiento Topológico)",
    )
    ax1.plot(
        p50_ms,
        recall_final,
        "s-",
        color=PALETTE["navy"],
        lw=2.5,
        markersize=8,
        label="Recall@10 Final (RaBitQ 1-bit + Re-rank INT8)",
    )
    ax1.plot(
        p95_ms,
        recall_final,
        "^:",
        color=PALETTE["amber"],
        lw=1.8,
        markersize=7,
        label="Latencia Cola P95 (ms)",
    )

    ax1.axhline(
        90.0,
        color=PALETTE["crimson"],
        linestyle="-.",
        alpha=0.7,
        label="Meta Académica (90.0% Recall@10)",
    )

    for i, prof in enumerate(["Rápido", "Balanceado", "Forense"]):
        ax1.annotate(
            f"{prof}\n{recall_final[i]:.2f}% ({p50_ms[i]:.2f} ms)",
            xy=(p50_ms[i], recall_final[i]),
            xytext=(p50_ms[i] - 0.6, recall_final[i] - 3.6),
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor=PALETTE["navy"],
                alpha=0.9,
            ),
        )

    ax1.set_xlabel("Latencia de Búsqueda Vectorial + Re-ranking SSD (ms)")
    ax1.set_ylabel("Recall@10 contra Fuerza Bruta Exacta (%)")
    ax1.set_title(
        "(a) Frontera de Pareto: Recall@10 vs. Latencia (Brecha RaBitQ = -0.20%)"
    )
    ax1.set_xlim(5.5, 20.5)
    ax1.set_ylim(76, 97.5)
    ax1.legend(loc="lower right", frameon=True)

    # Panel B: Exploración Sub-lineal (Nodos visitados vs Recall)
    bars = ax2.bar(
        profiles,
        visited_nodes,
        color=[PALETTE["slate"], PALETTE["teal"], PALETTE["emerald"]],
        width=0.5,
        edgecolor="black",
    )
    ax2.set_ylabel("Nodos Semánticos Visitados en RAM (de 3,240,802)")
    ax2.set_title("(b) Eficiencia Sub-Lineal: Fracción del Grafo Explorada (< 0.10%)")
    ax2.set_ylim(0, 3900)

    for bar, n_vis, pct, rec in zip(bars, visited_nodes, visited_pct, recall_final):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            n_vis + 90,
            f"{n_vis:,} nodos\n({pct:.4f}% del total)\nRecall: {rec:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    fig.tight_layout()
    save_fig(fig, "fig04_pareto_frontier_recall_latency")


# =============================================================================
# FIGURA 5: ESTUDIO DE ABLACIÓN DE OPTIMIZACIONES DE BAJO NIVEL (LOG SCALE)
# =============================================================================
def plot_fig05_ablation_low_level_speedups() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    categories = [
        "Codificación Transformer\n(`encode_query_ms`)",
        "Búsqueda Grafo + Re-rank\n(`vector_search_rerank_ms` R2)",
        "Barrido Filtrado RaBitQ\n(`vector_search_rerank_ms` R1)",
        "Huella de Memoria Proceso\n(`total_ram_rss_mb` en MB)",
    ]

    before_vals = np.array([705.02, 507.02, 281.95, 1405.2])
    after_vals = np.array([21.92, 8.08, 3.00, 851.4])
    speedup_labels = [
        "32.2× más rápido\n(MPS FP16)",
        "62.7× más rápido\n(Acceso Contiguo)",
        "94.0× más rápido\n(Caché Caliente)",
        "-39.4% RAM\n(Arrow Release)",
    ]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(
        x - width / 2,
        before_vals,
        width,
        label="Línea Base Inicial (CPU FP32 / Vista Strided / Sin Release)",
        color=PALETTE["crimson"],
        edgecolor="black",
    )
    rects2 = ax.bar(
        x + width / 2,
        after_vals,
        width,
        label="Motor Optimizado Final (MPS FP16 / Indexación por Fila / Warm)",
        color=PALETTE["emerald"],
        edgecolor="black",
    )

    ax.set_yscale("log")
    ax.set_ylabel("Magnitud en Escala Logarítmica (ms / MB)")
    ax.set_title(
        "Impacto de las Optimizaciones de Hardware y Memoria de Bajo Nivel (Estudio de Ablación)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(1, 4500)
    ax.legend(loc="upper right", frameon=True)

    for r1, r2, lbl in zip(rects1, rects2, speedup_labels):
        h1, h2 = r1.get_height(), r2.get_height()
        ax.text(
            r1.get_x() + r1.get_width() / 2,
            h1 * 1.12,
            f"{h1:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=PALETTE["crimson"],
        )
        ax.text(
            r2.get_x() + r2.get_width() / 2,
            h2 * 1.15,
            f"{h2:.2f}\n({lbl})",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=PALETTE["emerald"],
        )

    fig.tight_layout()
    save_fig(fig, "fig05_ablation_low_level_speedups")


# =============================================================================
# FIGURA 6: TELEMETRÍA END-TO-END FRÍO (SSD) VS CALIENTE (RAM) (PRUEBAS 2A Y 2B)
# =============================================================================
def plot_fig06_cold_vs_warm_telemetry() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    runs = [
        "Prueba 2A (Régimen 1: 4,410 nodos)\nPasada 1 — Caché Fría (SSD)",
        "Prueba 2A (Régimen 1: 4,410 nodos)\nPasada 2 — Caché Caliente (RAM)",
        "Prueba 2B (Régimen 2: 3.24M nodos)\nPasada 1 — Caché Fría (SSD)",
        "Prueba 2B (Régimen 2: 3.24M nodos)\nPasada 2 — Caché Caliente (RAM)",
    ]

    encode_ms = np.array([358.89, 21.92, 22.38, 22.10])
    filter_ms = np.array([46.18, 24.22, 0.00, 0.00])
    search_ms = np.array([281.95, 3.00, 402.17, 8.08])
    hydrate_ms = np.array([504.09, 53.90, 398.81, 54.42])
    totals = encode_ms + filter_ms + search_ms + hydrate_ms

    y = np.arange(len(runs))
    h = 0.52

    ax.barh(
        y,
        encode_ms,
        height=h,
        color=PALETTE["navy"],
        label="1. Codificación Consulta (`encode_query_ms` en MPS FP16)",
    )
    ax.barh(
        y,
        filter_ms,
        left=encode_ms,
        height=h,
        color=PALETTE["amber"],
        label="2. Evaluación Filtro O(N) (`filter_eval_ms`)",
    )
    ax.barh(
        y,
        search_ms,
        left=encode_ms + filter_ms,
        height=h,
        color=PALETTE["emerald"],
        label="3. Búsqueda Vectorial RaBitQ + Re-rank INT8 (`vector_search_rerank_ms`)",
    )
    ax.barh(
        y,
        hydrate_ms,
        left=encode_ms + filter_ms + search_ms,
        height=h,
        color=PALETTE["teal"],
        label="4. Hidratación Parquet + Re-rank Léxico (`hydrate_parquet_ms`)",
    )

    for i, tot in enumerate(totals):
        ax.text(
            tot + 18,
            i,
            f"Total: {tot:.2f} ms ( búsqueda vec: {search_ms[i]:.2f} ms )",
            va="center",
            fontsize=8.8,
            fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(runs)
    ax.invert_yaxis()
    ax.set_xlabel("Latencia Acumulada End-to-End (Milisegundos)")
    ax.set_title(
        "Desglose de Telemetría End-to-End: Caché Fría (Cold SSD) vs. Caché Caliente (Warm RAM)"
    )
    ax.set_xlim(0, 1480)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    save_fig(fig, "fig06_cold_vs_warm_telemetry")


# =============================================================================
# FIGURA 7: PUNTO DE CRUCE DEL CONMUTADOR DE RÉGIMEN DUAL (N_UMBRAL = 20,000)
# =============================================================================
def plot_fig07_dual_regime_crossover() -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.8))

    n_valid = np.logspace(2, 6.51, 200)  # De 100 a 3,240,802 nodos activos
    # Modelo empírico calibrado con los puntos medidos (4,410 nodos -> 3.00 ms; 20,000 -> ~9.5 ms)
    lat_regime1_direct = 1.15 + (n_valid / 4410.0) * 1.85
    # Régimen 2 (Grafo CSR Vamana): latencia casi constante (~8.08 - 14.15 ms), pero con penalización si hay desconexión en baja cardinalidad
    lat_regime2_graph = 8.08 + 1.8 * np.exp(-n_valid / 15000.0)

    ax.plot(
        n_valid,
        lat_regime1_direct,
        "-",
        color=PALETTE["emerald"],
        lw=2.5,
        label="Régimen 1: Barrido Directo RaBitQ L1-LUT (100% Exactitud Subgrafo)",
    )
    ax.plot(
        n_valid,
        lat_regime2_graph,
        "--",
        color=PALETTE["navy"],
        lw=2.5,
        label="Régimen 2: Navegación Grafo CSR Vamana + Medoides (Complejidad Sub-lineal)",
    )

    ax.axvline(
        20000,
        color=PALETTE["crimson"],
        linestyle=":",
        lw=2.2,
        label="Umbral de Conmutación (N_umbral = 20,000 nodos)",
    )
    ax.axvspan(
        100,
        20000,
        color="#D1FAE5",
        alpha=0.35,
        label="Zona Operativa Régimen 1 (Alta Selectividad)",
    )
    ax.axvspan(
        20000,
        3_240_802,
        color="#DBEAFE",
        alpha=0.35,
        label="Zona Operativa Régimen 2 (Baja Selectividad / Global)",
    )

    # Marcar puntos experimentales reales medidos en el Mac
    ax.scatter(
        [4410], [3.00], color=PALETTE["emerald"], s=90, zorder=5, edgecolor="black"
    )
    ax.annotate(
        "Prueba 2A Medida:\n4,410 nodos → 3.00 ms",
        xy=(4410, 3.00),
        xytext=(450, 12.0),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontweight="bold",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white"),
    )

    ax.scatter(
        [3_240_802], [8.08], color=PALETTE["navy"], s=90, zorder=5, edgecolor="black"
    )
    ax.annotate(
        "Prueba 2B Medida:\n3,240,802 nodos → 8.08 ms",
        xy=(3_240_802, 8.08),
        xytext=(180000, 22.0),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontweight="bold",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white"),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        "Número de Nodos Semánticos Activos que Cumplen el Filtro |V_valid| (Escala Log)"
    )
    ax.set_ylabel("Latencia de Búsqueda Vectorial + Re-rank (ms, Log)")
    ax.set_title(
        "Conmutación Dinámica de Régimen Dual para Búsqueda Filtrada bajo Memoria Restringida (P4 / H4)"
    )
    ax.set_xlim(100, 4_000_000)
    ax.set_ylim(1.0, 1200)
    ax.legend(loc="upper left", frameon=True, fontsize=8.2)

    fig.tight_layout()
    save_fig(fig, "fig07_dual_regime_crossover")


# =============================================================================
# FIGURA 8: DISTRIBUCIÓN ISOTRÓPICA NULA VS SIGNIFICANCIA Z-SCORE (+7.0σ A +9.2σ)
# =============================================================================
def plot_fig08_isotropic_zscore_distribution() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))

    sigma = 1.0 / np.sqrt(512.0)  # ~0.044194
    x = np.linspace(-0.20, 0.58, 1000)
    pdf = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (x / sigma) ** 2)

    ax.plot(
        x,
        pdf,
        color=PALETTE["slate"],
        lw=2.0,
        label="Distribución Nula Isotrópica N(0, 1/√512) tras W_fused (σ = 0.0442)",
    )
    ax.fill_between(
        x,
        0,
        pdf,
        where=(np.abs(x) <= 3 * sigma),
        color=PALETTE["light_slate"],
        alpha=0.4,
        label="Banda de Ruido Aleatorio (±3σ = ±0.1326)",
    )
    ax.axvline(
        3 * sigma,
        color=PALETTE["amber"],
        linestyle="--",
        lw=1.6,
        label="Umbral de Significancia Estadística (+3.0σ)",
    )

    # Contratos reales recuperados
    p2b_cos = [0.3090, 0.3175, 0.3237, 0.3322, 0.3350]
    p2a_cos = [0.3325, 0.3532, 0.3595, 0.3895, 0.4069]
    p2a_hyb = [0.4657, 0.4675, 0.4720, 0.5245, 0.5419]

    ax.scatter(
        p2b_cos,
        [0.6] * len(p2b_cos),
        color=PALETTE["teal"],
        s=80,
        marker="o",
        zorder=5,
        label="Prueba 2B (Coseno INT8: +7.0σ a +7.6σ)",
    )
    ax.scatter(
        p2a_cos,
        [1.4] * len(p2a_cos),
        color=PALETTE["navy"],
        s=80,
        marker="s",
        zorder=5,
        label="Prueba 2A (Coseno INT8: +7.5σ a +9.2σ)",
    )
    ax.scatter(
        p2a_hyb,
        [2.2] * len(p2a_hyb),
        color=PALETTE["emerald"],
        s=90,
        marker="^",
        zorder=5,
        label="Prueba 2A (Score Híbrido Cos + 0.18·Lex: hasta 0.5419)",
    )

    ax.annotate(
        "Top-1 Prueba 2A:\nCos = 0.4069 (+9.2σ)\nScore Híbrido = 0.5419",
        xy=(0.5419, 2.2),
        xytext=(0.38, 5.2),
        arrowprops=dict(arrowstyle="->", color=PALETTE["emerald"], lw=1.5),
        fontweight="bold",
        fontsize=8.5,
        bbox=dict(
            boxstyle="round,pad=0.25", facecolor="#D1FAE5", edgecolor=PALETTE["emerald"]
        ),
    )

    ax.set_xlabel(
        "Similitud Coseno / Score Híbrido en el Espacio Isotrópico de 512 Dimensiones"
    )
    ax.set_ylabel("Densidad de Probabilidad")
    ax.set_title(
        "Separación Estadística entre el Azar Isotrópico (D=512) y los Contratos Recuperados por el Motor"
    )
    ax.set_xlim(-0.18, 0.58)
    ax.set_ylim(0, 9.8)
    ax.legend(loc="upper right", frameon=True, fontsize=8.3)

    fig.tight_layout()
    save_fig(fig, "fig08_isotropic_zscore_distribution")


if __name__ == "__main__":
    print("Generando las 8 figuras científicas de la tesis en docs/figures/ ...")
    plot_fig01_corpus_quotient_and_validity()
    plot_fig02_memory_wall_8gb_comparison()
    plot_fig03_byte_layout_architecture()
    plot_fig04_pareto_frontier_recall_latency()
    plot_fig05_ablation_low_level_speedups()
    plot_fig06_cold_vs_warm_telemetry()
    plot_fig07_dual_regime_crossover()
    plot_fig08_isotropic_zscore_distribution()
    print(
        "\n✔ ¡Las 8 figuras (PNG a 300 DPI + PDF vectorial) fueron creadas con éxito!"
    )
