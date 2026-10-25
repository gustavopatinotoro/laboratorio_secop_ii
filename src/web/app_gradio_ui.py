from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
import gradio as gr
import pandas as pd
import psutil

# 1. Asegurar que Python encuentre la carpeta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src/")))

from ejecucion_controlada.secop_engine import SecopHybridEngine, SearchFilter

# Modelo de 3B parámetros en 4-bit (~1.7 GB en UMA): mucho más capaz en razonamiento analítico que el 1.5B
LOCAL_SLM_REPO = "mlx-community/Qwen2.5-3B-Instruct-4bit"
_slm_model = None
_slm_tokenizer = None

print("Cargando el Motor Híbrido SECOP II en memoria...")
engine = SecopHybridEngine()
print("¡Motor de búsqueda listo y operativo!")


def get_local_slm():
    """Carga perezosa del modelo SLM de 3B en 4 bits sobre Apple Silicon (MLX)."""
    global _slm_model, _slm_tokenizer
    if _slm_model is None:
        from mlx_lm import load

        print(f"Descargando/Cargando modelo analítico 3B 4-bit ({LOCAL_SLM_REPO})...")
        _slm_model, _slm_tokenizer = load(LOCAL_SLM_REPO)
        print("¡Modelo Qwen2.5-3B-Instruct-4bit listo en Memoria Unificada!")
    return _slm_model, _slm_tokenizer


def generate_local_llm_synthesis(
    query_text: str,
    results: list[dict],
    min_p: float,
    med_p: float,
    avg_p: float,
    max_p: float,
    top_entity: str,
    top_modality: str,
) -> tuple[str, float]:
    """Genera un dictamen forense estructurado sobre TODOS los contratos recuperados."""
    t_llm0 = time.perf_counter()
    try:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        return (
            "⚠️ *Para activar la redacción con IA local, ejecuta:* `uv add mlx-lm`",
            0.0,
        )

    try:
        model, tokenizer = get_local_slm()

        # Pasar TODOS los contratos recuperados (hasta 10) con su desviación frente a la mediana precalculada
        context_lines = []
        for idx, r in enumerate(results[:10], start=1):
            val = float(r["valor_del_contrato"] or 0.0)
            diff_med = ((val - med_p) / med_p * 100.0) if med_p > 0 else 0.0
            sign_str = f"+{diff_med:.0f}%" if diff_med >= 0 else f"{diff_med:.0f}%"
            obj_clean = " ".join(str(r["objeto_del_contrato"]).split())[:160]
            context_lines.append(
                f"#{idx} [{r['id_contrato']}] | Entidad: {r['nombre_entidad']} ({r['departamento']}) | "
                f"Modalidad: {r['modalidad_de_contratacion']} | Fecha: {r['fecha_de_firma']} | "
                f"Valor: ${val:,.0f} COP ({sign_str} vs mediana) | Hit Léxico: {r['lexical_hits']} | "
                f"Resumen Objeto: {obj_clean}"
            )
        context_block = "\n".join(context_lines)

        system_prompt = (
            "Eres un Auditor Forense Senior de la Contraloría experto en SECOP II. "
            "Tu tarea NO es repetir ni copiar literalmente el objeto de los contratos, sino "
            "ANALIZAR críticamente la tabla completa de contratos recuperados. "
            "Responde siempre en español con exactamente 3 viñetas cortas y completas."
        )
        user_prompt = (
            f" Necesidad buscada: '{query_text}'\n"
            f"Estadísticas globales del lote ({len(results)} contratos): "
            f"Mínimo=${min_p:,.0f} | Mediana=${med_p:,.0f} | Promedio=${avg_p:,.0f} | Máximo=${max_p:,.0f} COP. "
            f"Entidad más repetida: {top_entity}. Modalidad dominante: {top_modality}.\n\n"
            f"Tabla de contratos a auditar:\n{context_block}\n\n"
            "Entrega tu dictamen exactamente en estas 3 viñetas (sin introducciones ni despedidas, no te cortes a mitad de frase):\n"
            "1. **Pertinencia Técnica y Filtrado de Intrusos:** Indica qué contratos responden con mayor exactitud al bien/servicio buscado (citando su ID `[CO1...]` y departamento) y advierte si algún contrato de la lista corresponde a otro rubro distinto que deba descartarse del estudio de mercado.\n"
            "2. **Análisis de Dispersión de Precios y Posibles Sobrecostos:** Compara los contratos más costosos frente a la mediana (`${med_p:,.0f}` COP), explicando por qué se desvían (ej. compras masivas departamentales vs. compra unitaria) y señala los IDs con mayor desviación positiva.\n"
            "3. **Patrón de Modalidades y Recomendación de Auditoría:** Evalúa si el uso de Contratación Directa o Régimen Especial estaba justificado (por ejemplo, por urgencia manifiesta COVID-19 o naturaleza de E.S.E. hospitalaria) y qué modalidad se recomienda para una nueva licitación."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        sampler = make_sampler(temp=0.15)
        response_text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=480,
            sampler=sampler,
            verbose=False,
        )
        llm_ms = (time.perf_counter() - t_llm0) * 1000.0
        gc.collect()
        return response_text.strip(), llm_ms
    except Exception as exc:
        return f"⚠️ *Error al ejecutar el modelo local MLX:* `{exc}`", 0.0


def web_search(
    query_text: str,
    departamento_filter: str,
    modalidad_filter: str,
    anio_min: int,
    anio_max: int,
    valor_min_cop: float,
    top_k: int,
    use_local_llm: bool,
) -> tuple[pd.DataFrame, str, str]:
    """Ejecuta la búsqueda semántica e híbrida sobre SECOP-Vec y genera telemetría + síntesis RAG."""
    if not query_text.strip():
        return (
            pd.DataFrame(),
            "Por favor, introduce una consulta semántica válida.",
            "⚠️ Consulta vacía.",
        )

    depto = (
        departamento_filter.strip().upper()
        if departamento_filter and departamento_filter != "TODOS"
        else None
    )
    mod = (
        modalidad_filter.strip().upper()
        if modalidad_filter and modalidad_filter != "TODOS"
        else None
    )

    search_flt = SearchFilter(
        departamento=depto,
        modalidad=mod,
        anio_min=int(anio_min) if anio_min > 0 else None,
        anio_max=int(anio_max) if anio_max > 0 else None,
        valor_min_cop=float(valor_min_cop) if valor_min_cop > 0 else None,
    )

    t0 = time.perf_counter()
    res = engine.search(
        query_text=query_text,
        search_filter=search_flt,
        top_k=int(top_k),
        profile="forense",
    )
    search_elapsed_ms = (time.perf_counter() - t0) * 1000.0

    results = res["results"]
    telemetry = res["telemetry"]

    if not results:
        telemetry_str = (
            f"Régimen: {telemetry.get('regime')} | "
            f"Tiempo Búsqueda: {search_elapsed_ms:.2f} ms | Resultados: 0"
        )
        return (
            pd.DataFrame(),
            telemetry_str,
            "📊 **INFORME EJECUTIVO RAG:** No se encontraron contratos con los filtros seleccionados.",
        )

    table_data = []
    for r in results:
        val_cop = (
            f"${r['valor_del_contrato']:,.0f}" if r["valor_del_contrato"] else "$0"
        )
        z_score = r["cosine_score"] * (512**0.5)
        table_data.append(
            {
                "ID Contrato": r["id_contrato"],
                "Score Híbrido": f"{r['hybrid_score']:.4f} (+{z_score:.1f}σ)",
                "Coseno | Léxico": f"{r['cosine_score']:.4f} | {r['lexical_hits']}",
                "Entidad": r["nombre_entidad"],
                "Depto": r["departamento"],
                "Modalidad": r["modalidad_de_contratacion"],
                "Fecha": str(r["fecha_de_firma"]),
                "Valor COP": val_cop,
                "Objeto del Contrato": r["objeto_del_contrato"],
            }
        )

    df = pd.DataFrame(table_data)

    # --- CAPA 1: SÍNTESIS RAG ESTRUCTURADA DETERMINISTA ---
    prices = sorted(
        [r["valor_del_contrato"] for r in results if r["valor_del_contrato"] > 0]
    )
    if prices:
        min_p = prices[0]
        max_p = prices[-1]
        avg_p = sum(prices) / len(prices)
        n_p = len(prices)
        med_p = (
            prices[n_p // 2]
            if n_p % 2 == 1
            else 0.5 * (prices[n_p // 2 - 1] + prices[n_p // 2])
        )
    else:
        min_p = max_p = avg_p = med_p = 0.0

    entities = [r["nombre_entidad"] for r in results if r["nombre_entidad"]]
    top_entity = max(set(entities), key=entities.count) if entities else "N/A"

    modalities = [
        r["modalidad_de_contratacion"]
        for r in results
        if r["modalidad_de_contratacion"]
    ]
    top_modality = max(set(modalities), key=modalities.count) if modalities else "N/A"

    max_z = max(r["cosine_score"] * (512**0.5) for r in results)

    rag_summary = (
        f"### 📊 INFORME EJECUTIVO RAG (Top-{len(results)} Contratos Recuperados)\n"
        f"- **Rango de Cuantías (COP):** Mín: `${min_p:,.0f}` | Mediana: `${med_p:,.0f}` | "
        f"Promedio: `${avg_p:,.0f}` | Máx: `${max_p:,.0f}`\n"
        f"- **Entidad Predominante en Resultados:** {top_entity}\n"
        f"- **Modalidad de Selección Frecuente:** {top_modality}\n"
        f"- **Diagnóstico Estadístico:** Significancia máxima de **+{max_z:.1f}σ** sobre el ruido isotrópico (D=512)."
    )

    # --- CAPA 2: DICTAMEN FORENSE CON MODELO LOCAL 3B 4-BIT ---
    llm_ms = 0.0
    if use_local_llm:
        llm_text, llm_ms = generate_local_llm_synthesis(
            query_text,
            results,
            min_p,
            med_p,
            avg_p,
            max_p,
            top_entity,
            top_modality,
        )
        rag_summary += (
            f"\n\n---\n### 🤖 Dictamen Forense Generado Localmente (`{LOCAL_SLM_REPO}` — {llm_ms:.0f} ms)\n"
            f"{llm_text}"
        )

    rss_now_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)

    telemetry_summary = (
        f"⚡ Régimen: {telemetry.get('regime')} | "
        f"Codificación (MPS): {telemetry.get('encode_query_ms')} ms | "
        f"Filtro O(N): {telemetry.get('filter_eval_ms')} ms | "
        f"Vector Search + Re-rank: {telemetry.get('vector_search_rerank_ms')} ms | "
        f"Hidratación Parquet: {telemetry.get('hydrate_parquet_ms')} ms | "
        f"Total Búsqueda: {search_elapsed_ms:.2f} ms"
        + (f" | Generación LLM 3B: {llm_ms:.0f} ms" if use_local_llm else "")
        + f" | RAM RSS Actual: {rss_now_mb:.1f} MB"
    )

    return df, telemetry_summary, rag_summary


with gr.Blocks(
    theme=gr.themes.Soft(), title="Buscador Semántico SECOP II - Tesis"
) as demo:
    gr.Markdown("# 🏛️ Motor de Búsqueda Híbrido y RAG para SECOP II (5.65M Contratos)")
    gr.Markdown(
        "Herramienta de consulta forense para contratación pública optimizada para equipos de **8 GB de RAM**. "
        "Combina **Grafo Cociente CSR Vamana + RaBitQ (72 B) + Re-ranking INT8** con síntesis ejecutiva RAG 100% local."
    )

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Consulta Semántica en Lenguaje Natural",
                placeholder="Ej. Interventoría técnica y administrativa para obras viales...",
                value="adquisición de ambulancias medicalizadas y equipos biomédicos para urgencias hospitalarias",
            )
        with gr.Column(scale=1):
            top_k_slider = gr.Slider(
                minimum=1, maximum=20, value=10, step=1, label="Top Resultados (K)"
            )
            use_llm_checkbox = gr.Checkbox(
                label="🤖 Activar Dictamen Forense con LLM Local (Qwen 3B 4-bit)",
                value=False,
            )

    with gr.Row():
        depto_dropdown = gr.Dropdown(
            label="Departamento",
            choices=[
                "TODOS",
                "ANTIOQUIA",
                "DISTRITO CAPITAL DE BOGOTÁ",
                "VALLE DEL CAUCA",
                "CUNDINAMARCA",
                "ATLÁNTICO",
                "SANTANDER",
                "CALDAS",
                "TOLIMA",
                "BOLÍVAR",
            ],
            value="TODOS",
        )
        mod_dropdown = gr.Dropdown(
            label="Modalidad de Contratación",
            choices=[
                "TODOS",
                "CONCURSO DE MÉRITOS",
                "CONTRATACIÓN DIRECTA",
                "CONTRATACIÓN RÉGIMEN ESPECIAL",
                "CONTRATACIÓN RÉGIMEN ESPECIAL (CON OFERTAS)",
                "MÍNIMA CUANTÍA",
                "LICITACIÓN PÚBLICA",
                "SELECCIÓN ABREVIADA SUBASTA INVERSA",
            ],
            value="TODOS",
        )

    with gr.Row():
        anio_min_num = gr.Number(label="Año Mínimo", value=2020, precision=0)
        anio_max_num = gr.Number(label="Año Máximo", value=2026, precision=0)
        valor_min_num = gr.Number(
            label="Cuantía Mínima (COP)", value=100_000_000, precision=0
        )

    search_btn = gr.Button(
        "🔍 Ejecutar Búsqueda Híbrida + Síntesis RAG", variant="primary"
    )

    with gr.Row():
        telemetry_output = gr.Textbox(
            label="Telemetría y Rendimiento del Motor (Búsqueda + RAM RSS)",
            interactive=False,
        )

    rag_output = gr.Markdown(
        value="*Realiza una búsqueda para ver el informe ejecutivo RAG.*",
        label="Informe Ejecutivo RAG (Síntesis Automática)",
    )

    results_table = gr.Dataframe(
        label="Contratos Encontrados",
        headers=[
            "ID Contrato",
            "Score Híbrido",
            "Coseno | Léxico",
            "Entidad",
            "Depto",
            "Modalidad",
            "Fecha",
            "Valor COP",
            "Objeto del Contrato",
        ],
        interactive=False,
    )

    search_btn.click(
        fn=web_search,
        inputs=[
            query_input,
            depto_dropdown,
            mod_dropdown,
            anio_min_num,
            anio_max_num,
            valor_min_num,
            top_k_slider,
            use_llm_checkbox,
        ],
        outputs=[results_table, telemetry_output, rag_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
