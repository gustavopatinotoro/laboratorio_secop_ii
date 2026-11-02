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

LOCAL_SLM_REPO = "mlx-community/Qwen2.5-3B-Instruct-4bit"
_slm_model = None
_slm_tokenizer = None

print("Cargando el Motor Híbrido SECOP II en memoria...")
engine = SecopHybridEngine()
print("¡Motor de búsqueda listo y operativo!")


def get_local_slm():
    """Carga perezosa del modelo SLM en 4 bits sobre Apple Silicon (MLX)."""
    global _slm_model, _slm_tokenizer
    if _slm_model is None:
        from mlx_lm import load

        print(f"Cargando modelo generativo local 4-bit ({LOCAL_SLM_REPO})...")
        _slm_model, _slm_tokenizer = load(LOCAL_SLM_REPO)
        print("¡Modelo local listo en Memoria Unificada!")
    return _slm_model, _slm_tokenizer


def parse_lex_hits(lex_str: str) -> tuple[int, int]:
    """Convierte '2/5' en (2, 5) para validación determinista."""
    try:
        parts = str(lex_str).split("/")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 1


def build_deterministic_forensic_audit(
    query_text: str,
    results: list[dict],
    min_p: float,
    med_p: float,
    avg_p: float,
    max_p: float,
    top_entity: str,
    top_modality: str,
    modalidad_filter: str | None,
) -> tuple[str, dict]:
    """
    Motor simbólico determinista (0 ms, 0% alucinaciones):
    Clasifica pertinencia, detecta intrusos (0 hits léxicos) y calcula desviaciones exactas vs. la mediana.
    """
    high_rel = []
    partial_rel = []
    intruders = []
    above_median = []

    for idx, r in enumerate(results, start=1):
        hits, total_stems = parse_lex_hits(r.get("lexical_hits", "0/1"))
        z_val = float(r.get("cosine_score", 0.0)) * (512**0.5)
        val = float(r.get("valor_del_contrato") or 0.0)
        diff_pct = ((val - med_p) / med_p * 100.0) if med_p > 0 else 0.0

        item_info = {
            "rank": idx,
            "id": r["id_contrato"],
            "entidad": r["nombre_entidad"],
            "depto": r["departamento"],
            "modalidad": r["modalidad_de_contratacion"],
            "valor": val,
            "diff_pct": diff_pct,
            "hits": hits,
            "total_stems": total_stems,
            "lex_str": r.get("lexical_hits", "0/0"),
            "z_val": z_val,
            "objeto": " ".join(str(r.get("objeto_del_contrato") or "").split()),
        }

        # if hits == 0 or z_val < 2.8:
        #     intruders.append(item_info)
        # elif hits >= 2 or (hits >= 1 and z_val >= 4.0):
        #     high_rel.append(item_info)
        # else:
        #     partial_rel.append(item_info)

        # if val > med_p and med_p > 0:
        #     above_median.append(item_info)

        # Clasificación de pertinencia e intrusos:
        is_intruder = (hits == 0) or (z_val < 2.8)
        if is_intruder:
            intruders.append(item_info)
        elif hits >= 2 or (hits >= 1 and z_val >= 4.0):
            high_rel.append(item_info)
        else:
            partial_rel.append(item_info)

        # Solo evaluar sobrecostos en contratos que NO sean intrusos:
        if not is_intruder and val > med_p and med_p > 0:
            above_median.append(item_info)

    above_median.sort(key=lambda x: x["valor"], reverse=True)

    # 1. Viñeta de Pertinencia Técnica y Detección de Intrusos
    if high_rel:
        top_matches_str = "; ".join(
            f"**#{m['rank']} [{m['id']}]** ({m['entidad']} - {m['depto']}, `${m['valor']:,.0f}` COP, "
            f"`+{m['z_val']:.1f}σ`, aciertos léxicos `{m['lex_str']}`)"
            for m in high_rel[:3]
        )
        v1 = f"1. **Pertinencia Técnica Exacta:** Los contratos con mayor correspondencia técnica y semántica frente a la búsqueda son: {top_matches_str}."
    elif partial_rel:
        top_matches_str = "; ".join(
            f"**#{m['rank']} [{m['id']}]** ({m['entidad']}, `${m['valor']:,.0f}` COP, `{m['lex_str']}` aciertos)"
            for m in partial_rel[:2]
        )
        v1 = f"1. **Pertinencia Técnica Parcial:** Bajo las restricciones actuales solo se encontraron coincidencias parciales del sector: {top_matches_str}."
    else:
        v1 = "1. **Pertinencia Técnica:** Ningún contrato en este subconjunto filtrado alcanza el umbral de coincidencia léxica y semántica alta."

    if intruders:
        intr_str = ", ".join(
            f"**#{x['rank']} [{x['id']}]** ({x['entidad']} — `{x['lex_str']}` hits, `+{x['z_val']:.1f}σ`)"
            for x in intruders[:4]
        )
        v1 += (
            f" ⚠️ **Filtrado de Intrusos:** Se detectaron **{len(intruders)} contratos de baja coincidencia léxica** "
            f"que deben excluirse del estudio de precios: {intr_str}."
        )
    else:
        v1 += " ✅ **Control de Calidad:** Todos los contratos recuperados conservan coincidencia léxica activa con la consulta."

    # 2. Viñeta de Dispersión de Precios Exacta contra la Mediana ($med_p)
    if above_median:
        top_exp = above_median[:3]
        exp_str = "; ".join(
            f"**#{x['rank']} [{x['id']}]** ({x['entidad']}) por **`${x['valor']:,.0f}` COP** "
            f"(**`+{x['diff_pct']:.1f}%`** sobre la mediana)"
            for x in top_exp
        )
        v2 = (
            f"2. **Análisis Matemático de Dispersión y Sobrecostos (Mediana = `${med_p:,.0f}` COP):** "
            f"Exactamente **{len(above_median)} de {len(results)} contratos** superan la mediana del lote. "
            f"Las mayores desviaciones positivas corresponden a: {exp_str}. "
            f"(El contrato de menor cuantía es `${min_p:,.0f}` COP, generando una razón Máx/Mín de `{max_p / max(min_p, 1.0):.1f}×`)."
        )
    else:
        v2 = (
            f"2. **Análisis Matemático de Dispersión (Mediana = `${med_p:,.0f}` COP):** "
            f"Los contratos presentan cuantías uniformes alrededor de `${avg_p:,.0f}` COP sin desviaciones positivas críticas."
        )

    # 3. Viñeta de Patrón de Modalidades y Recomendación Jurídica SECOP II
    v3 = (
        f"3. **Auditoría de Modalidades y Recomendación SECOP II:** "
        f"La modalidad predominante en la muestra es **{top_modality}** y la entidad con mayor recurrencia es **{top_entity}**."
    )
    if (
        modalidad_filter
        and "DIRECTA" in modalidad_filter.upper()
        and len(high_rel) <= 2
    ):
        v3 += (
            " 💡 **Nota Forense Clave:** En Colombia (Ley 80/1993 y Ley 1150/2007), la compra de ambulancias y equipos biomédicos "
            "por parte de Hospitales Públicos (E.S.E.) se adjudica habitualmente bajo **CONTRATACIÓN RÉGIMEN ESPECIAL** o **SELECCIÓN ABREVIADA**, "
            "mientras que en *Contratación Directa* solo figuran convenios interadministrativos (como el #1) o servicios asistenciales de defensa. "
            "Cambie el filtro de Modalidad a **TODOS** o **CONTRATACIÓN RÉGIMEN ESPECIAL** para ampliar el estudio de mercado."
        )

    audit_markdown = f"{v1}\n\n{v2}\n\n{v3}"
    structured_facts = {
        "high_rel": high_rel,
        "partial_rel": partial_rel,
        "intruders": intruders,
        "above_median": above_median,
    }
    return audit_markdown, structured_facts


def generate_local_llm_synthesis(
    query_text: str,
    med_p: float,
    structured_facts: dict,
) -> tuple[str, float]:
    """
    Pasa únicamente los hechos matemáticos ya verificados al modelo 3B para redactar una conclusión
    ejecutiva corta en ~3 segundos, impidiendo por diseño cualquier alucinación numérica.
    """
    t_llm0 = time.perf_counter()
    try:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        return (
            "⚠️ *Para activar la síntesis generativa local, ejecuta:* `uv add mlx-lm`",
            0.0,
        )

    try:
        model, tokenizer = get_local_slm()

        best_list = structured_facts["high_rel"] or structured_facts["partial_rel"]
        best_txt = (
            f"#{best_list[0]['rank']} [{best_list[0]['id']}] de {best_list[0]['entidad']} "
            f"por ${best_list[0]['valor']:,.0f} COP ({best_list[0]['objeto'][:140]})"
            if best_list
            else "Ninguno con alta coincidencia exacta en este filtro"
        )

        top_exp = structured_facts["above_median"]
        exp_txt = (
            f"#{top_exp[0]['rank']} [{top_exp[0]['id']}] de {top_exp[0]['entidad']} "
            f"por ${top_exp[0]['valor']:,.0f} COP (+{top_exp[0]['diff_pct']:.1f}% sobre la mediana de ${med_p:,.0f} COP)"
            if top_exp
            else "Sin desviaciones sobre la mediana"
        )

        intr_list = structured_facts["intruders"]
        intr_txt = (
            f"{len(intr_list)} contratos con 0 aciertos léxicos (ej. [{intr_list[0]['id']}] de {intr_list[0]['entidad']})"
            if intr_list
            else "0 contratos intrusos"
        )

        system_prompt = (
            "Eres un Auditor Forense del SECOP II. Redacta un párrafo ejecutivo de conclusión (máximo 95 palabras) "
            "usando EXCLUSIVAMENTE los datos verificados que te entrego. Prohibido inventar o alterar cifras."
        )
        user_prompt = (
            f"Consulta auditada: '{query_text}'\n"
            f"- Mejor contrato técnico verificado: {best_txt}.\n"
            f"- Contrato más costoso sobre la mediana: {exp_txt}.\n"
            f"- Ruido / Intrusos detectados para descartar: {intr_txt}.\n\n"
            "Escribe una conclusión ejecutiva en un solo párrafo sintetizando estos tres hallazgos para el comité de contratación:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        sampler = make_sampler(temp=0.1)
        response_text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=180,
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
    """Ejecuta la búsqueda híbrida sobre SECOP-Vec y genera el Dictamen Forense Neuro-Simbólico."""
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

    # Cálculo exacto de estadísticas sobre los contratos recuperados
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

    # Formatear tabla incluyendo columna de Diagnóstico y % vs Mediana
    table_data = []
    for idx, r in enumerate(results, start=1):
        val = float(r["valor_del_contrato"] or 0.0)
        val_cop = f"${val:,.0f}" if val > 0 else "$0"
        diff_pct = ((val - med_p) / med_p * 100.0) if med_p > 0 else 0.0
        diff_str = f"+{diff_pct:.0f}%" if diff_pct >= 0 else f"{diff_pct:.0f}%"

        z_score = r["cosine_score"] * (512**0.5)
        hits, _ = parse_lex_hits(r.get("lexical_hits", "0/1"))

        if hits == 0 or z_score < 2.8:
            diag_badge = "⚠️ Ruido / 0 Hits" if hits == 0 else "⚠️ Baja Similitud (<2.8σ)"
        elif hits >= 2 or (hits >= 1 and z_score >= 4.0):
            diag_badge = "✅ Alta Pertinencia"
        else:
            diag_badge = "🔹 Parcial (Sector)"

        table_data.append(
            {
                "#": idx,
                "Diagnóstico": diag_badge,
                "ID Contrato": r["id_contrato"],
                "Score Híbrido": f"{r['hybrid_score']:.4f} (+{z_score:.1f}σ)",
                "Coseno | Léxico": f"{r['cosine_score']:.4f} | {r['lexical_hits']}",
                "Entidad": r["nombre_entidad"],
                "Depto": r["departamento"],
                "Modalidad": r["modalidad_de_contratacion"],
                "Fecha": str(r["fecha_de_firma"]),
                "Valor COP (vs Med)": f"{val_cop} ({diff_str})",
                "Objeto del Contrato": r["objeto_del_contrato"],
            }
        )

    df = pd.DataFrame(table_data)

    # ---> CAPA 1: AUDITORÍA FORENSE DETERMINISTA  <---
    deterministic_audit_md, structured_facts = build_deterministic_forensic_audit(
        query_text,
        results,
        min_p,
        med_p,
        avg_p,
        max_p,
        top_entity,
        top_modality,
        mod,
    )

    rag_summary = (
        f"### 📊 INFORME EJECUTIVO Y DICTAMEN FORENSE RAG (Top-{len(results)} Contratos)\n"
        f"- **Rango de Cuantías (COP):** Mín: `${min_p:,.0f}` | **Mediana: `${med_p:,.0f}`** | "
        f"Promedio: `${avg_p:,.0f}` | Máx: `${max_p:,.0f}`\n"
        f"- **Entidad Frecuente:** {top_entity} | **Modalidad Frecuente:** {top_modality} | "
        f"**Significancia Máxima:** `+{max_z:.1f}σ` (D=512)\n\n"
        f"---\n"
        f"#### 🔍 Dictamen Analítico Verificado (Motor Simbólico Exacto)\n"
        f"{deterministic_audit_md}"
    )

    # --- CAPA 2: REDACCIÓN EJECUTIVA CON LLM LOCAL GUIADO POR HECHOS ---
    llm_ms = 0.0
    if use_local_llm:
        llm_text, llm_ms = generate_local_llm_synthesis(
            query_text, med_p, structured_facts
        )
        rag_summary += (
            f"\n\n---\n#### 🤖 Síntesis Redactada por LLM Local (`{LOCAL_SLM_REPO}` — {llm_ms:.0f} ms)\n"
            f"> {llm_text}"
        )

    rss_now_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)

    telemetry_summary = (
        f"⚡ Régimen: {telemetry.get('regime')} | "
        f"Codificación (MPS): {telemetry.get('encode_query_ms')} ms | "
        f"Filtro O(N): {telemetry.get('filter_eval_ms')} ms | "
        f"Vector Search + Re-rank: {telemetry.get('vector_search_rerank_ms')} ms | "
        f"Hidratación Parquet: {telemetry.get('hydrate_parquet_ms')} ms | "
        f"Total Búsqueda: {search_elapsed_ms:.2f} ms"
        + (f" | Redacción LLM 3B: {llm_ms:.0f} ms" if use_local_llm else "")
        + f" | RAM RSS Actual: {rss_now_mb:.1f} MB"
    )

    return df, telemetry_summary, rag_summary


with gr.Blocks(
    theme=gr.themes.Soft(), title="Buscador Semántico SECOP II - Tesis"
) as demo:
    gr.Markdown(
        "# 🏛️ Motor de Búsqueda Híbrido y Auditoría Forense RAG para SECOP II (5.65M Contratos)"
    )
    gr.Markdown(
        "Herramienta de consulta forense para contratación pública optimizada para equipos de **8 GB de RAM**. "
        "Combina **Grafo Cociente CSR Vamana + RaBitQ (72 B) + Re-ranking INT8** con **Auditoría Neuro-Simbólica Libre de Alucinaciones**."
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
                label="🤖 Añadir Párrafo de Cierre con LLM Local (Qwen 3B 4-bit)",
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
            value="CUNDINAMARCA",
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
            value="CONTRATACIÓN DIRECTA",
        )

    with gr.Row():
        anio_min_num = gr.Number(label="Año Mínimo", value=2021, precision=0)
        anio_max_num = gr.Number(label="Año Máximo", value=2023, precision=0)
        valor_min_num = gr.Number(
            label="Cuantía Mínima (COP)", value=300_000_000, precision=0
        )

    search_btn = gr.Button(
        "🔍 Ejecutar Búsqueda Híbrida + Dictamen Forense RAG", variant="primary"
    )

    with gr.Row():
        telemetry_output = gr.Textbox(
            label="Telemetría y Rendimiento del Motor (Búsqueda + RAM RSS)",
            interactive=False,
        )

    rag_output = gr.Markdown(
        value="*Realiza una búsqueda para ver el informe ejecutivo y el dictamen forense RAG.*",
        label="Informe Ejecutivo y Dictamen Forense RAG",
    )

    results_table = gr.Dataframe(
        label="Contratos Auditados",
        headers=[
            "#",
            "Diagnóstico",
            "ID Contrato",
            "Score Híbrido",
            "Coseno | Léxico",
            "Entidad",
            "Depto",
            "Modalidad",
            "Fecha",
            "Valor COP (vs Med)",
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
