import base64
import logging
import os
import tempfile

import gradio as gr
from dotenv import load_dotenv

from src.db import get_collection
from pipeline import run as run_pipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# load once at startup
logger.info("Loading vector store…")
_collection = get_collection()
logger.info("Vector store ready")

EXAMPLE_QUESTIONS = [
    "What is the ROI of HCP channels for oncology brands and how do they compare to industry benchmarks?",
    "If I cut TV budget by 20% for vaccines and reinvest it in paid search, what happens to revenue?",
    "Which channels are underperforming for Keytruda and what should we do about them?",
    "Show me the spend vs revenue breakdown for oncology brands in 2024.",
    "What are the diminishing returns on Salesforce calls for oncology?",
]


def _insights_md(insights: list) -> str:
    if not insights:
        return "_No channel-level insights returned._"
    rows = ["| Channel | Current ROI | Benchmark ROI | Confidence | Recommendation |",
            "|---------|------------|--------------|------------|----------------|"]
    for ins in insights:
        rows.append(
            f"| **{ins.get('channel', '—').upper()}** "
            f"| {ins.get('current_roi', '—')}x "
            f"| {ins.get('benchmark_roi') or '—'} "
            f"| {ins.get('confidence', '—')} "
            f"| {ins.get('recommendation', '—')} |"
        )
    return "\n".join(rows)


def _b64_to_tempfile(b64: str) -> str:
    data = base64.b64decode(b64)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(data)
    tmp.flush()
    return tmp.name


def analyze(question: str):
    if not question.strip():
        yield (
            gr.update(value="Please enter a question.", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
        return

    # --- streaming status while the LLM thinks ---
    yield (
        gr.update(value="⏳ Running pipeline…", visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )

    try:
        answer = run_pipeline(question, _collection)
    except Exception as e:
        logger.error("Pipeline error: %s", e)
        yield (
            gr.update(value=f"❌ Pipeline error: {e}", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
        return

    summary   = answer.get("summary", "")
    narrative = answer.get("narrative", "")
    sources   = answer.get("sources", [])
    insights  = answer.get("insights", [])
    chart_b64 = answer.get("chart_b64")

    sources_md = f"\n\n---\n**Sources:** {', '.join(sources)}" if sources else ""
    summary_md = f"## Summary\n\n{summary}{sources_md}"
    insights_md = _insights_md(insights)
    narrative_md = f"## Analysis\n\n{narrative}"

    chart_path = _b64_to_tempfile(chart_b64) if chart_b64 else None

    yield (
        gr.update(value=summary_md,   visible=True),
        gr.update(value=insights_md,  visible=True),
        gr.update(value=narrative_md, visible=True),
        gr.update(value=chart_path,   visible=chart_path is not None),
        gr.update(visible=False),
    )


# ─── UI ───────────────────────────────────────────────────────────
with gr.Blocks(title="MMM Copilot") as demo:

    gr.Markdown(
        """
        # MMM Copilot
        ### Pharma Marketing Mix Modeling — AI-powered analysis
        Ask any question about channel ROI, budget scenarios, benchmarks, or spend efficiency.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Question",
                placeholder="e.g. Which HCP channels have the highest ROI for oncology brands?",
                lines=3,
                elem_id="question-box",
            )
            with gr.Row():
                submit_btn = gr.Button("Analyze", variant="primary", scale=2)
                clear_btn  = gr.Button("Clear", variant="secondary", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("**Example questions**")
            for ex in EXAMPLE_QUESTIONS:
                gr.Button(ex, size="sm").click(
                    fn=lambda q=ex: q,
                    outputs=question_box,
                )

    with gr.Column():
        summary_out   = gr.Markdown(visible=False)
        insights_out  = gr.Markdown(visible=False, label="Channel Insights")
        narrative_out = gr.Markdown(visible=False)
        chart_out     = gr.Image(
            label="Generated Chart",
            visible=False,
            elem_id="chart-output",
        )
        status_out = gr.Markdown(visible=False)

    outputs = [summary_out, insights_out, narrative_out, chart_out, status_out]

    submit_btn.click(fn=analyze, inputs=question_box, outputs=outputs)
    question_box.submit(fn=analyze, inputs=question_box, outputs=outputs)
    clear_btn.click(
        fn=lambda: ("", *[gr.update(visible=False)] * 5),
        outputs=[question_box, *outputs],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
            #question-box textarea { font-size: 15px !important; }
            #chart-output img { border-radius: 8px; }
        """,
    )
