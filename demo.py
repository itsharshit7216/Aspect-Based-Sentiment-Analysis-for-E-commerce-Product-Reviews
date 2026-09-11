import os
import csv
import datetime
import gradio as gr
from src.absa_engine import ABSAEngine

# Initialize ABSA Engine
engine = ABSAEngine()

# Ensure flagged directory exists
FLAG_DIR = "flagged"
FLAG_FILE = os.path.join(FLAG_DIR, "log.csv")
os.makedirs(FLAG_DIR, exist_ok=True)

def flag_data(review_text: str, results_json: dict):
    """Save flagged review and prediction to CSV log."""
    if not review_text or not review_text.strip():
        return gr.update(value="⚠️ Nothing to flag. Please analyze a review first.", visible=True)
    
    file_exists = os.path.isfile(FLAG_FILE)
    with open(FLAG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "input_review", "model_prediction"])
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            review_text.strip(),
            str(results_json)
        ])
    return gr.update(value="✅ Prediction flagged and recorded in flagged/log.csv for model review.", visible=True)

def update_counter(text: str):
    """Compute and format live character count out of 500."""
    count = len(text) if text else 0
    color = "#ef4444" if count > 500 else "#94a3b8"
    return f"<div style='text-align: right; font-size: 12px; color: {color}; font-weight: 500; margin-top: -6px; margin-bottom: 8px;'>{count}/500</div>"

def analyze_review(sentence: str):
    """Run ABSA engine and generate bright HTML visual cards + JSON."""
    if not sentence or not sentence.strip():
        empty_html = """
        <div style='padding: 20px; text-align: center; color: #64748b; background: #ffffff; border-radius: 10px; border: 1px dashed #cbd5e1;'>
            Please enter a product review above and click <b>Analyze Review</b>.
        </div>
        """
        return empty_html, {"error": "Please enter a valid review sentence."}

    data = engine.analyze_sentence(sentence)
    analysis = data.get("analysis", {})
    count = data.get("extracted_aspects_count", 0)

    if not analysis:
        no_aspects_html = """
        <div style='padding: 20px; text-align: center; color: #64748b; background: #ffffff; border-radius: 10px; border: 1px solid #e2e8f0;'>
            No specific aspect terms were detected in this sentence.
        </div>
        """
        return no_aspects_html, data

    cards_html = "<div style='display: flex; flex-wrap: wrap; gap: 14px; margin-top: 12px;'>"
    for aspect, sentiment in analysis.items():
        sent_lower = sentiment.lower()
        if "pos" in sent_lower:
            bg_color = "#ecfdf5"
            border_color = "#a7f3d0"
            text_color = "#065f46"
            badge_bg = "#10b981"
            badge_text = "#ffffff"
        elif "neg" in sent_lower:
            bg_color = "#fef2f2"
            border_color = "#fecaca"
            text_color = "#991b1b"
            badge_bg = "#ef4444"
            badge_text = "#ffffff"
        else:
            bg_color = "#f8fafc"
            border_color = "#e2e8f0"
            text_color = "#334155"
            badge_bg = "#64748b"
            badge_text = "#ffffff"

        cards_html += f"""
        <div style='background: {bg_color}; border: 1px solid {border_color}; border-radius: 12px; padding: 14px 20px; min-width: 180px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);'>
            <div style='font-size: 11px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.06em; color: #64748b; margin-bottom: 4px;'>Aspect</div>
            <div style='font-size: 17px; font-weight: 700; color: {text_color}; text-transform: capitalize; margin-bottom: 10px;'>{aspect}</div>
            <span style='background: {badge_bg}; color: {badge_text}; padding: 4px 12px; border-radius: 14px; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em;'>{sentiment}</span>
        </div>
        """
    cards_html += "</div>"

    summary_banner = f"""
    <div style='margin-bottom: 6px; font-size: 15px; font-weight: 700; color: #0f172a;'>
        Extracted <span style='color: #2563eb;'>{count}</span> Aspect{'s' if count != 1 else ''}:
    </div>
    """ + cards_html

    return summary_banner, data

# Strict Bright / Light Modern Design CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Force Light Theme Globally - Overrides System / Dark Preferences */
:root, html, body, .gradio-container, gradio-app, .dark, [data-theme="dark"], body.dark, html.dark, dialog, .modal, [role="dialog"] {
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
    --block-background-fill: #ffffff !important;
    --body-background-fill: #f8fafc !important;
    --body-text-color: #0f172a !important;
    --block-label-text-color: #0f172a !important;
    --block-title-text-color: #0f172a !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: #cbd5e1 !important;
    --border-color-primary: #e2e8f0 !important;
    --border-color-secondary: #e2e8f0 !important;
    --button-secondary-background-fill: #eff6ff !important;
    --button-secondary-text-color: #1d4ed8 !important;
    --button-secondary-border-color: #bfdbfe !important;
    --neutral-50: #f8fafc !important;
    --neutral-100: #f1f5f9 !important;
    --neutral-200: #e2e8f0 !important;
    --neutral-300: #cbd5e1 !important;
    --neutral-400: #94a3b8 !important;
    --neutral-500: #64748b !important;
    --neutral-600: #475569 !important;
    --neutral-700: #334155 !important;
    --neutral-800: #1e293b !important;
    --neutral-900: #0f172a !important;
    --neutral-950: #020617 !important;
    background-color: #f8fafc !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: #0f172a !important;
    color-scheme: light !important;
}

@media (prefers-color-scheme: dark) {
    :root, html, body, .gradio-container, gradio-app, .dark, [data-theme="dark"], body.dark, html.dark {
        --background-fill-primary: #ffffff !important;
        --background-fill-secondary: #f8fafc !important;
        --block-background-fill: #ffffff !important;
        --body-background-fill: #f8fafc !important;
        --body-text-color: #0f172a !important;
        --block-label-text-color: #0f172a !important;
        --block-title-text-color: #0f172a !important;
        --input-background-fill: #ffffff !important;
        --input-border-color: #cbd5e1 !important;
        --border-color-primary: #e2e8f0 !important;
        --border-color-secondary: #e2e8f0 !important;
        background-color: #f8fafc !important;
        color: #0f172a !important;
        color-scheme: light !important;
    }
}

.container {
    max-width: 980px;
    margin: 0 auto;
    padding: 20px 20px;
}

/* Header layout ensuring top-right alignment */
.header-box {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 24px;
    gap: 20px;
}

.title-area {
    flex: 1;
}

.title-area h1 {
    font-size: 30px !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.02em;
}

.title-area p {
    font-size: 15px !important;
    color: #475569 !important;
    margin: 0 0 10px 0 !important;
}

.accent-line {
    width: 46px;
    height: 3.5px;
    background: #2563eb;
    border-radius: 2px;
}

/* Top Right Callout Badge */
.callout-card {
    background: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 12px;
    padding: 10px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-left: auto;
    flex-shrink: 0;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}

.callout-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.callout-text {
    font-size: 13.5px !important;
    color: #1e40af !important;
    font-weight: 500 !important;
    line-height: 1.35;
    max-width: 210px;
}

/* Force Light Cards */
.dark .input-card, .dark .results-card, .input-card, .results-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 24px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04) !important;
    margin-bottom: 20px !important;
}

.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 17px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin-bottom: 4px;
}

.card-subhead {
    font-size: 13.5px !important;
    color: #64748b !important;
    margin-bottom: 16px;
}

/* Textarea in Bright Mode */
.block, .form, fieldset, .wrap, .gradio-textbox {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.dark textarea, textarea {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    font-size: 14.5px !important;
    line-height: 1.6 !important;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.02) !important;
}

textarea::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
}

/* Sample Buttons Row */
.samples-row {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

.sample-tag {
    font-size: 13.5px !important;
    font-weight: 600 !important;
    color: #475569 !important;
}

.sample-btn {
    border-radius: 8px !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
    background: #eff6ff !important;
    color: #1d4ed8 !important;
    border: 1px solid #bfdbfe !important;
    transition: all 0.15s ease-in-out !important;
}

.sample-btn:hover {
    background: #dbeafe !important;
    border-color: #93c5fd !important;
}

/* Analyze Button */
.analyze-btn {
    background: #2563eb !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 14.5px !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.3) !important;
    transition: all 0.15s ease-in-out !important;
}

.analyze-btn:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.4) !important;
}

.flag-btn {
    background: #f8fafc !important;
    color: #475569 !important;
    border: 1px solid #cbd5e1 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.15s ease-in-out !important;
}

.flag-btn:hover {
    background: #f1f5f9 !important;
    color: #0f172a !important;
    border-color: #94a3b8 !important;
}

.flag-row {
    margin-top: 18px;
    display: flex;
    align-items: center;
    gap: 14px;
}
"""

head_html = """
<script>
(function() {
    try {
        localStorage.setItem('gradio-theme', 'light');
        localStorage.setItem('theme', 'light');
        sessionStorage.setItem('gradio-theme', 'light');
    } catch(e) {}
    
    // Lock document to light theme immediately
    document.documentElement.classList.remove('dark');
    document.documentElement.classList.add('light');
    document.documentElement.setAttribute('data-theme', 'light');
    document.documentElement.style.colorScheme = 'light';
    
    // Override matchMedia for dark mode queries so 'System' option evaluates to light
    const originalMatchMedia = window.matchMedia;
    window.matchMedia = function(query) {
        if (query && (query.includes('prefers-color-scheme: dark') || query.includes('prefers-color-scheme:dark'))) {
            return {
                matches: false,
                media: query,
                onchange: null,
                addListener: function() {},
                removeListener: function() {},
                addEventListener: function() {},
                removeEventListener: function() {},
                dispatchEvent: function() { return false; }
            };
        }
        return originalMatchMedia.call(window, query);
    };
})();
</script>
<style>
/* Immediate light theme override */
:root, html, body, gradio-app, .gradio-container, .dark, [data-theme="dark"], body.dark, html.dark {
    color-scheme: light !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
}
</style>
"""

force_light_js = """
() => {
    try {
        localStorage.setItem('gradio-theme', 'light');
        localStorage.setItem('theme', 'light');
        sessionStorage.setItem('gradio-theme', 'light');
    } catch(e) {}

    const enforceLight = () => {
        document.documentElement.classList.remove('dark');
        document.body.classList.remove('dark');
        document.documentElement.classList.add('light');
        document.body.classList.add('light');
        document.documentElement.setAttribute('data-theme', 'light');
        document.body.setAttribute('data-theme', 'light');
        document.documentElement.style.colorScheme = 'light';
        document.body.style.colorScheme = 'light';
        
        const gradioApp = document.querySelector('gradio-app');
        if (gradioApp) {
            gradioApp.classList.remove('dark');
            gradioApp.classList.add('light');
            gradioApp.setAttribute('data-theme', 'light');
            gradioApp.style.colorScheme = 'light';
            if (gradioApp.shadowRoot) {
                const darkEls = gradioApp.shadowRoot.querySelectorAll('.dark');
                darkEls.forEach(el => el.classList.remove('dark'));
            }
        }
        document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
    };

    enforceLight();
    setInterval(enforceLight, 250);
}
"""

with gr.Blocks(title="Customer Review Intelligence") as demo:
    with gr.Column(elem_classes=["container"]):
        # Top Header Section
        gr.HTML("""
        <div class='header-box'>
            <div class='title-area'>
                <h1>Customer Review Intelligence</h1>
                <p>Aspect-Based Sentiment Analysis for E-commerce Product Reviews</p>
                <div class='accent-line'></div>
            </div>
            <div class='callout-card'>
                <div class='callout-icon'>
                    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <circle cx="12" cy="12" r="6"></circle>
                        <circle cx="12" cy="12" r="2"></circle>
                    </svg>
                </div>
                <div class='callout-text'>Turn customer feedback into meaningful insights</div>
            </div>
        </div>
        """)

        # Main Input Card
        with gr.Column(elem_classes=["input-card"]):
            gr.HTML("""
            <div class='card-header'>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                    <polyline points="9 12 11 14 15 10"></polyline>
                </svg>
                Enter a Product Review
            </div>
            <div class='card-subhead'>Type or paste a customer review below to analyze product aspects and their sentiment.</div>
            """)

            review_input = gr.Textbox(
                lines=4,
                placeholder="Type here...",
                value="",
                show_label=False,
                max_lines=8,
                elem_id="review_text_input"
            )

            char_counter = gr.HTML(
                value="<div style='text-align: right; font-size: 12px; color: #94a3b8; font-weight: 500; margin-top: -6px; margin-bottom: 8px;'>0/500</div>"
            )

            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row(elem_classes=["samples-row"]):
                        gr.HTML("<span class='sample-tag'>Try a sample:</span>")
                        btn_sample_laptop = gr.Button("Laptop Review", size="sm", elem_classes=["sample-btn"])
                        btn_sample_phone = gr.Button("Phone Review", size="sm", elem_classes=["sample-btn"])
                        btn_sample_headphone = gr.Button("Headphone Review", size="sm", elem_classes=["sample-btn"])

                with gr.Column(scale=1, min_width=160):
                    btn_analyze = gr.Button("🔍 Analyze Review", elem_classes=["analyze-btn"], size="lg")

        # Output Results Section
        with gr.Column(elem_classes=["results-card"]):
            gr.HTML("<div class='card-header'>Analysis Results</div>")
            visual_results = gr.HTML(
                value="<div style='padding: 16px; color: #64748b; background: #f8fafc; border-radius: 8px;'>Click <b>Analyze Review</b> above to see aspect extraction and sentiment breakdown.</div>"
            )

            with gr.Accordion("Raw JSON Output", open=False):
                json_output = gr.JSON(label="API Response Format")

            # Flag button section
            with gr.Row(elem_classes=["flag-row"]):
                btn_flag = gr.Button("🚩 Flag Incorrect Prediction", elem_classes=["flag-btn"], size="sm")
                flag_status = gr.Markdown(visible=False)

        # Dynamic counter on typing
        review_input.change(
            fn=update_counter,
            inputs=review_input,
            outputs=char_counter
        )

        # Sample click handlers
        sample_laptop_text = "The laptop performance is excellent but the battery life is terrible. The display is good and the keyboard is comfortable."
        sample_phone_text = "The camera takes stunning pictures in daylight, but the battery drains fast and the speakers are quiet."
        sample_headphone_text = "Sound quality and noise cancellation are top notch, though build quality feels slightly cheap."

        btn_sample_laptop.click(
            fn=lambda: (sample_laptop_text, update_counter(sample_laptop_text)),
            outputs=[review_input, char_counter]
        )
        btn_sample_phone.click(
            fn=lambda: (sample_phone_text, update_counter(sample_phone_text)),
            outputs=[review_input, char_counter]
        )
        btn_sample_headphone.click(
            fn=lambda: (sample_headphone_text, update_counter(sample_headphone_text)),
            outputs=[review_input, char_counter]
        )

        # Analysis click handler
        btn_analyze.click(
            fn=analyze_review,
            inputs=review_input,
            outputs=[visual_results, json_output]
        )

        # Flag button handler
        btn_flag.click(
            fn=flag_data,
            inputs=[review_input, json_output],
            outputs=flag_status
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Launching Customer Review Intelligence UI on port {port}...")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, css=custom_css, js=force_light_js, head=head_html)

