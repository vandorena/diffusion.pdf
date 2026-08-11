import base64
import argparse

from pdfrw.objects.pdfdict import PdfDict
from pdfrw.objects.pdfarray import PdfArray
import os
import sys

# Works whether this is run as `python3 scripts/generatePDF.py` from the repo
# root or as `python3 ./generatePDF.py` from inside scripts/, which is what
# build-pdfs.sh and the README both do.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdf_helpers import (  # noqa: E402
    PdfWriter,
    create_action_buttons,
    create_button,  # noqa: F401  (kept for anyone importing this module)
    create_field,
    create_page,
    create_script,
    create_text,
)


def process_template(template_path, llama_path, gguf_path, console_line_count):
    print("Constructing JS code")
    print("Processing template")
    # Read the template file
    with open(template_path, "r") as file:
        template_content = file.read()

    print("Processing llama.cpp code")
    # Read the llama module code
    with open(llama_path, "r") as file:
        llama_code = file.read()

    print("Processing model GGUF")
    # Read the GGUF file and encode to base64
    with open(gguf_path, "rb") as file:
        gguf_content = file.read()
    gguf_base64 = base64.b64encode(gguf_content).decode("utf-8")

    # Replace placeholders
    processed_template = template_content
    processed_template = processed_template.replace("// __MODULE_CODE__", llama_code)
    file_name = os.path.basename(gguf_path)
    processed_template = processed_template.replace("__FILE_NAME__", file_name)
    processed_template = processed_template.replace(
        "__CONSOLE_LINE_COUNT__", str(console_line_count)
    )
    processed_template = processed_template.replace("__GGUF_FILE__", gguf_base64)

    print("Finished JS code processing")

    return processed_template


if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Generate a PDF with embedded LLM capabilities"
    )

    parser.add_argument(
        "--template",
        type=str,
        default=os.path.join(file_dir, "../src/template.js"),
        help="Path to the JavaScript template file",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the GGUF model file",
    )

    parser.add_argument(
        "--llama",
        type=str,
        default=os.path.join(file_dir, "../llama/llama.js"),
        help="Path to the llama.js file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(file_dir, "../builds/llm.pdf"),
        help="Path where the output PDF will be saved",
    )

    args = parser.parse_args()

    TEMPLATE_PATH = args.template
    GGUF_PATH = args.model
    LLAMA_PATH = args.llama
    OUTPUT_PATH = args.output

    print(f"Using template: {TEMPLATE_PATH}")
    print(f"Using model: {GGUF_PATH}")
    print(f"Using llama.js: {LLAMA_PATH}")
    print(f"Output path: {OUTPUT_PATH}")

    # TEMPLATE_PATH = "../src/template.js"
    # # GGUF_PATH = "../models/tinystories-3m-q8.gguf"
    # GGUF_PATH = "../models/smollm2-135M-it-q1.gguf"
    # GGUF_PATH = "../models/smollm2-135M-it-q8.gguf"
    # GGUF_PATH = "../models/pythia-31m-chat-q8.gguf"
    # GGUF_PATH = "../models/tinystories-3m-q8.gguf"
    # GGUF_PATH = "../models/tinyllm-q8.gguf"
    # # GGUF_PATH = "../models/pythia-14m-q2.gguf"
    # LLAMA_PATH = "../llama/llama.js"

    # OUTPUT_PATH = "../llm.pdf"

    width = 700
    height = 700

    # Define title area

    # Define field size
    field_width = width
    field_height = 12

    writer = PdfWriter()
    page = create_page(width, height)
    page.Contents = PdfDict()

    fields = []

    # Calculate how many console fields we need
    # Leave space for title, prompt area, and info text
    prompt_area_height = 42
    info_text_height = 20
    console_area_height = height - prompt_area_height - info_text_height - 5
    rows = console_area_height // field_height

    for i in range(rows):
        y = height - (i + 1) * field_height
        field = create_field(
            f"console_{i}", 5, y - 5, field_width - 10, field_height, ""
        )
        fields.append(field)

    # Add prompt input field
    prompt_input_width = 315
    prompt_input_height = 25
    prompt_input_x = 100
    main_ui_height = prompt_area_height - 15
    prompt_input = create_field(
        "promptInput",
        prompt_input_x,
        main_ui_height,
        prompt_input_width,
        prompt_input_height,
        "",
    )
    fields.append(prompt_input)

    token_input_width = 50
    token_input_height = 25
    token_input_x = prompt_input_x + prompt_input_width + 5
    token_input = create_field(
        "tokenCount",
        token_input_x,
        main_ui_height,
        token_input_width,
        token_input_height,
        "10",
    )
    fields.append(token_input)

    context_input_width = 50
    context_input_height = 25
    context_input_x = token_input_x + token_input_width + 5
    context_input = create_field(
        "context",
        context_input_x,
        main_ui_height,
        context_input_width,
        context_input_height,
        "15",
    )
    fields.append(context_input)

    # Add "Prompt:" label
    page.Contents.stream = create_text(
        prompt_input_x, prompt_area_height + 14, 9, "Prompt:"
    )
    page.Contents.stream += create_text(5, prompt_area_height + 14, 9, "v1.0")
    page.Contents.stream += create_text(
        token_input_x, prompt_area_height + 14, 9, "Tok Out:"
    )
    page.Contents.stream += create_text(
        context_input_x, prompt_area_height + 14, 9, "Ctx Len:"
    )

    # Add buttons
    button_width = 80
    button_height = 25

    ask_button_x = context_input_x + token_input_width + 5
    complete_button_x = ask_button_x + button_width + 5

    # Create title text
    page.Contents.stream += create_text(5, main_ui_height + 5, 20, "llm.pdf")

    # Create the buttons
    action_buttons = create_action_buttons(
        [
            {
                "name": "askButton",
                "x": ask_button_x,
                "y": main_ui_height,
                "width": button_width,
                "height": button_height,
                "label": "Ask",
                "js_function": "runLlamaAsk()",
            },
            {
                "name": "completeButton",
                "x": complete_button_x,
                "y": main_ui_height,
                "width": button_width,
                "height": button_height,
                "label": "Complete",
                "js_function": "runLlamaCompletion()",
            },
        ]
    )
    fields.extend(action_buttons)

    # Add information text at the bottom
    page.Contents.stream += create_text(
        75,
        10,
        8,
        "This PDF only works in Chromium-based browsers. Visit https://github.com/EvanZhouDev/llm-pdf for more information.",
    )

    page.AA = PdfDict()
    page.AA.O = create_script(
        process_template(TEMPLATE_PATH, LLAMA_PATH, GGUF_PATH, rows)
    )

    page.Annots = PdfArray(fields)
    writer.addpage(page)
    writer.write(OUTPUT_PATH)
