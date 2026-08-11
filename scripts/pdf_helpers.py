"""PDF primitives shared by the builders.

Lifted verbatim from generatePDF.py so the llm.pdf builder and the diffusion
builder cannot drift apart. The additions are backward compatible: every new
argument defaults to the old behaviour, and the byte-for-byte output of
generatePDF.py is unchanged (tools/test_builder_identical.sh proves it).
"""

from pdfrw import PdfWriter  # noqa: F401  (re-exported for the builders)
from pdfrw.objects.pdfarray import PdfArray
from pdfrw.objects.pdfdict import PdfDict
from pdfrw.objects.pdfname import PdfName
from pdfrw.objects.pdfstring import PdfString


def create_script(js):
    action = PdfDict()
    action.S = PdfName.JavaScript
    action.JS = js
    return action


def create_page(width, height):
    page = PdfDict()
    page.Type = PdfName.Page
    page.MediaBox = PdfArray([0, 0, width, height])

    page.Resources = PdfDict()
    page.Resources.Font = PdfDict()
    page.Resources.Font.F1 = PdfDict()
    page.Resources.Font.F1.Type = PdfName.Font
    page.Resources.Font.F1.Subtype = PdfName.Type1
    page.Resources.Font.F1.BaseFont = PdfName.Courier

    return page


def create_field(name, x, y, width, height, value="", f_type=PdfName.Tx,
                 flags=None, read_only=False, da=None):
    annotation = PdfDict()
    annotation.Type = PdfName.Annot
    annotation.Subtype = PdfName.Widget
    annotation.FT = f_type
    annotation.Ff = 2
    annotation.Rect = PdfArray([x, y, x + width, y + height])
    annotation.T = PdfString.encode(name)
    annotation.V = PdfString.encode(value)

    annotation.BS = PdfDict()
    annotation.BS.W = 0

    if read_only:
        annotation.Ff = annotation.Ff | 1
    if flags is not None:
        annotation.F = flags
    if da is not None:
        # Default appearance: font, size and colour for the field's text.
        # Without one the viewer picks its own, which is fine for a console but
        # useless when the text has to line up on a grid.
        annotation.DA = PdfString.encode(da)

    appearance = PdfDict()
    appearance.Type = PdfName.XObject
    appearance.SubType = PdfName.Form
    appearance.FormType = 1
    appearance.BBox = PdfArray([0, 0, width, height])
    appearance.Matrix = PdfArray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    return annotation


def create_text(x, y, size, txt):
    return f"""
  BT
  /F1 {size} Tf
  {x} {y} Td ({txt}) Tj
  ET
  """


def create_button(name, x, y, width, height, value, background=None,
                  highlight=None, flags=None, read_only=False):
    """A pushbutton.

    `background` sets /MK/BG, which for a pushbutton *is* its whole appearance
    -- that is what field.fillColor writes to at runtime, and what makes the
    widget show something before any script has run.

    `highlight="N"` sets /H /N, no highlight. Without it a pushbutton inverts
    while the mouse is down, which flashes a rectangle across the picture when
    the grid is made of buttons.
    """
    button = create_field(name, x, y, width, height, f_type=PdfName.Btn,
                          flags=flags, read_only=read_only)
    button.AA = PdfDict()
    button.Ff = 65536 | (1 if read_only else 0)
    button.MK = PdfDict()
    button.MK.BG = PdfArray(background if background is not None else [0.90])
    if value is not None:
        button.MK.CA = value
    if highlight is not None:
        button.H = PdfName(highlight)
    return button


def create_action_buttons(buttons_info):
    """
    Create buttons that execute a single JavaScript function when clicked.

    Parameters:
    buttons_info -- List of dictionaries containing button information:
                   {
                       "name": "button_name",
                       "x": x_position,
                       "y": y_position,
                       "width": button_width,
                       "height": button_height,
                       "label": "Button Label",
                       "js_function": "functionName()"
                   }

    Returns:
    List of button annotations
    """
    buttons = []
    for info in buttons_info:
        name = info["name"]
        button = create_button(
            name,
            info["x"],
            info["y"],
            info["width"],
            info["height"],
            info.get("label", name),
        )
        button.AA = PdfDict()
        button.AA.U = create_script(info["js_function"])
        buttons.append(button)
    return buttons


def attach_acroform(writer, fields):
    """Register the widgets in the document catalogue.

    llm.pdf omits this: its widgets live only in page.Annots, which PDFium
    resolves but Acrobat generally will not, so getField() returns null there.
    Adding /AcroForm costs one dictionary and is what keeps the Acrobat path
    reachable.
    """
    # Each widget must become an indirect object first. Without this pdfrw
    # inlines the whole dictionary into both page.Annots and AcroForm.Fields,
    # so the file carries two independent copies of every field -- twice the
    # bytes, and two widgets claiming the same name, which is exactly the kind
    # of thing a viewer resolves in an unhelpful way.
    for f in fields:
        f.indirect = True

    form = PdfDict()
    form.Fields = PdfArray(fields)
    form.NeedAppearances = PdfName("true")

    # Resources for any /DA a field declares. A /DA naming /F1 is meaningless
    # unless /F1 resolves here, and the viewer then silently falls back to its
    # own font and size -- which shows up as text that will not line up.
    font = PdfDict()
    font.Type = PdfName.Font
    font.Subtype = PdfName.Type1
    font.BaseFont = PdfName.Courier
    form.DR = PdfDict()
    form.DR.Font = PdfDict()
    form.DR.Font.F1 = font
    form.DA = PdfString.encode("/F1 0 Tf 0 g")

    writer.trailer.Root.AcroForm = form
    return form


def render_template(text, replacements):
    """Placeholder substitution, with the failure modes made loud.

    The naive str.replace chain in generatePDF.py fails silently on a typo'd
    placeholder, and pdfrw will quietly re-encode the whole payload as UTF-16
    hex -- doubling the file -- if a single non-ASCII character sneaks into a
    comment. Both are checked here.
    """
    import re

    # Longest first, so __GRID__ cannot clobber part of __GRID_CELL__, and the
    # big base64 blob is substituted last to avoid rescanning it every pass.
    for key in sorted(replacements, key=lambda k: (-len(k), k)):
        text = text.replace(key, str(replacements[key]))

    leftover = sorted(set(re.findall(r"__[A-Z0-9_]+__", text)))
    if leftover:
        raise RuntimeError(f"unsubstituted placeholders: {', '.join(leftover)}")

    try:
        text.encode("ascii")
    except UnicodeEncodeError as e:
        bad = text[e.start:e.end]
        raise RuntimeError(
            f"template contains non-ASCII ({bad!r} at {e.start}). pdfrw would "
            "encode the payload as UTF-16 hex and double the file size."
        ) from e

    if "\r" in text or "\0" in text:
        raise RuntimeError("template contains a carriage return or NUL")

    return text
