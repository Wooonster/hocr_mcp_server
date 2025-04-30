import logging
import io
import base64
import re
from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI, File, UploadFile, Form
from openai import OpenAI, api_key
from PIL import Image

# set up api_key, api_base
# openrouter.ai
api_base = "https://openrouter.ai/api/v1"
api_key = "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"

vision_model_name = "qwen/qwen2.5-vl-32b-instruct:free"

# set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_fomatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_fomatter)
logger.addHandler(console_handler)

# create a file handler
file_handler = logging.FileHandler('mcp_server.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# set up OpenAI API
client = OpenAI(base_url=api_base, api_key=api_key)

# set up FastMCPServer
mcp = FastMCP()
# mcp_app = mcp.create_app()
# mcp_app = mcp.app

# set up fastapi
app = FastAPI()
app.mount("/mcp", mcp.sse_app())

def extract_answer(text):
    # extract matching patterns
    patterns = [
        (r"<latex>(.*?)</latex>", re.DOTALL),
        (r"<start_latex>(.*?)</end_latex>", re.DOTALL),
        (r"<start_latex>(.*?)<end_latex>", re.DOTALL),
        (r"\\\[(.*?)\\\]", re.DOTALL),
        (r"\\begin\{[^}]*\}(.*?)\\end\{[^}]*\}", re.DOTALL),
        (r'```latex\s*\n?(.*?)\n?```', re.DOTALL),
    ]
    
    for pattern, flags in patterns:
        match = re.search(pattern, text, flags)
        if match:
            extracted = match.group(1).strip().replace(' ', '')
            return extracted

    return None

# mcp tool 
@mcp.tool()
def extract_latex(image_base64, prompt=None):
    logger.info("Received image for extraction, with prompt: %s", prompt if prompt else "None")
    
    # image process
    image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
    image = image.convert("RGB")

    # re-formate to png
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    png_b64 = base64.b64encode(buffer.getvalue()).decode()
    data_uri = f"data:image/png;base64,{png_b64}"

    messages=[
        { "role": "system", "content": "You are a helpful assistant that helps user to extract latex from given images." },
        {
            "role": "user",
            "content": [
                { "type": "text", "text": "Please extract the latex of the mathematical formulas in the image. " + prompt if prompt else "" },
                { "type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
            ],
        },
    ]

    # logger.info("Sending messages to VLM: %s", messages)

    response = client.chat.completions.create(
        model=vision_model_name,
        messages=messages,
    )

    answer = response.choices[0].message.content
    logger.info("VLM answer: %s", answer)

    # post-process the answer
    answer = extract_answer(answer)

    return answer


# uplaod image
@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    prompt: str = Form(default="Please extract the latex of the mathematical formulas in the image.")
):
    if file.content_type not in ["image/jpeg", "image/png"]:
        return {"Image Format Error": "Only JPEG and PNG formats are supported."}
    
    data = await file.read()
    bs64 = base64.b64encode(data).decode()
    args = {
        "image_base64": bs64,
        "prompt": prompt
    }

    # call mcp tool
    try:
        result = await mcp.call_tool("extract_latex", args)
        logger.info("Extracted latex: %s", result)
        # get the latex string
        if isinstance(result, list) and result and hasattr(result[0], "text"):
            latex_str = result[0].text
        else:
            latex_str = str(result)
        # return to the frontend
        return {"latex": latex_str}
    except Exception as e:
        logger.error("Error in extraction: %s", str(e))
        return {"error": "Extraction failed."}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)