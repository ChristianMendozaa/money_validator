import os
import base64
import json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Money Validator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://tucaserito.com", "https://www.tucaserito.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hardcoded ranges mapped by denomination to list of tuples (start, end)
BILL_RANGES = {
    10: [
        (77100001, 77550000),
        (78000001, 78450000),
        (78900001, 96350000),
        (96350001, 96800000),
        (96800001, 97250000),
        (98150001, 98600000),
        (104900001, 105350000),
        (105350001, 105800000),
        (106700001, 107150000),
        (107600001, 108050000),
        (108050001, 108500000),
        (109400001, 109850000),
    ],
    20: [
        (87280145, 91646549),
        (96650001, 97100000),
        (99800001, 100250000),
        (100250001, 100700000),
        (109250001, 109700000),
        (110600001, 111050000),
        (111050001, 111500000),
        (111950001, 112400000),
        (112400001, 112850000),
        (112850001, 113300000),
        (114200001, 114650000),
        (114650001, 115100000),
        (115100001, 115550000),
        (118700001, 119150000),
        (119150001, 119600000),
        (120500001, 120950000),
    ],
    50: [
        (67250001, 67700000),
        (69050001, 69500000),
        (69500001, 69950000),
        (69950001, 70400000),
        (70400001, 70850000),
        (70850001, 71300000),
        (76310012, 85139995),
        (86400001, 86850000),
        (90900001, 91350000),
        (91800001, 92250000),
    ]
}

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tenacity retry logic for RateLimitError
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True
)
async def call_openai_with_retry(messages):
    return await client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0
    )

def calculate_and_print_cost(usage):
    if not usage:
        return
    
    # Prompt tokens (breakdown if available)
    prompt_tokens = usage.prompt_tokens
    cached_tokens = 0
    if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
        cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
    
    uncached_input_tokens = prompt_tokens - cached_tokens
    output_tokens = usage.completion_tokens

    # Pricing per million tokens
    cost_input_per_m = 0.10
    cost_cached_per_m = 0.025
    cost_output_per_m = 0.40

    cost_input = (uncached_input_tokens / 1_000_000.0) * cost_input_per_m
    cost_cached = (cached_tokens / 1_000_000.0) * cost_cached_per_m
    cost_output = (output_tokens / 1_000_000.0) * cost_output_per_m

    total_cost = cost_input + cost_cached + cost_output

    print("--- OpenAI API Usage & Cost ---")
    print(f"Tokens - Input: {uncached_input_tokens}, Cached: {cached_tokens}, Output: {output_tokens}")
    print(f"Cost - Input: ${cost_input:.6f}, Cached: ${cost_cached:.6f}, Output: ${cost_output:.6f}")
    print(f"Total Cost: ${total_cost:.6f}")
    print("-------------------------------")

@app.post("/validate")
async def validate_bill(file: UploadFile = File(...)):
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode("utf-8")
    mime_type = file.content_type or "image/jpeg"

    system_prompt = (
        "Eres un validador experto de billetes bolivianos. "
        "Tu tarea es extraer el número de serie completo (dígitos numéricos y la letra adyacente) y la denominación (corte) del billete. "
        "Si el texto del corte no es legible, deduce la denominación por el color predominante del billete: 10 Bs es azul, 20 Bs es naranja y 50 Bs es violeta/morado. "
        "Los cortes de 100 y 200 Bs deben considerarse inválidos, así como cualquier billete que no sea boliviano o sea ilegible.\n"
        "El output debe ser UNICAMENTE un JSON minificado estricto.\n"
        "Si logras extraer el número de serie y el corte válido (10, 20, o 50), devuelve: {\"s\": \"<serie>\", \"c\": <corte>}\n"
        "Ejemplo: {\"s\": \"12345678B\", \"c\": 10}\n"
        "Si es ilegible o de un corte no admitido (ej. 100, 200, falso, extranjero o irreconocible), devuelve exactamente: {\"e\": 1}"
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analiza este billete boliviano."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    try:
        response = await call_openai_with_retry(messages)
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail="Rate limit exceeded after retries.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with OpenAI: {str(e)}")

    if response.usage:
        calculate_and_print_cost(response.usage)

    response_content = response.choices[0].message.content
    try:
        data = json.loads(response_content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON format from OpenAI")

    if data.get("e") == 1:
        raise HTTPException(status_code=400, detail="billete no procesable")
    
    serial = data.get("s", "")
    corte = data.get("c")

    if not serial or corte not in BILL_RANGES:
         raise HTTPException(status_code=400, detail="billete no procesable")

    # Parse letter and digits from serial
    letter = ""
    digits_str = ""
    for char in serial:
        if char.isalpha():
            letter += char.upper()
        elif char.isdigit():
            digits_str += char
            
    if letter != "B":
        return {"status": "valido", "serie": serial}
        
    try:
        numeric_val = int(digits_str)
    except ValueError:
        # Cannot parse numeric part, return normally as "valido"
        return {"status": "valido", "serie": serial}

    ranges = BILL_RANGES.get(corte, [])
    for (start, end) in ranges:
        if start <= numeric_val <= end:
            return {"status": "siniestrado", "serie": serial}
            
    return {"status": "valido", "serie": serial}
