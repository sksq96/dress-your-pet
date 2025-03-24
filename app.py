import os, io
import requests
from google import genai
from PIL import Image as PILImage
from google.genai.types import GenerateContentConfig 
import base64
import modal
from pydantic import BaseModel
from typing import Optional

class ImageGenerationRequest(BaseModel):
    prompt: str
    image: str
    prompt_template: Optional[str] = None

image = modal.Image.debian_slim().pip_install("fastapi[standard]", "google-genai", "pillow", "requests")
app = modal.App(name="gemini-image-generation", image=image)

# Create client
client = genai.Client(api_key='API_KEY')
config = GenerateContentConfig(response_modalities=['Text', 'Image'])

# Define Gemini model id 
model_id = "gemini-2.0-flash-exp-image-generation"

@app.function()
@modal.fastapi_endpoint(method="POST", docs=True)
def generate_image(request: ImageGenerationRequest):
    try:
        prompt = request.prompt
        if request.prompt_template:
            prompt_template = request.prompt_template
            print(f"Prompt template: {prompt_template}")
            url = f"https://raw.githubusercontent.com/sksq96/dress-your-pet/refs/heads/main/{prompt_template}.md"
            prompt = requests.get(url).text

        image_base64 = request.image
        
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = PILImage.open(io.BytesIO(image_data))
        
        # Generate the image
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt, image],
            config=config
        )

        result = {
            'text': None,
            'image': None
        }

        # Process response
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                # Convert generated image to base64
                img_bytes = part.inline_data.data
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                result['image'] = img_base64
            else:
                result['text'] = part.text

        return result

    except Exception as e:
        return {'error': str(e)}

