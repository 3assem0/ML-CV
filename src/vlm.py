import requests # type: ignore
import base64
import cv2 # type: ignore
import numpy as np # type: ignore

def translate_to_arabic(text):
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {"client": "gtx", "sl": "en", "tl": "ar", "dt": "t", "q": text}
        res = requests.get(url, params=params, timeout=5)
        if res.status_code == 200:
            data = res.json()
            return "".join([x[0] for x in data[0] if x[0]])
    except Exception as e:
        return ""
    return ""

def analyze_image_with_vlm(image_bgr, api_key, model="nvidia/nemotron-nano-12b-v2-vl:free"):
    """
    Takes a BGR image (numpy array) from OpenCV, encodes it to base64,
    and sends it to OpenRouter.
    """
    if not api_key:
        return "No API key provided."
    api_key = api_key.strip()
    # Resize image to max 512px to significantly speed up API request and inference
    max_dim = 512
    h, w = image_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))

    # Convert BGR to JPEG with moderate compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    success, buffer = cv2.imencode('.jpg', image_bgr, encode_param)
    if not success:
        return "Failed to encode image."
        
    # Convert to base64 string
    base64_image = base64.b64encode(buffer).decode('utf-8')
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://archaeology-sieve.com", # Required by OpenRouter
        "X-Title": "Archaeology Sieve"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Provide a specific, simple, and short description of the objects in this image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                eng_text = data['choices'][0]['message']['content']
                ar_text = translate_to_arabic(eng_text)
                if ar_text:
                    return f"{eng_text}\n\n**عربي:**\n{ar_text}"
                return eng_text
            return "No response text in API."
        else:
            return f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Request failed: {str(e)}"
