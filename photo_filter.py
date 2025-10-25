import requests
import json
import base64
import os
import time
import math

# --- GLOBAL CONFIGURATION ---
API_KEY = os.environ.get("GEMINI_API_KEY", "") 
API_MODEL = 'gemini-2.5-flash-preview-09-2025'
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{API_MODEL}:generateContent?key={API_KEY}"

# Define the collection where photos are stored.
PHOTO_COLLECTION_PATH = "photos_to_be_filtered"

# Thresholds for vector distance comparison (Tweak these values based on your dataset)
EXACT_DUPLICATE_THRESHOLD = 0.001  # Near zero distance for floating-point comparison (delete)
SIMILAR_PHOTO_THRESHOLD = 5.0      # Distance threshold for grouping similar photos (group)


# --- CONFIG FOR STAGE 1: QUALITY CHECK ---
QUALITY_SYSTEM_PROMPT = (
    "You are a professional photo quality analyst. Your task is to objectively evaluate the technical quality of the image. "
    "Check specifically for: 1) High level of motion or focus blur, 2) Main subject's eyes being closed, and 3) Severely bad or distracting lighting/exposure issues (e.g., strong underexposure, overblown highlights, high noise). "
    "Return the result in the required JSON format."
)
QUALITY_USER_QUERY = "Analyze the image quality based on technical flaws (blur, closed eyes, bad lighting). State 'good' if acceptable, or 'bad' if it has severe flaws."

QUALITY_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "quality_status": {"type": "STRING", "description": "Must be 'good' or 'bad'."},
        "reasons": {"type": "STRING", "description": "Brief explanation if 'bad'."}
    },
    "propertyOrdering": ["quality_status", "reasons"]
}


# --- CONFIG FOR STAGE 2: FEATURE VECTOR / SIMILARITY CHECK ---
EMBEDDING_SYSTEM_PROMPT = (
    "You are an expert feature extractor for images. Analyze the visual content and output a dense, "
    "high-dimensional numerical feature vector (embedding) in the required JSON format. "
    "The embedding should numerically represent the image's content for similarity comparison. "
    "The array MUST contain exactly 256 floating-point numbers."
)
EMBEDDING_USER_QUERY = "Generate the feature vector for this photo."
EMBEDDING_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "embedding": {
            "type": "ARRAY",
            "description": "A dense feature vector representing the image.",
            "items": {"type": "NUMBER"}
        }
    },
    "propertyOrdering": ["embedding"]
}

# --- CONFIG FOR STAGE 3: FACIAL TAGGING ---
FACIAL_TAG_SYSTEM_PROMPT = (
    "You are an expert in identifying and summarizing dominant faces in a photo. "
    "Your task is to provide a single, concise string tag that uniquely identifies the primary person(s) in the image for the purpose of grouping photos of the same person. "
    "The tag should be descriptive but simple (e.g., 'man-with-glasses-and-black-shirt', 'woman-with-red-dress', 'two-children-in-park'). "
    "Return the result in the required JSON format."
)
FACIAL_TAG_USER_QUERY = "Generate a single, unique descriptive tag for the dominant faces in this photo."
FACIAL_TAG_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "facial_tag": {"type": "STRING", "description": "A unique, descriptive tag for the person(s) in the photo."}
    },
    "propertyOrdering": ["facial_tag"]
}

# --- UTILITY FUNCTIONS ---

def retry_request(func, max_retries=3):
    """Performs exponential backoff retry for API requests. Runs silently."""
    for attempt in range(max_retries):
        try:
            return func()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"[ERROR] API request failed after {max_retries} attempts: {e}")
                raise e
            delay = (2 ** attempt) + (time.time() % 1)
            time.sleep(delay)

def calculate_euclidean_distance(vec1, vec2):
    """Calculates the Euclidean distance between two feature vectors."""
    if len(vec1) != len(vec2):
        return float('inf') 
    
    squared_diffs = [(a - b) ** 2 for a, b in zip(vec1, vec2)]
    return math.sqrt(sum(squared_diffs))

def call_gemini_api(base64_data: str, mime_type: str, user_query: str, system_prompt: str, response_schema: dict) -> dict:
    """Generic function to call the Gemini API with structured JSON output."""
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": user_query}, {"inlineData": {"mimeType": mime_type, "data": base64_data}}]}
        ],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    }

    def make_api_request():
        headers = {'Content-Type': 'application/json'}
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()

    try:
        result = retry_request(make_api_request)
        json_string = result["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(json_string)
    except Exception:
        return {}


def check_image_quality(base64_data: str, mime_type: str) -> str:
    """Calls Gemini (Stage 1) to determine image quality ('good' or 'bad')."""
    if not API_KEY:
        return 'good' # Assume good if API key is missing to prevent total failure

    analysis = call_gemini_api(base64_data, mime_type, QUALITY_USER_QUERY, QUALITY_SYSTEM_PROMPT, QUALITY_RESPONSE_SCHEMA)
    
    return analysis.get('quality_status', 'bad').lower()

def get_image_embedding(base64_data: str, mime_type: str) -> list:
    """Generates a feature vector (Stage 2) for the image using the Gemini API."""
    if not API_KEY:
        return [] # Cannot generate embedding without API key

    analysis = call_gemini_api(base64_data, mime_type, EMBEDDING_USER_QUERY, EMBEDDING_SYSTEM_PROMPT, EMBEDDING_RESPONSE_SCHEMA)
    
    embedding = analysis.get('embedding', [])
    
    # Basic validation of the embedding array
    if embedding and all(isinstance(x, (int, float)) for x in embedding):
        return embedding
    return []

def get_facial_tag(base64_data: str, mime_type: str) -> str:
    """Generates a descriptive facial tag (Stage 3) for the image using the Gemini API."""
    if not API_KEY:
        return "" # Cannot generate tag without API key

    analysis = call_gemini_api(base64_data, mime_type, FACIAL_TAG_USER_QUERY, FACIAL_TAG_SYSTEM_PROMPT, FACIAL_TAG_RESPONSE_SCHEMA)
    
    return analysis.get('facial_tag', '')

