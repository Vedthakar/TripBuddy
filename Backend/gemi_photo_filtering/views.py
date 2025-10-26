import json
import logging
import time
import base64
import os
import math
import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# --- GLOBAL CONFIGURATION ---
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyD9V85af2p7KIols5pt-NkBiRY_JPU_Ca8")
API_MODEL = 'gemini-2.5-flash-preview-09-2025'
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{API_MODEL}:generateContent?key={API_KEY}"

# --- LOGGING ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- DUPLICATE AND SIMILARITY THRESHOLDS ---
# These are only used internally within a single POST request to compare multiple images
# if the frontend sends a batch.
EXACT_DUPLICATE_THRESHOLD = 0.001
SIMILAR_PHOTO_THRESHOLD = 5.0

# --- GEMINI STAGE CONFIGURATIONS (same as before) ---
QUALITY_SYSTEM_PROMPT = (
    "You are a professional photo quality analyst. Your task is to objectively evaluate the technical quality of the image. "
    "Check specifically for: 1) High level of motion or focus blur, 2) Main subject's eyes being closed, and 3) Severely bad or distracting lighting/exposure issues. "
    "Return the result in the required JSON format."
)
QUALITY_USER_QUERY = "Analyze the image quality based on technical flaws (blur, closed eyes, bad lighting). State 'good' if acceptable, or 'bad' if it has severe flaws."
QUALITY_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "quality_status": {"type": "STRING"},
        "reasons": {"type": "STRING"}
    },
}

EMBEDDING_SYSTEM_PROMPT = (
    "You are an expert feature extractor for images. Output a dense, 256-dimension embedding array in JSON."
)
EMBEDDING_USER_QUERY = "Generate the feature vector for this photo."
EMBEDDING_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {"embedding": {"type": "ARRAY", "items": {"type": "NUMBER"}}}
}

FACIAL_TAG_SYSTEM_PROMPT = (
    "You are an expert in summarizing dominant faces. Provide a concise string tag that identifies the primary person(s) in the image. For multiple people, use a comma-separated list of names/descriptions."
)
FACIAL_TAG_USER_QUERY = "Generate a descriptive facial tag for the dominant faces."
FACIAL_TAG_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {"facial_tag": {"type": "STRING"}}
}

# --- UTILITY FUNCTIONS ---
def retry_request(func, max_retries=3):
    # ... (existing function logic remains the same) ...
    for attempt in range(max_retries):
        try:
            return func()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                logger.error(f"API request failed after {max_retries} retries: {e}")
                raise
            time.sleep((2 ** attempt) + (time.time() % 1))

def calculate_euclidean_distance(vec1, vec2):
    # ... (existing function logic remains the same) ...
    if len(vec1) != len(vec2):
        return float("inf")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

def call_gemini_api(base64_data, mime_type, user_query, system_prompt, schema):
    # ... (existing function logic remains the same) ...
    if not API_KEY:
        logger.error("Gemini API key missing.")
        return {}
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": user_query}, {"inlineData": {"mimeType": mime_type, "data": base64_data}}]}
        ],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema},
    }
    def make_request():
        response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    try:
        result = retry_request(make_request)
        json_string = result["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(json_string)
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return {}

def check_image_quality(base64_data, mime_type):
    return call_gemini_api(base64_data, mime_type, QUALITY_USER_QUERY, QUALITY_SYSTEM_PROMPT, QUALITY_RESPONSE_SCHEMA)

def get_image_embedding(base64_data, mime_type):
    res = call_gemini_api(base64_data, mime_type, EMBEDDING_USER_QUERY, EMBEDDING_SYSTEM_PROMPT, EMBEDDING_RESPONSE_SCHEMA)
    return res.get("embedding", [])

def get_facial_tag(base64_data, mime_type):
    res = call_gemini_api(base64_data, mime_type, FACIAL_TAG_USER_QUERY, FACIAL_TAG_SYSTEM_PROMPT, FACIAL_TAG_RESPONSE_SCHEMA)
    tag_string = res.get("facial_tag", "")
    # Clean the tags: split by comma, remove whitespace, and filter out empty strings
    tags = [tag.strip() for tag in tag_string.split(',') if tag.strip()]
    # Return empty list if no clear face is detected or tag is generic
    if not tags or tags == ['unknown-face'] or tags == ['']:
        return []
    return tags

# --- CORE IMAGE PROCESSING LOGIC ---
def process_single_image_data(image_data):
    """
    Analyzes and tags a single image (as base64 data) for quality and facial features.
    """
    base64_data = image_data.get("base64_data")
    mime_type = image_data.get("mime_type", "image/jpeg")
    filename = image_data.get("filename", "uploaded_photo") # Use a placeholder if no filename

    if not base64_data:
        return {"filename": filename, "status": "ERROR", "message": "Missing base64 data"}

    try:
        # --- PROCESSING STAGES ---
        quality = check_image_quality(base64_data, mime_type)
        facial_tags = get_facial_tag(base64_data, mime_type)
        embedding = get_image_embedding(base64_data, mime_type) # Kept for potential batch comparison
        
        quality_status = quality.get("quality_status", "unknown").lower()
        
        # --- FOLDER TAGGING LOGIC ---
        folder_tags = ["All"] # Every photo goes to 'All'
        
        if quality_status == "bad":
            folder_tags.append("Bad")
        elif quality_status == "good":
            # For good photos, assign to face folders (Friends)
            # We treat the facial tags as the folder names (e.g., "John", "Jane")
            folder_tags.extend(facial_tags) 

        metadata = {
            "filename": filename,
            "quality_status": quality_status,
            "quality_reasons": quality.get("reasons", ""),
            "facial_tags": facial_tags,
            "folder_tags": folder_tags,      # The key output for the frontend
            "status": "PROCESSED",
            "embedding": embedding,
        }
        return metadata

    except Exception as e:
        logger.error(f"Image processing failed for {filename}: {e}")
        return {"filename": filename, "status": "ERROR", "message": str(e)}
    
# # --- S3-BASED PHOTO PROCESSING FUNCTION ---
# def process_s3_bucket_images():
#     """
#     Loads all images from an S3 bucket, runs photo analysis on each file,
#     and returns structured results.
#     """
#     processed_data_array = []
#     master_embeddings_data = []

#     try:
#         paginator = s3.get_paginator("list_objects_v2")
#         page_iterator = paginator.paginate(Bucket=S3_BUCKET)
#         all_objects = [obj for page in page_iterator for obj in page.get("Contents", [])]

#         logger.info(f"Found {len(all_objects)} files in S3 bucket '{S3_BUCKET}'.")

#         for obj in all_objects:
#             key = obj["Key"]
#             if not key.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".heic")):
#                 continue

#             try:
#                 s3_object = s3.get_object(Bucket=S3_BUCKET, Key=key)
#                 file_bytes = s3_object["Body"].read()
#                 base64_data = base64.b64encode(file_bytes).decode("utf-8")
#                 mime_type = s3_object.get("ContentType", "image/jpeg")
#             except Exception as e:
#                 logger.error(f"Failed to read {key} from S3: {e}")
#                 processed_data_array.append({"filename": key, "status": "READ_ERROR"})
#                 continue

#             # --- PROCESSING STAGES ---
#             quality = check_image_quality(base64_data, mime_type)
#             facial_tag = get_facial_tag(base64_data, mime_type)
#             embedding = get_image_embedding(base64_data, mime_type)

#             metadata = {
#                 "filename": key,
#                 "size_bytes": obj.get("Size", 0),
#                 "quality_status": quality.get("quality_status", "unknown"),
#                 "quality_reasons": quality.get("reasons", ""),
#                 "facial_tag": facial_tag,
#                 "group_id": None,
#                 "status": "PROCESSED",
#                 "embedding": embedding,
#             }

#             # --- DUPLICATE AND SIMILARITY CHECK ---
#             is_duplicate = False
#             for existing in master_embeddings_data:
#                 dist = calculate_euclidean_distance(embedding, existing["embedding"])
#                 if dist < EXACT_DUPLICATE_THRESHOLD:
#                     metadata["status"] = "DUPLICATE"
#                     metadata["duplicate_of"] = existing["filename"]
#                     metadata["group_id"] = existing["group_id"]
#                     is_duplicate = True
#                     break
#                 if dist < SIMILAR_PHOTO_THRESHOLD and existing.get("group_id"):
#                     metadata["group_id"] = existing["group_id"]
#                     break

#             if not is_duplicate:
#                 if not metadata["group_id"]:
#                     metadata["group_id"] = f"group_{len(master_embeddings_data) + 1}"
#                 master_embeddings_data.append(metadata)

#             del metadata["embedding"]
#             processed_data_array.append(metadata)

#         return processed_data_array

#     except Exception as e:
#         logger.error(f"S3 processing failed: {e}")
#         return []

# --- DJANGO VIEW ENTRYPOINT ---
@csrf_exempt
def handle_photo_upload(request):
    """
    Django endpoint that accepts base64 image data via POST, processes it,
    and returns tagging/folder information.
    """
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "Only POST supported for uploads."}, status=405)

    start = time.time()
    
    try:
        data = json.loads(request.body)
        # The request body is expected to contain a list of images (for batch upload)
        # or a single image object. We standardize it to a list for iteration.
        if isinstance(data, dict):
            images_to_process = [data]
        elif isinstance(data, list):
            images_to_process = data
        else:
            return JsonResponse({"status": "error", "message": "Invalid JSON format. Expected an object or a list."}, status=400)

    except json.JSONDecodeError:
        return JsonResponse({"status": "error", "message": "Invalid JSON in request body."}, status=400)
    
    final_results = []
    
    # In a batch scenario, we can compare the incoming photos against each other.
    master_embeddings_data = [] 

    for image_data in images_to_process:
        # 1. Process the image for tags and get its embedding
        metadata = process_single_image_data(image_data)
        
        # 2. Handle duplicates/similarity against other images in this batch (if any)
        embedding = metadata.pop("embedding", [])
        is_duplicate = False
        
        if embedding:
            # Check for duplicates only within this immediate batch
            for existing in master_embeddings_data:
                dist = calculate_euclidean_distance(embedding, existing["embedding"])
                if dist < EXACT_DUPLICATE_THRESHOLD:
                    metadata["status"] = "DUPLICATE_IN_BATCH"
                    metadata["duplicate_of"] = existing["filename"]
                    is_duplicate = True
                    break
            
            # If not a duplicate, add to the batch master list for subsequent checks
            if not is_duplicate and metadata["status"] == "PROCESSED":
                master_embeddings_data.append({"filename": metadata["filename"], "embedding": embedding})

        final_results.append(metadata)

    duration = time.time() - start
    
    # Filter the results for the final output
    processed = [r for r in final_results if r["status"] == "PROCESSED"]

    return JsonResponse({
        "status": "success",
        "total_files_received": len(final_results),
        "unique_processed": len(processed),
        "duration_s": f"{duration:.2f}",
        "results": final_results, # Return all results including duplicates/errors
    }, status=200)
