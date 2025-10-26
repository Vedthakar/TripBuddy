import json
import logging
import time
import base64
import os
import math
import requests # Required for calling the Gemini API
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# --- GLOBAL CONFIGURATION ---
# NOTE: In a real Django environment, it's best to configure the API key in settings.py
# For this example, we keep the user's proposed environment variable setup.
API_KEY = os.environ.get("GEMINI_API_KEY", "")
API_MODEL = 'gemini-2.5-flash-preview-09-2025'
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{API_MODEL}:generateContent?key={API_KEY}"

# Thresholds for vector distance comparison (Tweak these values based on your dataset)
EXACT_DUPLICATE_THRESHOLD = 0.001  # Near zero distance for floating-point comparison (delete)
SIMILAR_PHOTO_THRESHOLD = 5.0      # Distance threshold for grouping similar photos (group)

# Set up basic logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
config = {
  "apiKey": os.environ.get("APIKEY", ""),
  "authDomain":  os.environ.get("AUTHDOMAIN", ""),
  "projectId": os.environ.get("PROJECTID", ""),
  "storageBucket": os.environ.get("STORAGEBUCKET", ""),
  "messagingSenderId": os.environ.get("MESSAGESENDERID", ""),
  "appId": os.environ.get("APPID", ""),
  "measurementId": os.environ.get("MEASUREMENTID", "")
}
# --- FIREBASE ADMIN SDK SETUP ---
try:
    import firebase_admin
    from firebase_admin import firestore
    # WARNING: Firebase Admin must be initialized outside this function, typically
    # in settings.py or app config, using credentials.Certificate().
    if not firebase_admin._apps:
        logger.warning("Firebase Admin SDK is not initialized. Using a mock initialization.")
        # In a real app, you MUST initialize the SDK here if not done elsewhere.
        # firebase_admin.initialize_app(...)
except ImportError:
    logger.error("firebase_admin SDK not found. Firebase write operations will be simulated.")
    firebase_admin = None
    firestore = None

# Define a function to write processed image data to Firebase
def write_to_firebase_db(data_to_save):
    """
    Writes processed image data to a Firebase NoSQL database (Cloud Firestore).
    Requires Firebase Admin SDK to be initialized.
    """
    if not firebase_admin or not firestore:
        logger.error("Firebase Admin SDK is not initialized or imported. Cannot write to Firebase.")
        return False

    try:
        db = firestore.client()
        
        # We use a batch write for efficiency with many documents
        batch = db.batch()
        collection_ref = db.collection('processed_photos')
        
        for item in data_to_save:
            # We only write PROCESSED records, not duplicates marked for discard
            if item.get('status') == 'PROCESSED':
                new_doc_ref = collection_ref.document()
                # Remove the large embedding array before saving to prevent size issues 
                # or if the embedding is not needed for later queries (optional optimization)
                item_to_save = {k: v for k, v in item.items() if k != 'embedding'}
                batch.set(new_doc_ref, item_to_save)
            
        batch.commit()
        
        logger.info(f"Successfully wrote {len(data_to_save)} items to 'processed_photos' collection in Firebase.")
        return True
    except Exception as e:
        logger.error(f"Firebase database write failed: {e}")
        return False


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
                logger.error(f"[ERROR] API request failed after {max_retries} attempts: {e}")
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
    if not API_KEY:
        logger.error("API Key is missing. Skipping Gemini API call.")
        return {}

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
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return {}


def check_image_quality(base64_data: str, mime_type: str) -> dict:
    """Calls Gemini (Stage 1) to determine image quality ('good' or 'bad')."""
    if not API_KEY:
        return {'quality_status': 'good', 'reasons': 'API key missing, assumed good'}

    return call_gemini_api(base64_data, mime_type, QUALITY_USER_QUERY, QUALITY_SYSTEM_PROMPT, QUALITY_RESPONSE_SCHEMA)

def get_image_embedding(base64_data: str, mime_type: str) -> list:
    """Generates a feature vector (Stage 2) for the image using the Gemini API."""
    if not API_KEY:
        return [] 

    analysis = call_gemini_api(base64_data, mime_type, EMBEDDING_USER_QUERY, EMBEDDING_SYSTEM_PROMPT, EMBEDDING_RESPONSE_SCHEMA)
    
    embedding = analysis.get('embedding', [])
    
    if embedding and all(isinstance(x, (int, float)) for x in embedding):
        return embedding
    return []

def get_facial_tag(base64_data: str, mime_type: str) -> str:
    """Generates a descriptive facial tag (Stage 3) for the image using the Gemini API."""
    if not API_KEY:
        return "tag-generation-skipped"

    analysis = call_gemini_api(base64_data, mime_type, FACIAL_TAG_USER_QUERY, FACIAL_TAG_SYSTEM_PROMPT, FACIAL_TAG_RESPONSE_SCHEMA)
    
    return analysis.get('facial_tag', 'tag-generation-failed')


@csrf_exempt
def handle_photo_upload(request):
    """
    Handles POST requests for bulk image processing, filtering, and database write.
    """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST requests are supported.'}, status=405)

    # 1. Parse and validate images
    uploaded_files = request.FILES.getlist('images')

    if not uploaded_files:
        return JsonResponse({'status': 'error', 'message': 'No files found under the field name "images".'}, status=400)

    # This array will hold the final metadata for ALL images (including duplicates)
    processed_data_array = []
    
    # This list will hold embeddings for images that have already been fully processed 
    # (i.e., not duplicates) to compare subsequent images against.
    master_embeddings_data = [] 
    
    start_time = time.time()
    
    logger.info(f"Starting processing for {len(uploaded_files)} images.")

    # 2. Iterate, Process, and Filter/Group each image
    for index, image_file in enumerate(uploaded_files):
        
        # Read the file content and convert to base64 for the API
        try:
            image_file.seek(0) # Ensure we read from the start of the file
            base64_data = base64.b64encode(image_file.read()).decode('utf-8')
            mime_type = image_file.content_type
        except Exception as e:
            logger.error(f"Failed to read file {image_file.name}: {e}")
            processed_data_array.append({'original_filename': image_file.name, 'status': 'READ_ERROR'})
            continue


        ##############################################################
        # processing
        ##############################################################
        
        # STAGE 1 & 3: Quality Check and Facial Tagging
        quality_analysis = check_image_quality(base64_data, mime_type)
        facial_tag = get_facial_tag(base64_data, mime_type)
        
        # STAGE 2: Generate Feature Embedding
        embedding = get_image_embedding(base64_data, mime_type)

        current_metadata = {
            'original_filename': image_file.name,
            'size_bytes': image_file.size,
            'mimetype': mime_type,
            'processed_timestamp': int(time.time() * 1000),
            'quality_status': quality_analysis.get('quality_status', 'bad'),
            'quality_reasons': quality_analysis.get('reasons', ''),
            'facial_tag': facial_tag,
            'group_id': None,
            'status': 'PROCESSED', # Default status
            'embedding': embedding, # Store temporarily for comparison
        }

        # --- Filtering and Grouping Logic ---
        
        is_exact_duplicate = False
        
        if embedding:
            for existing_data in master_embeddings_data:
                existing_embedding = existing_data['embedding']
                distance = calculate_euclidean_distance(embedding, existing_embedding)
                
                # Check 1: Exact Duplicate (too close to be different)
                if distance < EXACT_DUPLICATE_THRESHOLD:
                    is_exact_duplicate = True
                    current_metadata['status'] = 'DUPLICATE'
                    current_metadata['duplicate_of'] = existing_data['original_filename']
                    current_metadata['group_id'] = existing_data['group_id']
                    break
                
                # Check 2: Similar Photo (for grouping)
                if distance < SIMILAR_PHOTO_THRESHOLD and existing_data['group_id'] is not None:
                    # Assign the new image to the existing group
                    current_metadata['group_id'] = existing_data['group_id']
                    break

        
        if is_exact_duplicate:
            # We don't save duplicates to the master list or Firebase, just record the metadata
            logger.info(f"Image {image_file.name} marked as DUPLICATE of {current_metadata['duplicate_of']}")
            
        else:
            # If it's not a duplicate and doesn't have a group_id yet, assign a new one
            if not current_metadata['group_id']:
                current_metadata['group_id'] = f"group_{len(master_embeddings_data) + 1}"
            
            # Add to the master list for subsequent comparison
            master_embeddings_data.append(current_metadata)
        
        # Remove the potentially very large base64 data from the object before appending
        del current_metadata['embedding']
        
        processed_data_array.append(current_metadata)


    end_time = time.time()
    total_time = end_time - start_time
    
    items_to_save = [item for item in processed_data_array if item['status'] == 'PROCESSED']

    logger.info(f"Processing complete. Total time: {total_time:.2f}s. {len(items_to_save)} items ready for DB.")
    
    # 3. Write all non-duplicate, fully processed data to the Firebase database
    if write_to_firebase_db(items_to_save):
        return JsonResponse({
            'status': 'success',
            'message': f'Successfully processed and saved {len(items_to_save)} unique images to Firebase.',
            'total_processing_time_s': f'{total_time:.2f}',
            'uploaded_image_count': len(uploaded_files),
            'unique_images_saved': len(items_to_save),
            'sample_data': processed_data_array[:3]
        }, status=200)
    else:
        # If the Firebase write failed
        return JsonResponse({
            'status': 'error', 
            'message': 'Processing complete, but Firebase database write failed. Check logs for API/Firebase key issues.', 
            'processed_count': len(uploaded_files),
            'items_prepared_for_save': len(items_to_save)
        }, status=500)
