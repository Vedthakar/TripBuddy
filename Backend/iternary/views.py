import os
from io import BytesIO
from dotenv import load_dotenv
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from django.core.files.uploadedfile import UploadedFile
from elevenlabs import ElevenLabs

# --- ElevenLabs Setup ---
# NOTE: You must have a .env file in your Django project root 
# with ELEVENLABS_API_KEY set for this to work in production.
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if ELEVENLABS_API_KEY:
    try:
        elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    except Exception as e:
        # Handle the case where the client initialization fails
        print(f"ElevenLabs client initialization failed: {e}")
        elevenlabs_client = None
else:
    print("WARNING: ELEVENLABS_API_KEY not found. API calls will fail.")
    elevenlabs_client = None
# ------------------------


@csrf_exempt
def handle_mp4_upload(request: HttpRequest):
    """
    Handles POST requests containing an uploaded MP4 file, processes it using
    the ElevenLabs speech-to-text API, and returns the transcription.
    
    Expected form field name for the file: 'video_file'
    """
    if request.method == 'POST':
        if not elevenlabs_client:
             return JsonResponse({
                'status': 'error', 
                'message': 'ElevenLabs API client not initialized. Check ELEVENLABS_API_KEY.'
            }, status=500)
            
        # 1. Check for the uploaded file in the request.
        if 'video_file' in request.FILES:
            uploaded_file: UploadedFile = request.FILES['video_file']

            # --- CORE LOGIC: Reading and preparing the file handle ---

            try:
                # Read the entire file content into a byte variable (necessary for BytesIO)
                mp4_data: bytes = uploaded_file.read()
                
                # Wrap the raw bytes in a BytesIO object, which acts as the 
                # in-memory 'file' handle you need for the ElevenLabs SDK.
                audio_file_handle = BytesIO(mp4_data)
                audio_file_handle.name = uploaded_file.name # Preserve the file name metadata

            except Exception as e:
                return JsonResponse({
                    'status': 'error', 
                    'message': f'Failed to read or wrap file content: {e}'
                }, status=500)

            # -------------------------------------------------------------------
            # SUCCESS! The 'audio_file_handle' variable now holds the file-like 
            # object, ready to be passed to the external library, exactly 
            # like in your example script.
            
            print(f"File Received: {uploaded_file.name}, Size: {len(mp4_data)} bytes")
            print("audio_file_handle (BytesIO object) is ready for modification/processing.")
            
            # --- ElevenLabs Processing (Your modification stage) ---
            try:
                # NOTE: We assume the ElevenLabs API can correctly extract the audio track 
                # from the MP4 container for transcription.
                transcription = elevenlabs_client.speech_to_text.convert(
                    file=audio_file_handle,
                    model_id="scribe_v1", 
                    tag_audio_events=True,
                    language_code="eng",
                    diarize=True,
                )
                
                # The transcription object is likely a structured object from the SDK.
                # We convert it to a dictionary or string for JSON response.
                if hasattr(transcription, 'to_dict'):
                    transcription_output = transcription.to_dict()
                else:
                    transcription_output = str(transcription)

                return JsonResponse({
                    'status': 'success',
                    'message': f'MP4 file "{uploaded_file.name}" processed successfully.',
                    'transcription': transcription_output
                }, status=200)

            except Exception as e:
                return JsonResponse({
                    'status': 'error', 
                    'message': f'ElevenLabs API call failed: {e}'
                }, status=500)

        else:
            return JsonResponse({
                'status': 'error', 
                'message': 'No file found. Ensure your curl/POST request uses the form field name "video_file".'
            }, status=400)
    else:
        return JsonResponse({'status': 'error', 'message': 'Only POST requests are accepted at this endpoint.'}, status=405)