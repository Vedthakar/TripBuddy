import eventlet
import socketio
import json
from datetime import datetime
import time

# --- Configuration ---
HOST = '127.0.0.1'
PORT = 5000
LOCATION_ROOM = 'global_location_share'
# Dictionary to store the latest known location for a user ID (for server-side processing)
current_user_locations = {} 

# Initialize the Socket.IO server
# Set cors_allowed_origins='*' to allow connections from any client (essential for testing)
sio = socketio.Server(cors_allowed_origins='*')
app = socketio.WSGIApp(sio)

print(f"Starting Socket.IO Server on ws://{HOST}:{PORT}")
print(f"Location Room Name: {LOCATION_ROOM}")

# --- Event Handlers ---

@sio.event
def connect(sid, environ):
    """
    Handles a new client connection.
    SID (Session ID) is the unique ID for this specific socket connection.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] CONNECT: New client connected. SID: {sid}")

@sio.event
def disconnect(sid):
    """
    Handles a client disconnection.
    Removes the user from the location tracking dictionary.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] DISCONNECT: Client disconnected. SID: {sid}")
    
    # Simple logic to find and remove the user from our tracking dict
    # Note: In a real app, you'd track users associated with the SID.
    user_to_remove = None
    for user_id, location_data in current_user_locations.items():
        if location_data['sid'] == sid:
            user_to_remove = user_id
            break
    
    if user_to_remove:
        del current_user_locations[user_to_remove]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] INFO: User {user_to_remove} removed from tracking.")
        # Optionally, broadcast a user_left event to the room
        sio.emit('user_left', {'user_id': user_to_remove}, room=LOCATION_ROOM)


@sio.on('join_location_feed')
def on_join_location_feed(sid, data):
    """
    Custom event triggered by the client (Swift app) to join the main room.
    Data is expected to be a dictionary like {'user_id': 'user_X'}.
    """
    try:
        user_id = data.get('user_id')
        if not user_id:
            sio.emit('error_message', {'message': 'User ID required to join.'}, room=sid)
            return

        # 1. Join the room
        sio.enter_room(sid, LOCATION_ROOM)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] JOIN: User '{user_id}' with SID {sid} joined room '{LOCATION_ROOM}'.")

        # 2. Store the user's initial state
        current_user_locations[user_id] = {'sid': sid, 'lat': 0.0, 'lon': 0.0}

        # 3. Send confirmation back to the client
        sio.emit('room_joined', {'status': 'success', 'room': LOCATION_ROOM}, room=sid)

        # 4. (Optional) Send current locations of ALL other users to the new user
        # In a real app, this ensures the new client sees existing users immediately.
        initial_data = [
            {'user_id': uid, 'lat': loc['lat'], 'lon': loc['lon']} 
            for uid, loc in current_user_locations.items() if uid != user_id
        ]
        sio.emit('initial_locations', {'locations': initial_data}, room=sid)

    except Exception as e:
        print(f"Error processing join_location_feed: {e}")
        sio.emit('error_message', {'message': f'Server error on join: {e}'}, room=sid)


@sio.on('location_update')
def on_location_update(sid, data):
    """
    Handles real-time location updates from a single client.
    Data expected: {'user_id': 'user_X', 'lat': 40.7128, 'lon': -74.0060}.
    """
    try:
        user_id = data.get('user_id')
        latitude = data.get('lat')
        longitude = data.get('lon')

        if not all([user_id, latitude, longitude]):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Invalid data received from {sid}.")
            sio.emit('error_message', {'message': 'Invalid location data format.'}, room=sid)
            return

        # 1. Server-Side Logic (The "Mathematics" step)
        # You would perform distance calculations, security checks, and geofencing here.
        # For simplicity, we just update the stored location.
        current_user_locations[user_id].update({'lat': latitude, 'lon': longitude})
        
        # Prepare the data for broadcast (only sending the necessary info)
        broadcast_data = {
            'user_id': user_id,
            'lat': latitude,
            'lon': longitude,
            'timestamp': datetime.now().timestamp()
        }

        # 2. Broadcast the update to the entire room (Pub/Sub)
        # 'skip_sid=sid' ensures the sender doesn't receive their own update, saving bandwidth.
        sio.emit('new_user_location', broadcast_data, room=LOCATION_ROOM, skip_sid=sid)

        # print(f"[{datetime.now().strftime('%H:%M:%S')}] BROADCAST: {user_id} @ ({latitude}, {longitude})")

    except Exception as e:
        print(f"Error processing location_update: {e}")

# --- Start the Server ---
if __name__ == '__main__':
    try:
        # eventlet.wsgi.server is used to wrap the Socket.IO app for async serving
        eventlet.wsgi.server(eventlet.listen((HOST, PORT)), app)
    except KeyboardInterrupt:
        print("\nServer shutting down.")
    except Exception as e:
        print(f"An error occurred while running the server: {e}")
