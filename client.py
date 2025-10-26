import socketio
import time

# Create a Socket.IO client
sio = socketio.Client()

@sio.event
def connect():
    """Handle connection to server"""
    print('Successfully connected to server!')
    print('Waiting for sensor data...\n')

@sio.event
def disconnect():
    """Handle disconnection from server"""
    print('Disconnected from server')

@sio.on('updateSensorData')
def on_sensor_data(data):
    """Handle incoming sensor data"""
    print(f"Sensor Value: {data['value']:<8} | Date: {data['date']}")

if __name__ == '__main__':
    try:
        # Connect to the server
        print('Connecting to server...')
        sio.connect('http://localhost:5000')
        
        # Keep the client running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print('\nShutting down client...')
        sio.disconnect()
    except Exception as e:
        print(f'Error: {e}')