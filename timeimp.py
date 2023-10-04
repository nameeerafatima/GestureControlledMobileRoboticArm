from dronekit import connect, VehicleMode
import time
import socket

connection_string = 'COM7'  # Replace with the serial port of your Pixhawk
baud_rate = 57600  # Replace with the baud rate that your Pixhawk is configured to use
vehicle = connect(connection_string, baud=baud_rate, wait_ready=True)

NO_MOVEMENT_THRESHOLD = 5
NO_MOVEMENT_THRESHOLD1 = 7  # Time threshold for no movement in seconds
no_movement_timer = 0  # Timer to track no movement time
is_armed = False  # Flag indicating if the rover is armed

def arm_and_set_guided_mode():
    global is_armed
    print("Arming motors...")
    vehicle.mode = VehicleMode("MANUAL")
    vehicle.armed = True
    is_armed = True
    time.sleep(1)
    print(vehicle.armed)
    
    vehicle.flush()

def disarm_vehicle():
    global is_armed
    print("Disarming motors...")
    vehicle.armed = False
    is_armed = False
    
    vehicle.flush()

def handle_no_movement():
    global no_movement_timer, is_armed
    no_movement_timer += 1

    if no_movement_timer >= NO_MOVEMENT_THRESHOLD and not is_armed:
        arm_and_set_guided_mode()
        no_movement_timer = 0

    if no_movement_timer >= NO_MOVEMENT_THRESHOLD1 and is_armed:
        disarm_vehicle()
        no_movement_timer = 0

def handle_movement():
    global no_movement_timer
    no_movement_timer = 0  # Reset the no movement timer

# Initialize the rover control
rover = vehicle  # Replace with your rover control class or logic

# Start a TCP/IP server to listen for gesture commands
host = '172.168.3.113'  # Replace with the IP address or hostname of the rover
port = 7  # Replace with the port number the rover is listening on
gesture_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
gesture_socket.bind((host, port))
gesture_socket.listen(1)

# for arm socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host1 = "172.168.3.157"
port1 = 5000
client_socket.connect((host1, port1))
print("Connected to Jetson Nano")





# ##################
print(f"Listening for gesture commands on {host}:{port}...")
c = 0
while True:
    conn, addr = gesture_socket.accept()
    print(f"Connected to gesture recognition at {addr[0]}:{addr[1]}")

    while True:
        # additional not armed RED 
        if(vehicle.armed == False):
            gesture_socket.send(b'unarmed')
        
        data = conn.recv(1024)

        if not data:
            break

        command = data.decode()
        print(command)

        if command == 'rstop':
            handle_no_movement()
        else:
            handle_movement()

        if not is_armed:
            continue

        if command == 'rforward':
            vehicle.channels.overrides['3'] = 1800  # Throttle
            time.sleep(1)
            vehicle.channels.overrides['1'] = 1500
            vehicle.channels.overrides['3'] = 1500
            time.sleep(1)
            # rover.move_forward()
        elif command == 'rright':
            vehicle.channels.overrides['1'] = 1300
            vehicle.channels.overrides['3'] = 1500
            time.sleep(0.8)
            vehicle.channels.overrides['1'] = 1500
            vehicle.channels.overrides['3'] = 1500
            time.sleep(0.1)
        elif command == 'rleft':
            vehicle.channels.overrides['1'] = 1710
            vehicle.channels.overrides['3'] = 1500
            time.sleep(0.9)
            vehicle.channels.overrides['1'] = 1500
            vehicle.channels.overrides['3'] = 1500
            time.sleep(0.1)

 

        elif command == 'rbackward':
            vehicle.channels.overrides['3'] = 1300 # Throttle
            time.sleep(1)
            vehicle.channels.overrides['1'] = 1500
            vehicle.channels.overrides['3'] = 1500
            time.sleep(1)

        # elif command=='open':
        #     message = 'aleft'
        #     client_socket.send(message.encode())

        else:
            message = command
            client_socket.send(message.encode())
            
            
        # elif command=='abackward':
        #     message = command
        #     client_socket.send(message.encode())
            
        # elif command=='aleft':
        #     message = command
        #     client_socket.send(message.encode())
            
        # elif command=='aright':
        #     message = command
        #     client_socket.send(message.encode())

        # elif command=='close':
        #     message = command
        #     client_socket.send(message.encode())

        # elif command=='open':
        #     message = command
        #     client_socket.send(message.encode())