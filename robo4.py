#!/usr/bin/env python

import time
import smbus
import socket

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Get the local machine IP address
host = '172.168.3.157'
port = 5000

# Bind the socket to a specific address and port
server_socket.bind((host, port))


print("Waiting for a connection...")
# Listen for incoming connections
server_socket.listen(1)

# Accept a client connection
client_socket, address = server_socket.accept()
print("Connected to:", address)


# PCA9685 Registers
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06

# PCA9685 Constants
I2C_BUS = 1  # I2C bus number
PCA9685_ADDRESS = 0x40  # I2C address of PCA9685
FREQUENCY = 50  # PWM frequency (Hz)
SERVO_MIN = 500  # Min pulse length for servos (0 degrees)
SERVO_MAX = 2500  # Max pulse length for servos (180 degrees)
SERVO_RANGE = SERVO_MAX - SERVO_MIN  # Pulse length range for servos

def set_pwm_frequency(bus, address, frequency):
    prescale_val = int((25000000 / (4096 * frequency)) - 1)
    bus.write_byte_data(address, MODE1, 0x10)  # Sleep mode
    bus.write_byte_data(address, PRESCALE, prescale_val)
    bus.write_byte_data(address, MODE1, 0x00)  # Wake up

def set_servo_angle(bus, address, channel, angle):
    pulse_width = SERVO_MIN + (float(angle) / 180.0) * SERVO_RANGE
    pulse_width_value = int(pulse_width * 4096 / 20000)  # Convert to 12-bit value
    # Set the PWM pulse for the servo motor
    bus.write_byte_data(address, LED0_ON_L + 4 * channel, 0x00)
    bus.write_byte_data(address, LED0_ON_L + 4 * channel + 1, 0x00)
    bus.write_byte_data(address, LED0_ON_L + 4 * channel + 2, pulse_width_value & 0xFF)
    bus.write_byte_data(address, LED0_ON_L + 4 * channel + 3, pulse_width_value >> 8)


bus = smbus.SMBus(I2C_BUS)
set_pwm_frequency(bus, PCA9685_ADDRESS, FREQUENCY)
print("Starting servo motor control. Press CTRL+C to exit.")

print(f"Listening for gesture commands on {host}:{port}...")

set_servo_angle(bus, PCA9685_ADDRESS, 1, 70)  # Turn left by setting servo angle to 20
time.sleep(1)
print("12")
set_servo_angle(bus, PCA9685_ADDRESS, 2, 17)  # Turn left by setting servo angle to 20
time.sleep(1)
print("3")
set_servo_angle(bus, PCA9685_ADDRESS, 3, 90)  # Turn left by setting servo angle to 20
time.sleep(1)
print("41")
set_servo_angle(bus, PCA9685_ADDRESS, 11, 90)  
time.sleep(1)
flag=0
angle=90
while True:
    
    while True:
        
        # Receive data from the client
        command = client_socket.recv(1024).decode()

        print("Received data:", command)


        if command=='astop':
            print("No stopping in arm")
        elif command=='abackward':
            if(flag==0):
                set_servo_angle(bus, PCA9685_ADDRESS, 3, 45)  # Turn left by setting servo angle to 20
                time.sleep(1)
                print("41")
                set_servo_angle(bus, PCA9685_ADDRESS, 2, 30)  # Turn left by setting servo angle to 20
                time.sleep(1)
                print("41")
                for angle1 in range(100,141,10):
                    set_servo_angle(bus, PCA9685_ADDRESS, 1, angle1)  # Turn left by setting servo angle to 20
                    time.sleep(0.5)
                    print("41")
                time.sleep(1)
                flag=1
            elif(flag==1):           
                set_servo_angle(bus, PCA9685_ADDRESS, 1, 70)  # Turn left by setting servo angle to 20
                time.sleep(1)
                print("12")
                set_servo_angle(bus, PCA9685_ADDRESS, 2, 17)  # Turn left by setting servo angle to 20
                time.sleep(1)
                print("3")
                set_servo_angle(bus, PCA9685_ADDRESS, 3, 90)  # Turn left by setting servo angle to 20
                time.sleep(1)
                print("41")
                flag=0

                     
        elif command=='aforward':
            set_servo_angle(bus, PCA9685_ADDRESS, 1, 70)  # Turn left by setting servo angle to 20
            time.sleep(1)
            print("12")
            set_servo_angle(bus, PCA9685_ADDRESS, 2, 17)  # Turn left by setting servo angle to 20
            time.sleep(1)
            print("3")
            set_servo_angle(bus, PCA9685_ADDRESS, 3, 90)  # Turn left by setting servo angle to 20
            time.sleep(1)
            print("41")
            set_servo_angle(bus, PCA9685_ADDRESS, 11, 90)  
            time.sleep(1)
            angle=90

        elif command=='aright':
            
            
            if(angle>0 and angle<=180):
                angle=angle-10
                set_servo_angle(bus, PCA9685_ADDRESS, 11, angle)  # Turn left by setting servo angle to 20
                time.sleep(1)
                print("12")
        elif command=='aleft':
            
            if(angle>=0 and angle<180):
                angle=angle+10
                set_servo_angle(bus, PCA9685_ADDRESS, 11, angle)  # Turn left by setting servo angle to 20
                time.sleep(1)
                print("12")
        elif command=='open':
            set_servo_angle(bus, PCA9685_ADDRESS, 8, 0)  # Turn left by setting servo angle to 20
            time.sleep(1)
            print("12")
        elif command=='close':
            set_servo_angle(bus, PCA9685_ADDRESS, 8, 180)  # Turn left by setting servo angle to 20
            time.sleep(1)
            print("12")

client_socket.close()
server_socket.close()
