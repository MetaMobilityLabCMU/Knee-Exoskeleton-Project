import can
import os
import struct

class Device:
    def __init__(self):
        self.Motor_pos_data = []
        self.Motor_vel_data = []
        self.IMU1_accel_data = []
        self.IMU1_gyro_data = []
        self.IMU2_accel_data = []
        self.IMU2_gyro_data = []

def configure_can_interface(channel='can0', bitrate=1000000, dbitrate=2000000):
    print(f"--- Configuring {channel} Hardware ---")
    commands = [
        'sudo modprobe can', 'sudo modprobe can_raw', 'sudo modprobe mttcan',
        f'sudo ip link set {channel} down',
        'sudo rmmod can_raw', 'sudo rmmod can', 'sudo rmmod mttcan',
        'sudo modprobe can', 'sudo modprobe can_raw', 'sudo modprobe mttcan',
        f'sudo ip link set {channel} down',
        f'sudo ip link set {channel} type can bitrate {bitrate} dbitrate {dbitrate} fd on',
        f'sudo ip link set {channel} txqueuelen 1000 up'
    ]
    
    for cmd in commands:
        os.system(cmd)
        
    print(f"{channel} is UP and Ready.")


class JetsonCanInterface:
    def __init__(self, device_storage, channel='can0', teensy_id=0x100):
        self.channel = channel
        self.device = device_storage
        self.bus = None
        self.notifier = None
        self.Teensy_id = teensy_id
        self.data_recieve_id = teensy_id + 0x001
        self.data_request_id = teensy_id + 0x002
        self.torque_cmd_id = teensy_id + 0x003

    def connect(self):
        try:
            self.bus = can.interface.Bus(
                channel=self.channel,
                interface='socketcan',
                fd=True,
                receive_own_messages=False
            )
            print(f"Connected to {self.channel} | Listening for ID: {hex(self.Teensy_id)}")
        except Exception as e:
            print(f"Failed to connect: {e}")
            return

        self.notifier = can.Notifier(self.bus, [self._on_message_received])

    def _send_message(self, msg):
        if self.bus:
            try:
                self.bus.send(msg)
            except can.CanError:
                pass

    def getParameters(self):
        msg = can.Message(arbitration_id=self.data_request_id, data=[], 
                          is_extended_id=False, is_fd=True, bitrate_switch=True)

        self._send_message(msg)

    def setTorque(self, torque_float: float):
        data = struct.pack('<f', torque_float)
        msg = can.Message(arbitration_id=self.torque_cmd_id, data=data, 
                          is_extended_id=False, is_fd=True, bitrate_switch=True)
        self._send_message(msg)

    def _on_message_received(self, msg: can.Message):
        if msg.arbitration_id == self.data_recieve_id:
            data_bytes = msg.data
            NUM_FLOATS = 15
            expected_len = NUM_FLOATS * 4
            
            if len(data_bytes) < expected_len:
                return

            try:
                unpacked = struct.unpack(f'<{NUM_FLOATS}f', data_bytes[:expected_len])
                
                if self.device:
                    self.device.Motor_pos_data.append(unpacked[0])
                    self.device.Motor_vel_data.append(unpacked[1])
                    self.device.IMU1_accel_data.append(unpacked[3:6])
                    self.device.IMU2_accel_data.append(unpacked[6:9])
                    self.device.IMU1_gyro_data.append(unpacked[9:12])
                    self.device.IMU2_gyro_data.append(unpacked[12:15])

            except struct.error:
                pass

    def close(self):
        if self.notifier: self.notifier.stop()
        if self.bus: self.bus.shutdown()
        print(f"Connection for ID {hex(self.Teensy_id)} closed.")