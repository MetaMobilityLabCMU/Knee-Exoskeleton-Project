 /****************************************************************************************** 
  AUTHOR: Rajiv Joshi

  PURPOSE: For CAN communication with Jetson and Teensy.

  UPDATED: 08/19/2025
******************************************************************************************/

#include "Jetson_yap.h"

void send_data_all(Motor Motors[], IMU IMUs[]) {

  data_send data;

  data.motor_1_pos = Motors[0].pos;   // in degrees
  data.motor_1_vel = Motors[0].vel;   // in deg/s
  data.motor_1_torque = Motors[0].torque;   // in Nm
  data.IMU_1_accel_x = (float)IMUs[0].accel.acceleration.x;
  data.IMU_1_accel_y = (float)IMUs[0].accel.acceleration.y;
  data.IMU_1_accel_z = (float)IMUs[0].accel.acceleration.z;
  data.IMU_2_accel_x = (float)IMUs[1].accel.acceleration.x;
  data.IMU_2_accel_y = (float)IMUs[1].accel.acceleration.y;
  data.IMU_2_accel_z = (float)IMUs[1].accel.acceleration.z;
  data.IMU_1_gyro_x = (float)IMUs[0].gyro.gyro.x;
  data.IMU_1_gyro_y = (float)IMUs[0].gyro.gyro.y;
  data.IMU_1_gyro_z = (float)IMUs[0].gyro.gyro.z;
  data.IMU_2_gyro_x = (float)IMUs[1].gyro.gyro.x;
  data.IMU_2_gyro_y = (float)IMUs[1].gyro.gyro.y;
  data.IMU_2_gyro_z = (float)IMUs[1].gyro.gyro.z;

  // format msg to be sent
  msg_fd.id = Teensy_ID + 0x001;    // arbitration id
  msg_fd.len = data_send_size;      // length of data in bytes
  msg_fd.flags.extended = 0;        // standard frame
  msg_fd.edl = 1;                   // extended data length
  msg_fd.brs = 1;                   // bit rate switching on

  // copying buffer of data to message
  memcpy(msg_fd.buf, &data, msg_fd.len);

  // send message to bus using mailbox 1
  if (can3FD.write(MB1, msg_fd)) {
    // Serial.println("Sent motor position CANFD frame to Jetson");
  }
}