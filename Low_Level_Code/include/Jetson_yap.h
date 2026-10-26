/******************************************************************************************
  AUTHOR: Rajiv Joshi

  PURPOSE: Header file defines prototype functions for Jetson_yap.cpp file.

  UPDATED: 08/19/2025
******************************************************************************************/

#ifndef JETSON_YAP_H
#define JETSON_YAP_H


#include <FlexCAN_T4.h>
#include "motors.h"
#include "IMU_config.h"

// Referenced External variables that are declared in another file.
extern FlexCAN_T4FD<CAN3, RX_SIZE_256, TX_SIZE_16> can3FD;  // uses CAN 3 on Teensy (CTX3 and CRX3)
extern CANFD_message_t msg_fd;
extern uint32_t Teensy_ID; // Base ID for this Teensy controller

typedef struct __attribute__((packed)) {
    float motor_1_pos;      // 0-3
    float motor_1_vel;      // 4-7
    float motor_1_torque;   // 8-11
    
    // IMU 1 Accel (12-23)
    float IMU_1_accel_x;
    float IMU_1_accel_y;
    float IMU_1_accel_z;
    
    // IMU 2 Accel (24-35)
    float IMU_2_accel_x;
    float IMU_2_accel_y;
    float IMU_2_accel_z;
    
    // IMU 1 Gyro (36-47)
    float IMU_1_gyro_x;
    float IMU_1_gyro_y;
    float IMU_1_gyro_z;
    
    // IMU 2 Gyro (48-59)
    float IMU_2_gyro_x;
    float IMU_2_gyro_y;
    float IMU_2_gyro_z;
} data_send;

// Define the size as a constant for easy use
#define data_send_size sizeof(data_send) // 56 bytes

void send_data_all(Motor Motors[], IMU IMUs[]);

#endif