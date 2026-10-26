/******************************************************************************************
  AUTHOR: Rajiv Joshi

  PURPOSE: Header file defines prototype functions for motors.cpp file.

  UPDATED: 03/27/2026
******************************************************************************************/


#ifndef MOTORS_H
#define MOTORS_H

// Includes:
#include "motor_config.h"
#include <FlexCAN_T4.h>

// Referenced External variables that are declared in another file.
extern FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> can1; // uses CAN 1 on Teensy (CTX1 and CRX1)
extern CAN_message_t msg;




/******************************************************************************************
                                  Data Strucutres:
******************************************************************************************/
/*********************************************
  STRUCTURE: Motor Data

  NOTE: what information can a motor have?

*********************************************/
class Motor {
  public:
    String name;
    uint8_t motor_id;        // motor CAN ID [0x__]
    const char* motor_type; // motor type ["AK_-_"]

    // Limits based on motor type
    float pos_lim_U;          // position upper limit [rad]
    float pos_lim_L;          // position lower limit [rad]
    float vel_lim_U;          // velocity upper limit [rad/s]
    float vel_lim_L;          // velocity lower limit [rad/s]
    float torque_lim_U;       // torque upper limit [Nm]
    float torque_lim_L;       // torque lower limit [Nm]
    float rated_torque_lim_U; // rated torque upper limit [Nm]
    float rated_torque_lim_L; // rated torque lower limit [Nm]
    float kp_lim_U;           // position gain upper limit
    float kp_lim_L;           // position gain lower limit
    float kd_lim_U;           // velocity gain upper limit
    float kd_lim_L;           // velocity gain lower limit
    float gear_ratio;         // gear ratio
    float pole_pairs;         // motor pole pairs
    float KT_constant;        // torque constant


    // Values that will be updated based on motor feedback
    float pos;              // motor position [-3200 deg to 3200 deg]
    float ERPM;             // motor velocity [-320000 ERPM to 320000 ERPM]
    float current;          // motor current [-60A to 60A]
    int8_t temperature;     // motor temperature [-20 C to 127 C]
    int8_t error_code;      // motor error code [0 = no fault, 1 = motor over-temperature, 2 = over-current fault, 3 = over-voltage fault, 4 = under-voltage fault, 5 = encoder fault, 6 = MOSFET over-temperature, 7 = motor lock-up]

    float vel;
    float torque;
    

    // Velocity calculation parameters
    static constexpr float RPM_to_deg = 6.0f; // conversion factor from RPM to deg/s : 360 deg / 60 sec = 6 deg/s per RPM

    // Constructor
    Motor(String name, uint8_t id, const char* type) {
      // Assign basic motor information only
      this->name = name;
      this->motor_id = id;
      this->motor_type = type;
    }

    // Call this after all global objects are constructed (e.g., in setup())
    void initFromDatabase() {
      // Based on motor type we get the motor limits
      const MotorParams& data = motor_database.at(motor_type);

      // Update the Motor limit parameters
      this->pos_lim_L = data.position_limits[0];
      this->pos_lim_U = data.position_limits[1];
      this->vel_lim_L = data.velocity_limits[0];
      this->vel_lim_U = data.velocity_limits[1];
      this->torque_lim_L = data.torque_limits[0];
      this->torque_lim_U = data.torque_limits[1];
      this->rated_torque_lim_L = data.rated_torque_limits[0];
      this->rated_torque_lim_U = data.rated_torque_limits[1];
      this->kp_lim_L = data.kp_limits[0];
      this->kp_lim_U = data.kp_limits[1];
      this->kd_lim_L = data.kd_limits[0];
      this->kd_lim_U = data.kd_limits[1];
      this->gear_ratio = data.gear_ratio;
      this->pole_pairs = data.pole_pairs;
      this->KT_constant = data.KT_constant;
    }
    

    // Member Function Prototypes
    void set_torque(float pos_hold, float des_torque);
    void set_speed(float des_speed, float kd);
    void set_position(float des_position, float kp, float kd);
    void motor_receive(const CAN_message_t &msg);
    
    void print_motor_data ();
    void motors_teleplot();

    float getPosition();
    float getVelocity();
    

  private:
      // Since this is a private function it can only be called by other functions that are in this class like set_torque
      void pack_motor_message(float p_des, float v_des, float kp, float kd, float t_ff);
};




/******************************************************************************************
                                  Funciton Prototypes:
******************************************************************************************/
void comm_can_transmit_sid(uint32_t id, const uint8_t *data, uint8_t len);
void comm_can_transmit_eid(uint32_t id, const uint8_t *data, uint8_t len);

void buffer_append_int32(uint8_t *buffer, int32_t number, int32_t *index);
void buffer_append_int16(uint8_t *buffer, int16_t number, int32_t *index);

void comm_can_set_duty(uint8_t controller_id, float duty);
void comm_can_set_current(uint8_t controller_id, float current);
void comm_can_set_cb(uint8_t controller_id, float current);
void comm_can_set_rpm(uint8_t controller_id, float rpm);
void comm_can_set_pos(uint8_t controller_id, float pos);
void comm_can_set_origin(uint8_t controller_id, uint8_t set_origin_mode);
void comm_can_set_pos_spd(uint8_t controller_id, float pos, int16_t spd, int16_t RPA);

// unsigned int float_to_uint(float x, float x_min, float x_max, unsigned int num_bits);
int float_to_uint(float x, float x_min, float x_max, unsigned int num_bits);
float uint_to_float(uint16_t x_int, float x_min, float x_max, int num_bits);

float fmaxf(float x, float y);
float fminf(float x, float y);


#endif // end of header file
