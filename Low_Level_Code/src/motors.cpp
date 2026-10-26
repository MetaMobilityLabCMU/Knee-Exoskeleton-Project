/****************************************************************************************** 
  AUTHOR: Rajiv Joshi

  PURPOSE: For understanding of how Arduino implementaion of AK motors CAN + Teensy 
  works.

  UPDATED: 03/27/2026
******************************************************************************************/

// Includes:
#include "motors.h"

// check header file for other include definitions.

/*********************************************
AK 80-9 has 7 different Servo Contorl Modes
*********************************************/
enum CAN_PACKET_ID { // enum assigns automatically a paired named-integer constant to this predefined limited list
  CAN_PACKET_SET_DUTY = 0, // 0: Specifies the motors Duty Cycle voltage in a square wave driving form.
  CAN_PACKET_SET_CURRENT, // 1: Specifies the Iq current of the motor. As the motor output torque = Iq * KT, it can be used as a torque loop.
  CAN_PACKET_SET_CURRENT_BRAKE, // 2: Specifies the braking current of the motor to fix the motor at the current position (pay attention to motor temperature during use).
  CAN_PACKET_SET_RPM, // 3: Specifies the running speed of the motor.
  CAN_PACKET_SET_POS, // 4: Specifies the position of the motor, and the motor will run to the specified position at the maximum speed.
  CAN_PACKET_SET_ORIGIN_HERE, // 5: Specifies a temporary origin (zero) position (clears after power-off), or a permanent zero point (for dual encoder modes only).
  CAN_PACKET_SET_POS_SPD, // 6: Specifies the position, speed, and acceleration of the motor. The motor will run to the specified position with the given acceleration and maximum speed.
  CAN_PACKET_SET_MIT = 8
};


/************************************************************************************************************************************************************************************ 
                                                                                  Helper Functions:
************************************************************************************************************************************************************************************/
/******************************************************************************************
  FUNCTION: Adds data to the message buffer, specifies the CAN message length, and writes 
  the CAN signal to the bus.
******************************************************************************************/
void comm_can_transmit_sid(uint32_t id, const uint8_t *data, uint8_t len) { // pass in the ID, data, and length of the data
  msg.id = id; // idea for arbitration (aka deciding which CAN signal will get to use the bus)
  msg.len = len; // how long the legth of the data is
  memcpy(msg.buf, data, len); // copys the data to the message buffer at the specified length
  can1.write(msg); // execution of writing the CAN signal with data to the bus
}




/******************************************************************************************
  FUNCTION: Extended Version, Adds data to the message buffer, specifies the CAN message 
  length, and writes the CAN signal to the bus.
******************************************************************************************/
void comm_can_transmit_eid(uint32_t id, const uint8_t *data, uint8_t len) { // pass in the ID, data, and length of the data
  msg.id = id; // idea for arbitration (aka deciding which CAN signal will get to use the bus)
  msg.len = len; // how long the legth of the data is
  msg.flags.extended = 1; // extended CAN ID length
  memcpy(msg.buf, data, len); // copys the data to the message buffer at the specified length
  can1.write(msg); // execution of writing the CAN signal with data to the bus
}




/******************************************************************************************
  FUNCTION: Adding a int32 to the data buffer Consdier the int32_t = 0x12345678. This 
  function will split the data in to 4 parts since our buffer is a 
  unit8_t: [0x12 0x34 0x56 0x78]
******************************************************************************************/
void buffer_append_int32(uint8_t *buffer, int32_t number, int32_t *index) { // variables passed in, buffer size is passed in here (how long is the data buffer)
  buffer[(*index)++] = number >> 24; // using our example: 0x12345678 >> 24 = 0x12 so this is stored into the buffer[0], the index pointer gets incremented to 1
  buffer[(*index)++] = number >> 16; // Next: 0x12345678 >> 16 = 0x1234 so the leats significat byte is stored: 0x34 into buffer[1], index = 2
  buffer[(*index)++] = number >> 8; // Next: 0x12345678 >> 8 = 0x123456 so the leats significat byte is stored: 0x56 into buffer[2], index = 3
  buffer[(*index)++] = number; // Next: 0x12345678 = 0x12345678 so the leats significat byte is stored: 0x78 into buffer[3], index = 4
}




/******************************************************************************************
  FUNCTION: Adding a int16 to the data buffer. Consdier the int16_t = 0x1234. This function
  will split the data in to 4 parts since our buffer is a uint8_t: [0x12 0x34]
******************************************************************************************/
void buffer_append_int16(uint8_t *buffer, int16_t number, int16_t *index) { // variables passed in, buffer size is passed in here (how long is the data buffer), index is where we start filling in
  buffer[(*index)++] = number >> 8; // using our example: 0x1234 >> 8 = 0x12 so this is stored into buffer[0]. index pointer is incremented to 1
  buffer[(*index)++] = number; // Next: 0x1234 = 0x1234 so the leats significat byte is stored: 0x78 into buffer[1], index = 2
}




/******************************************************************************************
  FUNCTION: Servo Mode set Duty Cycle for AK 80-9 Motor. Check datasheet for understanding 
  communcaition data format.

  Data Buffer Format:
    Range:     [         0~0xFF      ,         0~0xFF       ,          0~0xFF       ,         0~0xFF       ]
    Variables: [duty cycle 25-32 bits, duty cycle 17-24 bits, duty cycle 9 - 16 bits, duty cycle 1 - 8 bits]

******************************************************************************************/
void comm_can_set_duty(uint8_t controller_id, float duty) {
  int32_t send_index = 0;
  uint8_t buffer[4];
  buffer_append_int32(buffer, (int32_t)(duty*100000.0), &send_index);
  comm_can_transmit_eid(controller_id | ((uint32_t)CAN_PACKET_SET_DUTY << 8), buffer, send_index);
}




/******************************************************************************************
  FUNCTION: Servo Mode set current for AK 80-9 Motor. Check datasheet for understanding
  communcaition data format.

  Data Buffer Format:
    Range:     [       0~0xFF     ,       0~0xFF      ,        0~0xFF      ,       0~0xFF      ]
    Variables: [current 25-32 bits, current 17-24 bits, current 9 - 16 bits, current 1 - 8 bits]

******************************************************************************************/
void comm_can_set_current(uint8_t controller_id, float current) {
  int32_t send_index = 0;
  uint8_t buffer[4];
  buffer_append_int32(buffer, (int32_t)(current*1000.0), &send_index);
  comm_can_transmit_eid(controller_id | ((uint32_t)CAN_PACKET_SET_CURRENT << 8), buffer, send_index);
}




/******************************************************************************************
  FUNCTION: Servo Mode set Brake Current for AK 80-9 Motor. Check datasheet for 
  understanding communcaition data format.

  Data Buffer Format:
    Range:     [          0~0xFF        ,          0~0xFF         ,           0~0xFF         ,          0~0xFF         ]
    Variables: [brake current 25-32 bits, brake current 17-24 bits, brake current 9 - 16 bits, brake current 1 - 8 bits]

******************************************************************************************/
void comm_can_set_cb(uint8_t controller_id, float current) {
  int32_t send_index = 0;
  uint8_t buffer[4];
  buffer_append_int32(buffer, (int32_t)(current*1000.0), &send_index);
  comm_can_transmit_eid(controller_id | ((uint32_t)CAN_PACKET_SET_CURRENT_BRAKE << 8), buffer, send_index);
}




/******************************************************************************************
  FUNCTION: Servo Mode set speed for AK 80-9 Motor. Check datasheet for understanding
  communcaition data format.

  Data Buffer Format:
    Range:     [      0~0xFF    ,      0~0xFF     ,       0~0xFF     ,      0~0xFF     ]
    Variables: [speed 25-32 bits, speed 17-24 bits, speed 9 - 16 bits, speed 1 - 8 bits]

******************************************************************************************/
void comm_can_set_rpm(uint8_t controller_id, float rpm) {
  int32_t send_index = 0;
  uint8_t buffer[4];
  buffer_append_int32(buffer, (int32_t)(rpm), &send_index);
  comm_can_transmit_eid(controller_id | ((uint32_t)CAN_PACKET_SET_RPM << 8), buffer, send_index);
}




/******************************************************************************************
  FUNCTION: Servo Mode set position for AK 80-9 Motor. Check datasheet for understanding
  communcaition data format.

  Data Buffer Format:
    Range:     [       0~0xFF      ,       0~0xFF       ,        0~0xFF       ,       0~0xFF       ]
    Variables: [position 25-32 bits, position 17-24 bits, position 9 - 16 bits, position 1 - 8 bits]

******************************************************************************************/
void comm_can_set_pos(uint8_t controller_id, float pos) {
  int32_t send_index = 0;
  uint8_t buffer[4];
  buffer_append_int32(buffer, (int32_t)(pos * 10000.0f), &send_index);
  comm_can_transmit_eid(controller_id | ((uint32_t)CAN_PACKET_SET_POS << 8), buffer, send_index);
}




/******************************************************************************************
  FUNCTION: Servo Mode setOrigin Function for AK 80-9 Motor. Check datasheet for
  understanding communcaition data format.

  Data Buffer Format:
    Range:     [                   0~0x2                   ]
    Variables: [0 (temp origin) or 1 (permanant zero point)]

******************************************************************************************/
void comm_can_set_origin(uint8_t controller_id, uint8_t set_origin_mode) {
  int32_t send_index = 0; // defining our index variable and initialize to 0 (temp, erased on power loss)
  uint8_t buffer[8] = {0}; // defines a buffer of size 8 where each spot is of size uint8_t
  buffer[0] = set_origin_mode; // adds the data for origin to the first spot
  send_index = 1; // increment the index value our buffer has a length of 1

  comm_can_transmit_eid(controller_id | ((uint32_t)CAN_PACKET_SET_ORIGIN_HERE << 8), buffer, send_index); // using our funciton to piece CAN data together for writting to bus
}




/******************************************************************************************
  FUNCTION: Servo Mode set position and speed for AK 80-9 Motor. Check datatsheet for 
  understanding communication data format

  Data Buffer Format:
    Range:     [       0~0xFF      ,       0~0xFF       ,       0~0xFF      ,      0~0xFF      ,      0~0xFF      ,      0~0xFF     ,      0~0xFF      ,     0~0xFF      ]
    Variables: [position 25-32 bits, position 17-24 bits, position 9-16 bits, positoin 1-8 bits, speed high bits 8, speed low bits 8, accel high bits 8, accel low bits 8]

******************************************************************************************/
void comm_can_set_pos_spd(uint8_t controller_id, float pos, int16_t spd, int16_t RPA) {
  int32_t send_index = 0;
  int16_t send_index1 = 4;
  uint8_t buffer[8] = {0};
  
  buffer_append_int32(buffer, (int32_t)(pos * 10000.0), &send_index);
  buffer_append_int16(buffer, (int16_t)(spd / 10.0), &send_index1);
  buffer_append_int16(buffer, (int16_t)(RPA / 10.0), &send_index1);
  

  comm_can_transmit_eid(controller_id | ((uint32_t)CAN_PACKET_SET_POS_SPD << 8), buffer, send_index1);
    
}




/******************************************************************************************
  FUNCTION: Interpolates a floating point number to an unsigned integer of num_bits length.
  A number of x_max will be the largest integer of num_bits, and x_min would be 0.

  ARGS:
    x (float): The floating point number to convert
    x_min (float): The minimum value for the floating point number
    x_max (float): The maximum value for the floating point number
    num_bits (int): The number of bits for the unsigned integer

  RETURNS:
    int: The unsigned integer representation of the floating point number
******************************************************************************************/
int float_to_uint(float x, float x_min, float x_max, unsigned int num_bits) {

  // Clamping the value of x to be within the range of x_min and x_max to avoid overflow or underflow when converting to uint
  x = constrain(x, x_min, x_max);

  // Calcualting the span of the range
  float span = x_max - x_min;

  // max integer value for the given number of bits is (1 << num_bits) - 1
  int max_int = (1 << num_bits) - 1;

  // Scale safely so max maps to max_int (NOT 4096)
  int result = (int)((x - x_min) * ((float)max_int / span));

  // Extra safety (prevents any rounding overflow)
  if (result < 0) result = 0;
  if (result > max_int) result = max_int;

  return result;
}



/******************************************************************************************
  FUNCTION: Interpolates an unsigned integer of num_bits length to a floating point number
  between  x_min and x_max.

  ARGS:
    x (int): The int number to convert
    x_min (int): The minimum value for the floating point number
    x_max (int): The maximum value for the floating point number
    num_bits (int): The number of bits for the unsigned integer

  RETURNS:
    float: The floating point representation of the unsigned integer
******************************************************************************************/
float uint_to_float(uint16_t x_int, float x_min, float x_max, int num_bits) {

  float span = x_max - x_min;
  return float(x_int * span / ((1 << num_bits) - 1) + x_min);
}




/******************************************************************************************
  FUNCTION: Returns maximum of x, y
******************************************************************************************/
float fmaxf(float x, float y) {
  return (((x) > (y))?(x):(y));
}




/******************************************************************************************
  FUNCTION: Returns minimum of x, y
******************************************************************************************/
float fminf(float x, float y) {
  return (((x) < (y))?(x):(y));
}




/******************************************************************************************
  FUNCTION: To send a desired position, velocity, or torque command to a motor with
  set gains, Kp and Kd.
******************************************************************************************/
void Motor::pack_motor_message(float p_des, float v_des, float kp, float kd, float t_ff) {

  // Conversion to uint and clamping the variables to be in the correct limits based on the motor's specifications
  int p_int = float_to_uint(constrain(p_des, pos_lim_L, pos_lim_U), pos_lim_L, pos_lim_U, 16); // convert and clamp postion
  int v_int = float_to_uint(constrain(v_des, vel_lim_L, vel_lim_U), vel_lim_L, vel_lim_U, 12); // convert and clamp velocity
  int kp_int = float_to_uint(constrain(kp, kp_lim_L, kp_lim_U), kp_lim_L, kp_lim_U, 12); // convert and clamp gain Kp
  int kd_int = float_to_uint(constrain(kd, kd_lim_L, kd_lim_U), kd_lim_L, kd_lim_U, 12); // convert and clamp gain Kd
  int t_int = float_to_uint(constrain(t_ff, torque_lim_L, torque_lim_U), torque_lim_L, torque_lim_U, 12); // convert and clamp torque
  uint8_t buffer[8]; // Create a buffer of size 8 (we will be adding our inputs for positon, velocity, gains, and torque to this)

  /// pack ints into the can buffer //
  buffer[0] = kp_int >> 4;                              // KP high 8
  buffer[1] = ((kp_int & 0xF) << 4) | (kd_int >> 8);    // KP low 4 bits, Kd high bits
  buffer[2] = kd_int & 0xFF;                            // Kd low 8 bits
  buffer[3] = p_int >> 8;                               // position high 8 bits
  buffer[4] = p_int & 0xFF;                             // position low 8 bits
  buffer[5] = v_int >> 4;                               // speed high 8 bits
  buffer[6] = ((v_int & 0xF) << 4) | (t_int >> 8);      // speed low 4 bits, torque high 4 bits
  buffer[7] = t_int & 0xFF;                             // torque low 8 bits

  comm_can_transmit_eid(motor_id | ((uint32_t)CAN_PACKET_SET_MIT << 8), buffer, 8); // calls function to send can message with relavant information
}




/******************************************************************************************
  FUNCTION: From Motor Class we implement function for setting torque for ease of 
  use when we want to send a torque command to the motor
******************************************************************************************/
void Motor::set_torque (float pos_hold, float des_torque) {
  pack_motor_message(pos_hold, 0.0, 0.0, 0.0, des_torque);
}




/******************************************************************************************
  FUNCTION: From Motor Class we implement function for setting torque for ease of 
  use when we want to send a torque command to the motor
******************************************************************************************/
void Motor::set_speed (float des_speed, float kd) {
  pack_motor_message(0.0, des_speed, 0.0, kd, 0.0);
}



/******************************************************************************************
  FUNCTION: From Motor Class we implement function for setting torque for ease of 
  use when we want to send a torque command to the motor
******************************************************************************************/
void Motor::set_position (float des_position, float kp, float kd) {
  pack_motor_message(des_position, 0.0, kp, kd, 0.0);
}


/******************************************************************************************
  FUNCTION: To recieve motor command for an AK80-9 to assign its position,
  velocity, and torque to the corresponding variables.
******************************************************************************************/
void Motor::motor_receive(const CAN_message_t &msg) {

  // Assigning data buffer to corresponding variables
  // Consider buffer = [0x12 0x34 0x56 0x78 0x98 0x76]
  int16_t pos_int = (int16_t)((msg.buf[0] << 8) | msg.buf[1]); // puts together the first two 8 bit numbers: 0x12 | 0x34 = 0x1234
  int16_t spd_int = msg.buf[2] << 8 | msg.buf[3]; // Example: 0x56 | 0x78 = 0x5678
  int16_t cur_int = msg.buf[4] << 8 | msg.buf[5]; // Example: 0x98 | 0x76 = 0x9876
  pos = (float)(pos_int * 0.1f);      // converting to correct scale and assigning to position variable
  ERPM = (float)(spd_int * 10.0f);    // converting to correct scale and assigning to ERPM variable
  current = (float)(cur_int * 0.01f); // converting to correct scale and assigning to current variable
  temperature = msg.buf[6];           // assigning to temperature
  error_code = msg.buf[7];            // assigning to error code

  vel = (ERPM / (pole_pairs * gear_ratio)) * RPM_to_deg; // converting ERPM to output shaft velocity in deg/s
  torque = current * KT_constant;
}




/******************************************************************************************
  FUNCTION: From Motor Class fuction for printing Motors stored parameters to the
  Serial Monitor for debugging
******************************************************************************************/
void Motor::print_motor_data() {

  Serial.print(String("CAN ID: ") + motor_id + "| ");
  Serial.print(String("Error code: " + error_code) + "| ");
  Serial.print(String("Temperature: ") + temperature + " C| ");
  Serial.print(String("Position: ") + pos + " deg| ");
  Serial.print(String("ERPM_stator: ") + ERPM + " ERPM| ");
  Serial.print(String("Velocity_output: ") + vel + " Deg/s| ");
  Serial.print(String("Current: ") + current + " A| ");
  Serial.println(String("Torque: ") + torque + "Nm");
}



/******************************************************************************************
  FUNCTION: From Motor Class function for teleplotting motor position to Serial Monitor
******************************************************************************************/
void Motor::motors_teleplot() {
  Serial.print(">" + name + " pos (deg):");
  Serial.println(pos);
  Serial.print(">" + name + " torque (Nm):");
  Serial.println(torque);
}




float Motor::getPosition() {
  return pos;
}

float Motor::getVelocity() {
  return vel;
}


// END OF FILE