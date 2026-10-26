/******************************************************************************************
  AUTHOR: Rajiv Joshi

  PURPOSE: Header file defines prototype functions for motor_config.cpp file.

  UPDATED: 11/20/2025
******************************************************************************************/
#ifndef MOTOR_CONFIG_H
#define MOTOR_CONFIG_H

// Includes:
#include <string>
#include <unordered_map>
#include <array>



/******************************************************************************************
                                  Data Strucutres:
******************************************************************************************/
struct MotorParams { // Structure of motor parameters
  // Members that motors can have
  std::array<float, 2> position_limits;     // {min, max}
  std::array<float, 2> velocity_limits;     // {min, max}
  std::array<float, 2> torque_limits;       // {min, max}
  std::array<float, 2> rated_torque_limits; // {min, max}
  std::array<float, 2> kp_limits;           // {min, max}
  std::array<float, 2> kd_limits;           // {min, max}

  float gear_ratio;             // gear ratio
  float pole_pairs;             // motor pole pairs
  float KT_constant;            // torque constant
};  // must end struct with ;


// Declare the map as extern in the header
extern std::unordered_map<std::string, MotorParams> motor_database;



#endif // end of header file



