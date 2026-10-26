/******************************************************************************************
  AUTHOR: Rajiv Joshi

  PURPOSE: For list of parameters please refrence the below link. This code serves to 
  store the paramters for most AK series motors that will be used for bounding variables
  in MIT Control mode.

  - CubeMars AK page: https://www.cubemars.com/category-155-AK+Series.html

  NOTES: Units of below limits: 
  - Position [rad]
  - Velocity [rad/s]
  - Torque [Nm]

  UPDATED: 11/20/2025
******************************************************************************************/

// Includes:
#include "motor_config.h"

std::unordered_map<std::string, MotorParams> motor_database = {
    // Key: "AK80-9", Value: MotorParams struct
    {"AK80-9", 
        { 
            {-12.56, 12.56},    // position_limits
            {-65.0, 65.0},      // velocity_limits
            {-18.0, 18.0},      // torque_limits
            {-9.0, 9.0},        // rated_torque_limits
            {0.0, 500.0},       // kp_limits
            {0.0, 5.0},         // kd_limits
            (9.0/1.0),          // gear_ratio
            (21.0),             // pole_pairs
            (0.5701)            // KT_constant = bus current KT * gear ratio * 2/3 FOC constant
        }
    },
    // Key: "AKE60-8", Value: MotorParams struct
    {"AKE60-8", 
        { 
            {-12.56, 12.56},    // position_limits
            {-40.0, 40.0},      // velocity_limits
            {-12.5, 12.5},      // torque_limits
            {-5.0, 5.0},        // rated_torque_limits
            {0.0, 500.0},       // kp_limits
            {0.0, 5.0},         // kd_limits
            (8.0/1.0),          // gear_ratio
            (14.0),             // pole_pairs
            (0.35)            // KT_constant = bus current KT * gear ratio * 2/3 FOC constant
        }
    }
};



// END OF FILE