
#include "IMPD_CTRL.h"

float IMPD_Controller::compute(Motor &motor) {
    // Get current position and velocity from the motor
    float current_position = motor.getPosition();
    float current_velocity = motor.getVelocity();

    // Calculate position and velocity errors
    float position_error = des_position - current_position;
    // Compute the IMPD control output
    float cmd_torque = (k_spring * position_error) - (d_damper * current_velocity);

    return cmd_torque;
}