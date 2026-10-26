/******************************************************************************************
  AUTHOR: Rajiv Joshi

  PURPOSE: Header file defines prototype functions for IMPD_CTRL.cpp file.

  UPDATED: 11/24/2025
******************************************************************************************/




#ifndef IMPD_CTRL_H
#define IMPD_CTRL_H

#include "motors.h"

/******************************************************************************************
                                  Function Prototypes:
******************************************************************************************/

class IMPD_Controller {
    public:
        // Variables/Objects
        float k_spring;     // spring constant
        float d_damper;     // damper constant
        float des_position; // desired position
        float des_velocity = 0.0f; // desired velocity    
        // Constructor
        IMPD_Controller(float k_spring, float d_damper, float des_position = 0.0f) {
            this->k_spring = k_spring;
            this->d_damper = d_damper;
            this->des_position = des_position;
        }
    
        // Functions
        float compute(Motor &motor);
    
    private:
};




# endif // end of header file