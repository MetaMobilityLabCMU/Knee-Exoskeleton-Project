/******************************************************************************************
  AUTHOR: Rajiv Joshi

  PURPOSE: Header file defines prototype functions for IMU_config.cpp file.

  UPDATED: 04/29/2026
******************************************************************************************/




#ifndef IMU_CONFIG_H
#define IMU_CONFIG_H


/******************************************************************************************
                                  Include/Define Statements:
******************************************************************************************/

#include <Adafruit_ICM20X.h>
#include <Adafruit_ICM20948.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

// extern Adafruit_ICM20948 icm1;
// extern Adafruit_ICM20948 icm2;

class IMU {
  public:
    // Variables/Objects
    String Location;
    Adafruit_ICM20948 icm;
    sensors_event_t accel, gyro, mag, temp;

    // Constructor
    IMU(String Location) {
      this->Location = Location;
    };

    // Functions
    void IMU_setup(int cs);
    void IMU_getData();
    void IMU_teleplot();
    void print_IMU_data();


  private:

};
/******************************************************************************************
                                  Function Prototypes:
******************************************************************************************/

#endif // end of header file