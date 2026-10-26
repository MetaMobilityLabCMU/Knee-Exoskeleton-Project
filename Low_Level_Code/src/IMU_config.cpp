/******************************************************************************************
  AUTHOR: Rajiv Joshi

  PURPOSE: For configuring SPI communication for Inertial Measurement Units (IMUs).

  - IMUs: ICM20948 from Adafruit x2

  NOTES: Speed of readings should be deterministic and consistent.
         - Frequency of about 250 Hz is minimum
         - IMUs max is around 1000 Hz

  UPDATED: 04/29/2026
******************************************************************************************/




/******************************************************************************************
                                Include/Declare Statements:
******************************************************************************************/

#include "IMU_config.h"

/******************************************************************************************
                                  Function Definitions:
******************************************************************************************/

void IMU::IMU_setup (int cs) {
    if(!this->icm.begin_SPI(cs, &SPI)) {  // hardware SPI version
        Serial.println("Failed to initialize IMU: " + Location);
        while (1) delay(10);
    }
    Serial.println("IMU Found!");

    this->icm.setAccelRange(ICM20948_ACCEL_RANGE_4_G);
    this->icm.setGyroRange(ICM20948_GYRO_RANGE_1000_DPS);
    this->icm.setAccelRateDivisor(0);                      // ~1125 Hz -->  1125 / (1.0 + accel_divisor)
    this->icm.setGyroRateDivisor(0);                       // ~1100 Hz -->  1100 / (1.0 + gyro_divisor)
    this->icm.setMagDataRate(AK09916_MAG_DATARATE_50_HZ);  // 50 Hz (Doesn't work for SPI)
}




void IMU::IMU_getData() {

    icm.getEvent(&accel, &gyro, &temp, &mag);

}




void IMU::print_IMU_data() {
    Serial.print(Location);
    Serial.print(" | Temp: ");
    Serial.print(temp.temperature);
    Serial.print(" °C |");

    Serial.print(" Accel (m/s^2): X= ");
    Serial.print(accel.acceleration.x, 2);
    Serial.print(", Y= ");
    Serial.print(accel.acceleration.y, 2);
    Serial.print(", Z= ");
    Serial.print(accel.acceleration.z, 2);

    Serial.print(" | Gyro (rad/s): X= ");
    Serial.print(gyro.gyro.x, 2);
    Serial.print(", Y= ");
    Serial.print(gyro.gyro.y, 2);
    Serial.print(", Z= ");
    Serial.println(gyro.gyro.z, 2);
}




void IMU::IMU_teleplot() {

    Serial.print(">" + Location + " Accel X (m/s^2):");
    Serial.println(accel.acceleration.x, 2);
    Serial.print(">" + Location + " Accel Y (m/s^2):");
    Serial.println(accel.acceleration.y, 2);
    Serial.print(">" + Location + " Accel Z (m/s^2):");
    Serial.println(accel.acceleration.z, 2);

    Serial.print(">" + Location + " Gyro X (rad/s):");
    Serial.println(gyro.gyro.x, 2);
    Serial.print(">" + Location + " Gyro Y (rad/s):");
    Serial.println(gyro.gyro.y, 2);
    Serial.print(">" + Location + " Gyro Z (rad/s):");
    Serial.println(gyro.gyro.z, 2);
}








