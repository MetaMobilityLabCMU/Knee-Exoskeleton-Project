/******************************************************************************************
  AUTHOR: Ray Sarabu and Rajiv Joshi

  PURPOSE: For writing IMU and Motor data to an SD card via Teensy 4.1.

  NOTES: Sample material for Teensy 4.1 SD card library in Examples SD card

  UPDATED: 11/20/2025
******************************************************************************************/




/******************************************************************************************
                                Include/Declare Statements:
******************************************************************************************/

#include "SD_CSV.h"

/******************************************************************************************
                                  Function Definitions:
******************************************************************************************/

void SD_setup() {

  // Attempt to initialize SD card
  if (!SD.begin(BUILTIN_SDCARD)) {    // for teensy 4.1 SPI CS pin for SD is 254, defined in SD.h header
    Serial.println("SD card initializatioN: FAILED...");
    while (1);  // Halt if SD fails
  }
  Serial.println("SD card initialization: SUCCESS...");

}




void DataFiles::writeToFile(String info) {
  File dataFile = SD.open(FilePath, FILE_WRITE);
  if (dataFile) {
    dataFile.println(info);
    dataFile.close();
  }
}




void DataFiles::writeSensorData(float dataCollectionTimer, Motor Motors[], IMU IMUs[]) {

  File dataFile =  SD.open(FilePath, FILE_WRITE);

  if (dataFile) {
    
    // Timer for data collection
    dataFile.print(dataCollectionTimer);
    dataFile.print(",");

    // Right Knee Motor
    dataFile.print(String(Motors[0].pos));
    dataFile.print(",");

    // Thigh IMU
    dataFile.print(String(IMUs[0].accel.acceleration.x));
    dataFile.print(",");
    dataFile.print(String(IMUs[0].accel.acceleration.y));
    dataFile.print(",");
    dataFile.print(String(IMUs[0].accel.acceleration.z));
    dataFile.print(",");
    dataFile.print(String(IMUs[0].gyro.gyro.x));
    dataFile.print(",");
    dataFile.print(String(IMUs[0].gyro.gyro.y));
    dataFile.print(",");
    dataFile.print(String(IMUs[0].gyro.gyro.z));
    dataFile.print(",");

    // Shank IMU
    dataFile.print(String(IMUs[1].accel.acceleration.x));
    dataFile.print(",");
    dataFile.print(String(IMUs[1].accel.acceleration.y));
    dataFile.print(",");
    dataFile.print(String(IMUs[1].accel.acceleration.z));
    dataFile.print(",");
    dataFile.print(String(IMUs[1].gyro.gyro.x));
    dataFile.print(",");
    dataFile.print(String(IMUs[1].gyro.gyro.y));
    dataFile.print(",");
    dataFile.println(String(IMUs[1].gyro.gyro.z));
    dataFile.close();

  }   // End of if

}




//  END-OF-FILE