/******************************************************************************************
  AUTHOR: Ray Sarabu and Rajiv Joshi

  PURPOSE: Header file defines prototype functions for SD_CSV.cpp file.

  UPDATED: 11/20/2025
******************************************************************************************/




#ifndef SD_CSV_H
#define SD_CSV_H


/******************************************************************************************
                                  Include/Define Statements:
******************************************************************************************/

#include <SD.h>
#include <SPI.h>
#include "IMU_config.h"
#include "motors.h"

/******************************************************************************************
                                  Function Prototypes:
******************************************************************************************/
class DataFiles  {
  public:
    // Variables/Objects
    const char *FilePath;

    // Constructor
    DataFiles(const char* FilePath) {
      this->FilePath = FilePath;
    }


    // Funcitons
    void writeToFile(String info);
    void writeSensorData(float dataCollectionTimer, Motor Motor[], IMU IMUs[]);


  private:

};

void SD_setup();  // prototype to setup SD card for read/write

#endif // end of header file