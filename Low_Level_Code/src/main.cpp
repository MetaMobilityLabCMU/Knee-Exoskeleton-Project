/******************************************************************************************
  AUTHOR: Rajiv Joshi

  PURPOSE: Main file for setup and loop

  VERSION: 3

  UPDATED: 03/27/2026
******************************************************************************************/

/******************************************************************************************
                                Include/Declare Statements:
******************************************************************************************/

#include <Arduino.h>
#include <FlexCAN_T4.h>
#include <IntervalTimer.h>

#include "motors.h"
#include "IMU_config.h"
#include "SD_CSV.h"
#include "Jetson_yap.h"

#include "IMPD_CTRL.h"

FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> can1;  // uses CAN 1 on Teensy (CTX1 and CRX1)
CAN_message_t msg;

FlexCAN_T4FD<CAN3, RX_SIZE_256, TX_SIZE_16> can3FD;
CANFD_message_t msg_fd;

// Teensy ID for CAN bus
uint32_t Teensy_ID = 0x100; // Base ID for this Teensy controller

// Motor Parameters
uint8_t motor_1_CAN_ID = 0x04; // CAN ID for the motor (e.g., 0x04 for motor 1)
String motor_1_name = "Left_Knee_Motor"; // Name of the motor for identification
const char* motor_1_type = "AK80-9"; // Type of the motor for database lookup

// Deterministic Loop Variables
IntervalTimer loopTimer;              // Timer for deterministic loop
volatile bool det_loop_flag = false;  // flag for deterministic loop
unsigned long currentTime = 0;        // variable to hold current time in microseconds
unsigned long previousTime = 0;       // variable to hold previous time in microseconds
unsigned long elapsedTime = 0;        // variable to hold elapsed time between loops in microseconds

// Watch dog Timer for detecting if loop is taking too long (e.g., due to a bug or unexpected delay)
unsigned long lastTorqueMsgTime = 0;
unsigned long watchdogTimeout = 100; // 100 ms or 10 Hz timeout for watchdog


// Creating motor objects
Motor Motor_1(motor_1_name, motor_1_CAN_ID, motor_1_type); // Motor 1 object created
Motor Motors[1] = {Motor_1};            // Array for holding all motors
size_t num_Motors = sizeof(Motors) / sizeof(Motors[0]);


// Creating IMUs
IMU Thigh_IMU("Thigh_IMU");
IMU Shank_IMU("Shank_IMU");
IMU IMUs[2] = {Thigh_IMU, Shank_IMU};


//  SPI pins for Teensy 4.1
#define ICM1_CS 9   // Thigh CS pin
#define ICM2_CS 32  // Shank CS pin
// ICM_SCK 13
// ICM_MISO 12
// ICM_MOSI 11


DataFiles Knee_Exo_Data("Knee_Exo_Data.csv");


// Control Variables
float setLoopFrequency = 1000.0f; // Desired loop frequency in Hz
volatile float commandedTorque = 0.0f;
float safeTorque = 0.0f;

// AK Motor Impedance Controller Testing
// IMPD_Controller impdController(0.05f, 0.002f, 0.0f);
// IMPD_Controller impdController(0.1f, 0.003f, 0.0f);

// AKE Motor Impedance Controller Testing
// IMPD_Controller impdController(0.08f, 0.0015f, 0.0f);
// IMPD_Controller impdController(0.00f, 0.005f, 0.0f);


/******************************************************************************************
  FUNCTION: Depending on message ID that is recieved we want to call the correct function 
  to unpack and read that message.
******************************************************************************************/
void canMessageReceived(const CAN_message_t &msg) {
  for (size_t i = 0; i < num_Motors; ++i) {
    if (msg.id == (uint32_t)(0x2900 + Motors[i].motor_id)) {
      Motors[i].motor_receive(msg);
      break;
    }
  }
}




/******************************************************************************************
  FUNCTION: When we receive a CAN message on CAN3 bus (mostly from jetson)
******************************************************************************************/
void JetsonMessageReceived(const CANFD_message_t &msg) {
  if (msg.id == Teensy_ID + 0x002) {
    // Serial.print("Received reply from Jetson, ID=0x");
    // Serial.println(msg.id, HEX);
    send_data_all(Motors, IMUs);
  }
  if (msg.id == Teensy_ID + 0x003) {
    // Assuming the torque value is sent as a float in the first 4 bytes of the message
    if (msg.len >= 4) { // Check if we have at least 4 bytes to read a float
      float receivedTorque;
      memcpy(&receivedTorque, msg.buf, sizeof(float));

      if (!isfinite(receivedTorque)) {
        commandedTorque = 0.0f;
      } 
      else {
        commandedTorque = constrain(receivedTorque, Motors[0].torque_lim_L, Motors[0].torque_lim_U); // Constrain the received torque to motor limits
      }
      // Update the last time we received a torque command for watchdog purposes  
      lastTorqueMsgTime = millis();

      // Serial.print("Received torque command from Jetson: ");
      // Serial.println(receivedTorque);
    }

  }
}




/******************************************************************************************
  FUNCTION: Timer ISR - sets the deterministic loop flag, this function should be as short
  and fast as possible to avoid delaying other interrupts. So no heavy processing should be
  done here or printstatements.

  Basically when the timer overflows it will call this interrupt service routine
******************************************************************************************/
void timerISR() {
  det_loop_flag = true;
}


void setup() {

  delay(4000); // wait for 4 seconds before starting setup (since motors need to initialize first)

  // Serial Communication Setup
  // while (!Serial) delay(10); // will pause till Serial console opens
  Serial.begin(115200);     // specify baudrate
  // Serial.println("--------------------------------------------------------------------");
  // Serial.println("Serial Baud Rate = 115200");
  // Serial.println("Serial communcaiton setup COMPLETE...");
  delay(100);    // small delay [100 ms]


  // Setup motor parameters from database
  // Serial.println("--------------------------------------------------------------------");
  // Serial.println("Initializing motor parameters from database...");
    // Initialize motor parameters from database after all globals are constructed
  for (size_t i = 0; i < num_Motors; ++i) {
    Motors[i].initFromDatabase();
  }
  // Serial.println("Motor parameters initialization COMPLETE...");
  // delay(2000);    // small delay [2000 ms]


  // Setting up SD card and file for CSV data saving
  // Serial.println("--------------------------------------------------------------------");
  // Serial.println("Starting setup for SD card...");
  // SD_setup();
  // Serial.println("SD card setup COMPLETE...");
  // delay(100);    // small delay [2000 ms]
  

  // header for CSV file
  // Knee_Exo_Data.writeToFile(

  //   String("Time (sec)") + ","
    
  //   + Motors[0].name + " pos (deg)" + ","

  //   + IMUs[0].Location + " Accel X (m/s^2)" + ","
  //   + IMUs[0].Location + " Accel Y (m/s^2)" + ","
  //   + IMUs[0].Location + " Accel Z (m/s^2)" + ","
  //   + IMUs[0].Location + " Gyro X (rad/s)" + ","
  //   + IMUs[0].Location + " Gyro Y (rad/s)" + ","
  //   + IMUs[0].Location + " Gyro Z (rad/s)" + ","

  //   + IMUs[1].Location + " Accel X (m/s^2)" + ","
  //   + IMUs[1].Location + " Accel Y (m/s^2)" + ","
  //   + IMUs[1].Location + " Accel Z (m/s^2)" + ","
  //   + IMUs[1].Location + " Gyro X (rad/s)" + ","
  //   + IMUs[1].Location + " Gyro Y (rad/s)" + ","
  //   + IMUs[1].Location + " Gyro Z (rad/s)"

  // );
  // delay(100);    // small delay [2000 ms]


  // Getting Data for Motor Type
  // Serial.println("--------------------------------------------------------------------");
  // Serial.println(String("Motor 1 type: ") + Motors[0].motor_type);
  // Serial.println(String("Motor 2 type: ") + Motors[1].motor_type);
  // delay(2000);    // small delay [1000 ms]


  // Getting Data for Motor CAN ID
  // Serial.println("--------------------------------------------------------------------");
  // Serial.println(String("Motor 1 CAN ID: ") + Motors[0].motor_id);
  // Serial.println(String("Motor 2 CAN ID: ") + Motors[1].motor_id);
  // delay(2000);    // small delay [1000 ms]


  // CAN1 Communication Setup
  // Serial.println("--------------------------------------------------------------------");
  // Serial.println("Setting up CAN communication...");
  can1.begin();
  can1.setBaudRate(1000000);  // 1Mbps
  delay(100);    // small delay [1000 ms]
  // Enable FIFO Interrupts 
  can1.enableFIFO();                      // enable First in First Out
  can1.enableFIFOInterrupt();             // Enable Interrupts
  can1.onReceive(canMessageReceived);     // When we interrupt on a revieced message from other devices
  can1.mailboxStatus();                   // setting up all mailboxes (look more into how mailboxes work for FlexCAN library)
  // Serial.println("CAN communication setup COMPLETE...");
  delay(100);    // small delay [100 ms]


  // CAN3FD Communication Setup
  // Serial.println("--------------------------------------------------------------------");
  // Serial.println("Setting up CANFD communication...");
  can3FD.begin();
  CANFD_timings_t config;
  config.clock = CLK_24MHz;
  config.baudrate = 1000000;
  config.baudrateFD = 2000000;
  config.propdelay = 190;
  config.bus_length = 1;
  config.sample = 70;
  can3FD.setBaudRate(config);
  can3FD.setRegions(64);    // basically lets each mailbox hold 64 bytes worth of data
  can3FD.setMB(MB1, TX);    // TX mailbox for sending messages to the Jetson
  can3FD.setMB(MB0, RX);    // RX mailbox for receiving messages from the Jetson
  // basically whenever the Jetson sends a CAN message to the teensy we receive it to this mailbox and handle it with the JetsonMessageReceived function
  can3FD.onReceive(MB0, JetsonMessageReceived);
  // in order for the .onReceive function to be called, we need to enable interrupts for the mailbox
  can3FD.enableMBInterrupts(true);
  // Serial.println("CANFD communication setup COMPLETE...");
  delay(100);    // small delay [100 ms]


  pinMode(ICM1_CS, OUTPUT);
  pinMode(ICM2_CS, OUTPUT);
  digitalWrite(ICM1_CS, HIGH);
  digitalWrite(ICM2_CS, HIGH);


  // IMU SPI Communication Setup
  Serial.println("--------------------------------------------------------------------");
  Serial.println("Setting up IMU 1 SPI communication...");
  IMUs[0].IMU_setup(ICM1_CS);
  Serial.println("Setting up IMU 2 SPI communication...");
  IMUs[1].IMU_setup(ICM2_CS);
  Serial.println("IMU SPI communication setup COMPLETE...");
  delay(100);    // small delay [100 ms]


  // Clear current Motor data/inforamtion
  comm_can_set_origin(Motors[0].motor_id, 0);
  // comm_can_set_origin(Motors[1].motor_id, 0);
  // for (size_t j = 0; j < (sizeof(Motors)/sizeof(Motors[0])); j++) { // looping through all motors to check if cleared
  //   Serial.println("--------------------------------------------------------------------");
  //   Serial.println( String("Motor ") + j + ": pos = " + Motors[j].pos + " [rad], vel = " + Motors[j].ERPM + " [rad/s], torque = " + Motors[j].current + " [Nm]");
  //   Serial.println(String("Clearing Motor ") + j + " data COMPLETE...");
  //   delay(2000);    // small delay [2000 ms]
  // }


  // Flush any stale CAN messages before starting main loop
  // Serial.println("Flushing stale CAN messages...");
  unsigned long flushStart = millis();
  while (millis() - flushStart < 1000) { // Read for 1000ms
    can3FD.events(); // Process any pending messages
    // Force variable back to 0 immediately after processing
    commandedTorque = 0.0f; 
  }


  // Starting message countdown
  // Serial.println("--------------------------------------------------------------------");
  // Serial.println("Starting in 3...");
  // flash Teensy LED to indicate countdown to start
  // pinMode(LED_BUILTIN, OUTPUT);
  // digitalWrite(LED_BUILTIN, HIGH);  // turn the LED on (HIGH is the voltage level)
  // delay(500);
  // digitalWrite(LED_BUILTIN, LOW);   // turn the LED off by making the voltage LOW
  // delay(500);

  // // Serial.println("2...");
  // digitalWrite(LED_BUILTIN, HIGH);  // turn the LED on (HIGH is the voltage level)
  // delay(500);
  // digitalWrite(LED_BUILTIN, LOW);   // turn the LED off by making the voltage LOW
  // delay(500);

  // // Serial.println("1...");
  // digitalWrite(LED_BUILTIN, HIGH);  // turn the LED on (HIGH is the voltage level)
  // delay(500);
  // digitalWrite(LED_BUILTIN, LOW);   // turn the LED off by making the voltage LOW
  // delay(500);

  // convert desired loop frequency to microseconds period
  float loopPeriodMicrosec = 1000000.0f / setLoopFrequency;

  // Setting up and Starting Deterministic Loop Timer
  if (!loopTimer.begin(timerISR, static_cast<unsigned long>(loopPeriodMicrosec))) {
    // Serial.println("Failed to start timer!");
    while (1); // halt if timer fails to start
  }
  
  // Record current time and set previous time as same for first loop
  currentTime = micros();
  previousTime = currentTime;



} // end of setup





void loop() {

  // Checking for new CANFD messages
  can3FD.events();

  // Check if the deterministic loop flag is set
  if (det_loop_flag) {
    noInterrupts();           // Disable interrupts during deterministic loop
    det_loop_flag = false;    // Reset flag
    float position_snapshot = Motors[0].getPosition();
    interrupts();             // Re-enable interrupts

    // Watchdog check: if we haven't received a new torque command within the timeout period, set torque to 0 for safety
    if (millis() - lastTorqueMsgTime > watchdogTimeout) {
      safeTorque = 0.0f; // Set torque to 0 if we haven't received a command recently
    }
    else {
      safeTorque = commandedTorque; // Use the last commanded torque if it's still within the timeout period
    }

    // Loop Code Start

    // Collect current IMU Data
    IMUs[0].IMU_getData();
    IMUs[1].IMU_getData();

    // Impedance Control Computation
    // impdController.des_position = 0.0f;
    // safeTorque = constrain(impdController.compute(Motors[0]), Motors[0].torque_lim_L, Motors[0].torque_lim_U); // Constrain the computed torque to motor limits

    // Apply Torque to Motor Data
    Motors[0].set_torque(position_snapshot, safeTorque);

    // Print Parameters
    // Motors[0].print_motor_data();
    // Motors[0].motors_teleplot();
    // IMUs[0].IMU_teleplot();
    // IMUs[1].IMU_teleplot();


    // Calculate frequency loop is running at
    currentTime = micros();
    elapsedTime = currentTime - previousTime;
    previousTime = currentTime;
    float loopFrequency = 1000000.0f / static_cast<float>(elapsedTime);   // covert from microsec to Hz
    // Serial.println("Loop Frequency: " + String(loopFrequency) + " Hz");
  } // end of deterministic loop check
  

} // end of main loop
