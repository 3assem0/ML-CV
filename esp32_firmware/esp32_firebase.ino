#include <WiFi.h>
#include <Firebase_ESP_Client.h>
#include "HX711.h"

// Provide the token generation process info.
#include "addons/TokenHelper.h"
// Provide the RTDB payload printing info and other helper functions.
#include "addons/RTDBHelper.h"

// ==========================================
// 1. WIFI & FIREBASE CONFIGURATION
// ==========================================
#define WIFI_SSID "YOUR_WIFI_SSID"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"

// Insert Firebase project API Key
#define API_KEY "YOUR_FIREBASE_WEB_API_KEY"

// Insert RTDB URL (e.g. https://archologestdb-default-rtdb.firebaseio.com/)
#define DATABASE_URL "https://archologestdb-default-rtdb.firebaseio.com/" 

// Define Firebase Data objects
FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

// ==========================================
// 2. HARDWARE PINS CONFIGURATION
// ==========================================

// --- LOAD CELLS (HX711) ---
const int DOUT_PIN_1 = 16;
const int SCK_PIN_1 = 4;
const int DOUT_PIN_2 = 17;
const int SCK_PIN_2 = 5;
const int DOUT_PIN_3 = 18;
const int SCK_PIN_3 = 19;

HX711 scale1;
HX711 scale2;
HX711 scale3;

// Calibration factors (Adjust these based on your specific 5kg load cells)
float CALIBRATION_FACTOR_1 = 228.0f; 
float CALIBRATION_FACTOR_2 = 228.0f;
float CALIBRATION_FACTOR_3 = 228.0f;

// --- DC MOTORS (L298 Driver) ---
// Assuming standard ENA/IN1/IN2 style control. 
// For simple ON/OFF we just use IN1 and IN2
const int MOTOR1_IN1 = 13;
const int MOTOR1_IN2 = 12;

const int MOTOR2_IN1 = 14;
const int MOTOR2_IN2 = 27;

const int MOTOR3_IN1 = 26;
const int MOTOR3_IN2 = 25;

// --- VIBRATION MOTORS (MOSFETs) ---
// We control all 3 vibration motors together using 1 signal pin connected to the MOSFET gates
const int VIBRATION_PIN = 33;


// ==========================================
// 3. GLOBAL VARIABLES
// ==========================================
unsigned long sendDataPrevMillis = 0;
// Interval to upload weights (every 1 second)
unsigned long timerDelay = 1000;

bool signupOK = false;

// ==========================================
// SETUP
// ==========================================
void setup() {
  Serial.begin(115200);

  // --- Initialize Motors ---
  pinMode(MOTOR1_IN1, OUTPUT);
  pinMode(MOTOR1_IN2, OUTPUT);
  pinMode(MOTOR2_IN1, OUTPUT);
  pinMode(MOTOR2_IN2, OUTPUT);
  pinMode(MOTOR3_IN1, OUTPUT);
  pinMode(MOTOR3_IN2, OUTPUT);
  pinMode(VIBRATION_PIN, OUTPUT);

  // Ensure everything is OFF initially
  digitalWrite(MOTOR1_IN1, LOW);
  digitalWrite(MOTOR1_IN2, LOW);
  digitalWrite(MOTOR2_IN1, LOW);
  digitalWrite(MOTOR2_IN2, LOW);
  digitalWrite(MOTOR3_IN1, LOW);
  digitalWrite(MOTOR3_IN2, LOW);
  digitalWrite(VIBRATION_PIN, LOW);

  // --- Initialize Load Cells ---
  scale1.begin(DOUT_PIN_1, SCK_PIN_1);
  scale1.set_scale(CALIBRATION_FACTOR_1);
  scale1.tare();

  scale2.begin(DOUT_PIN_2, SCK_PIN_2);
  scale2.set_scale(CALIBRATION_FACTOR_2);
  scale2.tare();

  scale3.begin(DOUT_PIN_3, SCK_PIN_3);
  scale3.set_scale(CALIBRATION_FACTOR_3);
  scale3.tare();

  // --- Connect to Wi-Fi ---
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(300);
  }
  Serial.println();
  Serial.print("Connected with IP: ");
  Serial.println(WiFi.localIP());
  Serial.println();

  // --- Initialize Firebase ---
  config.api_key = API_KEY;
  config.database_url = DATABASE_URL;

  // Sign up anonymously (No email/password required for Test Mode)
  if (Firebase.signUp(&config, &auth, "", "")) {
    Serial.println("Firebase Auth Setup Successful");
    signupOK = true;
  } else {
    Serial.printf("Firebase error: %s\n", config.signer.signupError.message.c_str());
  }

  // Assign the callback function for the long running token generation task
  config.token_status_callback = tokenStatusCallback; 
  
  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
}

// ==========================================
// LOOP
// ==========================================
void loop() {
  if (Firebase.ready() && signupOK) {

    // ---------------------------------------------------------
    // 1. READ FROM FIREBASE (Update Motor States)
    // ---------------------------------------------------------

    // Read DC Motor 1 State
    if (Firebase.RTDB.getBool(&fbdo, "/controls/dc_motor1")) {
      bool state = fbdo.boolData();
      digitalWrite(MOTOR1_IN1, state ? HIGH : LOW);
      digitalWrite(MOTOR1_IN2, LOW); // Assumes unidirectional spinning
    }

    // Read DC Motor 2 State
    if (Firebase.RTDB.getBool(&fbdo, "/controls/dc_motor2")) {
      bool state = fbdo.boolData();
      digitalWrite(MOTOR2_IN1, state ? HIGH : LOW);
      digitalWrite(MOTOR2_IN2, LOW);
    }

    // Read DC Motor 3 State
    if (Firebase.RTDB.getBool(&fbdo, "/controls/dc_motor3")) {
      bool state = fbdo.boolData();
      digitalWrite(MOTOR3_IN1, state ? HIGH : LOW);
      digitalWrite(MOTOR3_IN2, LOW);
    }

    // Read Vibration Motors State
    if (Firebase.RTDB.getBool(&fbdo, "/controls/vibration")) {
      bool state = fbdo.boolData();
      digitalWrite(VIBRATION_PIN, state ? HIGH : LOW);
    }

    // ---------------------------------------------------------
    // 2. WRITE TO FIREBASE (Upload Weights Every X Seconds)
    // ---------------------------------------------------------
    if (millis() - sendDataPrevMillis > timerDelay || sendDataPrevMillis == 0) {
      sendDataPrevMillis = millis();
      
      // Read weights (average of 3 readings for speed, or 10 for stability)
      float weight1 = scale1.get_units(3);
      float weight2 = scale2.get_units(3);
      float weight3 = scale3.get_units(3);

      // Write to Firebase
      Firebase.RTDB.setFloatAsync(&fbdo, "/weights/area1", weight1);
      Firebase.RTDB.setFloatAsync(&fbdo, "/weights/area2", weight2);
      Firebase.RTDB.setFloatAsync(&fbdo, "/weights/area3", weight3);
      
      Serial.printf("Weights: A1:%.2f  A2:%.2f  A3:%.2f\n", weight1, weight2, weight3);
    }
  }
}
