#include <WiFi.h>
#include <Firebase_ESP_Client.h>
#include "HX711.h"
#include <time.h>

// Provide the token generation process info.
#include "addons/TokenHelper.h"
// Provide the RTDB payload printing info and other helper functions.
#include "addons/RTDBHelper.h"

// ================= WIFI & FIREBASE CONFIGURATION =================
#define WIFI_SSID "DCA&TFA"
#define WIFI_PASSWORD "01007227792"

// Insert Firebase project API Key
#define API_KEY "AIzaSyC0oZR5teyYHUgzXvC7Iy0ZWUru9kpFlv4"
// Insert RTDB URL
#define DATABASE_URL "https://cv-ml-4b693-default-rtdb.firebaseio.com" 

// Firebase Data objects
FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

// ================= HARDWARE PINS CONFIGURATION =================

// --- MOTORS (L298 Driver) ---
const int M1_IN1 = 12;
const int M1_IN2 = 14;
const int M2_IN1 = 27;
const int M2_IN2 = 26;

// --- VIBRATION (MOSFETs) ---
const int VIB_1 = 32;
const int VIB_2 = 33;
const int VIB_3 = 25;

// --- LOAD CELLS (HX711) ---
const int DOUT_PIN_1 = 16;
const int SCK_PIN_1 = 4;
const int DOUT_PIN_2 = 17;
const int SCK_PIN_2 = 5;
const int DOUT_PIN_3 = 18;
const int SCK_PIN_3 = 19;

HX711 scale1, scale2, scale3;
float CALIBRATION_FACTOR = 228.0f;

// ================= GLOBAL VARIABLES =================
unsigned long sendDataPrevMillis = 0;
unsigned long timerDelay = 1000; // Upload weights every 1 second
bool signupOK = false;

// ================= TIME SYNC =================
void syncTime() {
  configTime(0, 0, "pool.ntp.org", "time.nist.gov");
  time_t now = time(nullptr);
  Serial.print("Syncing time");
  while (now < 8 * 3600 * 2) {
    delay(500);
    now = time(nullptr);
    Serial.print(".");
  }
  Serial.println("\nTime Synced");
}

// ================= MOTOR HELPER =================
void motorControl(int in1, int in2, String dir) {
  if (dir == "forward") {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
  }
  else if (dir == "backward") {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
  }
  else { // "stop" or other
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
  }
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);

  // --- Initialize Actuators ---
  pinMode(M1_IN1, OUTPUT);
  pinMode(M1_IN2, OUTPUT);
  pinMode(M2_IN1, OUTPUT);
  pinMode(M2_IN2, OUTPUT);
  pinMode(VIB_1, OUTPUT);
  pinMode(VIB_2, OUTPUT);
  pinMode(VIB_3, OUTPUT);

  digitalWrite(M1_IN1, LOW);
  digitalWrite(M1_IN2, LOW);
  digitalWrite(M2_IN1, LOW);
  digitalWrite(M2_IN2, LOW);
  digitalWrite(VIB_1, LOW);
  digitalWrite(VIB_2, LOW);
  digitalWrite(VIB_3, LOW);

  // --- Initialize Load Cells ---
  scale1.begin(DOUT_PIN_1, SCK_PIN_1);
  scale1.set_scale(CALIBRATION_FACTOR);
  scale1.tare();

  scale2.begin(DOUT_PIN_2, SCK_PIN_2);
  scale2.set_scale(CALIBRATION_FACTOR);
  scale2.tare();

  scale3.begin(DOUT_PIN_3, SCK_PIN_3);
  scale3.set_scale(CALIBRATION_FACTOR);
  scale3.tare();

  // --- Connect to Wi-Fi ---
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi OK");

  syncTime();

  // --- Initialize Firebase ---
  config.api_key = API_KEY;
  config.database_url = DATABASE_URL;
  config.max_token_generation_retry = 10;

  if (Firebase.signUp(&config, &auth, "", "")) {
    Serial.println("Firebase Auth Setup Successful");
    signupOK = true;
  } else {
    Serial.printf("Firebase error: %s\n", config.signer.signupError.message.c_str());
  }

  config.token_status_callback = tokenStatusCallback; 
  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);

  Serial.println("Firebase Ready");
}

// ================= LOOP =================
void loop() {
  if (Firebase.ready() && signupOK) {

    // ---------------------------------------------------------
    // 1. READ ALL CONTROLS (Efficient JSON Fetch)
    // ---------------------------------------------------------
    if (Firebase.RTDB.getJSON(&fbdo, "/controls")) {
      FirebaseJson &json = fbdo.jsonObject();
      FirebaseJsonData v;

      String m1_cmd = "stop";
      String m2_cmd = "stop";
      bool vib_cmd = false;

      if (json.get(v, "dc_motor1") && v.type == "string") m1_cmd = v.stringValue;
      if (json.get(v, "dc_motor2") && v.type == "string") m2_cmd = v.stringValue;
      if (json.get(v, "vibration")) vib_cmd = v.boolValue;

      // Apply Motors
      motorControl(M1_IN1, M1_IN2, m1_cmd);
      motorControl(M2_IN1, M2_IN2, m2_cmd);

      // Apply Vibration
      digitalWrite(VIB_1, vib_cmd);
      digitalWrite(VIB_2, vib_cmd);
      digitalWrite(VIB_3, vib_cmd);
    }

    // ---------------------------------------------------------
    // 2. WRITE WEIGHTS TO FIREBASE (Every X seconds)
    // ---------------------------------------------------------
    if (millis() - sendDataPrevMillis > timerDelay || sendDataPrevMillis == 0) {
      sendDataPrevMillis = millis();
      
      float w1 = scale1.get_units(3);
      float w2 = scale2.get_units(3);
      float w3 = scale3.get_units(3);

      Firebase.RTDB.setFloatAsync(&fbdo, "/weights/area1", w1);
      Firebase.RTDB.setFloatAsync(&fbdo, "/weights/area2", w2);
      Firebase.RTDB.setFloatAsync(&fbdo, "/weights/area3", w3);
      
      Serial.printf("Weights: A1:%.2f  A2:%.2f  A3:%.2f\n", w1, w2, w3);
    }
  }
}
