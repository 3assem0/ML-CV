#include <WiFi.h>
#include <WebServer.h>
#include "HX711.h"

// WiFi Credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Initialize WebServer on port 80
WebServer server(80);

// Load Cell 1 (Area 1) Pins
const int DOUT_PIN_1 = 16;
const int SCK_PIN_1 = 4;

// Load Cell 2 (Area 2) Pins
const int DOUT_PIN_2 = 17;
const int SCK_PIN_2 = 5;

// Load Cell 3 (Area 3) Pins
const int DOUT_PIN_3 = 18;
const int SCK_PIN_3 = 19;

// Initialize HX711 instances
HX711 scale1;
HX711 scale2;
HX711 scale3;

// Calibration factors (Adjust these based on your specific load cells)
float CALIBRATION_FACTOR_1 = 2280.f; // Example value
float CALIBRATION_FACTOR_2 = 2280.f; // Example value
float CALIBRATION_FACTOR_3 = 2280.f; // Example value

void setup() {
  Serial.begin(115200);

  // Initialize Scales
  scale1.begin(DOUT_PIN_1, SCK_PIN_1);
  scale1.set_scale(CALIBRATION_FACTOR_1);
  scale1.tare();

  scale2.begin(DOUT_PIN_2, SCK_PIN_2);
  scale2.set_scale(CALIBRATION_FACTOR_2);
  scale2.tare();

  scale3.begin(DOUT_PIN_3, SCK_PIN_3);
  scale3.set_scale(CALIBRATION_FACTOR_3);
  scale3.tare();

  // Connect to WiFi
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected.");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // Define HTTP Endpoint
  server.on("/weights", HTTP_GET, []() {
    float weight1 = scale1.get_units(10); // Average of 10 readings
    float weight2 = scale2.get_units(10);
    float weight3 = scale3.get_units(10);

    String jsonResponse = "{";
    jsonResponse += "\"area1\": " + String(weight1, 2) + ", ";
    jsonResponse += "\"area2\": " + String(weight2, 2) + ", ";
    jsonResponse += "\"area3\": " + String(weight3, 2);
    jsonResponse += "}";

    server.send(200, "application/json", jsonResponse);
  });

  // Start Server
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
}
