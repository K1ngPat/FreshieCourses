// Arduino pins for PWM outputs connected to Motor Enables.
#define MOTOR1_PWM 10
// Arduino digital pins for motor direction
#define MOTOR1_PLUS 7
#define MOTOR1_MINUS 8
void setup() {
  int i;
  //Motor control pins in output mode
  pinMode(MOTOR1_PWM, OUTPUT);
  pinMode(MOTOR1_PLUS, OUTPUT);
  pinMode(MOTOR1_MINUS, OUTPUT);
  //At power on, all motors should be OFF
  idle();
  //For potential debugging
  //Serial.begin(9600);
}
void idle(void) {
  digitalWrite(MOTOR1_PLUS, LOW);
  digitalWrite(MOTOR1_MINUS, LOW);
  digitalWrite(MOTOR1_PWM, LOW);
}
void run_motor(int speed) {
  int direction, m_plus, m_minus, m_pwm;
  //negative values of speed imply speed in reverse direction.
  if (speed < 0) {
    direction = -1;
    speed = -speed;
  } else direction = 1;
  if (speed > 255) speed = 255;

     m_plus = MOTOR1_PLUS;
     m_minus = MOTOR1_MINUS;
     m_pwm = MOTOR1_PWM;
  if (direction == 1) {
    digitalWrite(m_plus, HIGH);
    digitalWrite(m_minus, LOW);
  } else {
    digitalWrite(m_plus, LOW);
    digitalWrite(m_minus, HIGH);
  }
  //write values to debug code
  //Serial.print(" m_plus: ");
  //Serial.print(m_plus);
  //Serial.print(" m_minus: ");
  //Serial.print(m_minus);
  //Serial.print(" speed: ");
  //Serial.println(speed);
  //Set rotation direction
  //And set the pwm value
  analogWrite(m_pwm, speed);
  //Done!
}

void loop(void) {
int speed;
  //In Expt. 5B, we accelerate, decelerate, stop
  //Then accelerate decelerate in the opposite direction. 
  //Accelerate from 0 to max
  for (speed = 0; speed < 255; speed += 16) {
    run_motor(speed);
    delay(200);
  }
  //Decelerate to 0 then accelerate in the reverse direction to max
  for (speed = 240; speed > -255; speed -= 16) {
    run_motor(speed);
    delay(200);
  }
  //Decelerate from max speed in reverse direction to static
  for (speed = -240; speed < 0; speed += 16) {
    run_motor(speed);
    delay(200);
  }
  //Take a breath ... and repeat the loop
  delay(500);
}
