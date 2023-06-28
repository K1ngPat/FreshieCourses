// Arduino pins for PWM outputs connected to Motor Enables.
#define MOTOR1_PWM 10
// Arduino digital pins for motor direction
#define MOTOR1_PLUS 7
#define MOTOR1_MINUS 8
void setup() {
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
  //Negative speed values imply the given speed in opposite direction.
  //separate speed and direction
  if (speed < 0) {
    direction = -1;
    speed = -speed;
  } else direction = 1;
  if (speed > 255) speed = 255;
     m_plus = MOTOR1_PLUS;
     m_minus = MOTOR1_MINUS;
     m_pwm = MOTOR1_PWM;
  //write values to debug code
  //Serial.print(" m_plus: ");
  //Serial.print(m_plus);
  //Serial.print(" m_minus: ");
  //Serial.print(m_minus);
  //Serial.print(" speed: ");
  //Serial.println(speed);

  //Set rotation direction
  if (direction == 1) {
    digitalWrite(m_plus, HIGH);
    digitalWrite(m_minus, LOW);
  } else {
    digitalWrite(m_plus, LOW);
    digitalWrite(m_minus, HIGH);
  }
  //And set the pwm value
  analogWrite(m_pwm, speed);
  //Done!
}

void loop(void) {
int speed;
  //In Expt. 5A, we run the motor at constant speed.
  speed = 128; //edit this value for different speeds.
//Run the motor at this PWM value, measure RPM
//And  Repeat for other PWM values from 32 to 224 in steps of 32.
    run_motor(speed);
}
