// POSITRONIC FLUFFY GARBANZO - AIR HOCKEY ROBOT

// Robot Moves depends directly on robot status
// robot status:
//   5: (Default) Manual mode => User send direct commands to robot
void robotStrategy()
{
  max_speed = user_max_speed;  // default to max robot speed and accel
  max_acceleration = user_max_accel;
  
  // User manual control
  max_speed = user_target_speed;
  
  // Control acceleration
  max_acceleration = user_target_accel;
  setPosition_straight(user_target_x, user_target_y, robotCoordX, robotCoordY);
}

// Test sequence to check mechanics, motor drivers...
void testMovements()
{
  if (loop_counter >= 9000) {
    testmode = false;
    return;
  }
  max_speed = user_max_speed;
  if (loop_counter > 8000) {
    // setPosition_straight(ROBOT_INITIAL_POSITION_X, ROBOT_INITIAL_POSITION_Y);
    Serial.println("Taunt!");
  }
  else if (loop_counter > 6260) {
    // setPosition_straight(100, 200);
    Serial.println("Taunt!");
  }
  else if (loop_counter > 6000) {
    // setPosition_straight(320, 200);
    Serial.println("Taunt!");
  }
  else if (loop_counter > 5000) {
    // setPosition_straight(ROBOT_INITIAL_POSITION_X, ROBOT_INITIAL_POSITION_Y);
    Serial.println("Taunt!");
  }
  else if (loop_counter > 3250) {
    // setPosition_straight(300, 280);
    Serial.println("Taunt!");
  }
  else if (loop_counter > 3000) {
    // setPosition_straight(ROBOT_INITIAL_POSITION_X, 280);
    Serial.println("Taunt!");
  }
  else if (loop_counter > 2500) {
    // setPosition_straight(ROBOT_INITIAL_POSITION_X, ROBOT_INITIAL_POSITION_Y);
    Serial.println("Taunt!");
  }
  else if (loop_counter > 1500) {
    // setPosition_straight(ROBOT_INITIAL_POSITION_X, ROBOT_MAX_Y);
    Serial.println("Taunt!");
  }
  else {
    // setPosition_straight(ROBOT_INITIAL_POSITION_X, ROBOT_INITIAL_POSITION_Y);
    Serial.println("I am now going to taunt you, silly human!");
  }
}
