#include "dvs_read_txt.h"

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "dvs_displayer");

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  DVSReadTxt syncronizer(nh, nh_private);
  syncronizer.readFile();
  while (ros::ok())
  {
    ros::spinOnce(); // Handle ROS events
    syncronizer.loopOnce();
  }

  return 0;
}
