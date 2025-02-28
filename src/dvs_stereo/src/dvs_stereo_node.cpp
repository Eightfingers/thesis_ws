#include "dvs_read_txt.h"

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "dvs_displayer");

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  DVSStereo syncronizer(nh, nh_private);
  while (ros::ok())
  {
    ros::spinOnce(); // Handle ROS events
    // syncronizer.loopOnce();
  }

  return 0;
}
