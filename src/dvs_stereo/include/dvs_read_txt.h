#pragma once

#define DEBUG_MODE

#include <ros/ros.h>
#include <dvs_msgs/EventArray.h>
#include <dvs_msgs/Event.h>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// Dynamic reconfigure
#include <dynamic_reconfigure/server.h>


using std::chrono::duration;

class DVSStereo
{
public:
    DVSStereo(ros::NodeHandle &nh, ros::NodeHandle p_nh);

    void publishOnce(double start_time, double end_time);
    void loopOnce();
    void getSuitableSlice(dvs_msgs::EventArray &event_array, double start_time, double end_time);
    virtual ~DVSStereo();

private:
    void readEventArray(
        dvs_msgs::EventArray &event_array,
        cv::Mat &event_image_polarity,
        cv::Mat &map1_x,
        cv::Mat &map1_y);

    void applyNearestNeighborFilter(cv::Mat &event_image_pol, int value_of_empty_cell);

    void createAdaptiveWindowMap(cv::Mat &event_image_pol,
        cv::Mat &sobel_x,
        cv::Mat &sobel_y,
        cv::Mat &mean,
        cv::Mat &mean_sq,
        cv::Mat &gradient_sq,
        cv::Mat &image_window_sizes);

    void calcPublishDisparity(
        cv::Mat &event_image_polarity_left,
        cv::Mat &event_image_polarity_right);

    void syncCallback(const dvs_msgs::EventArray::ConstPtr &msg1, const dvs_msgs::EventArray::ConstPtr &msg2);

    ros::NodeHandle nh_;
    ros::NodeHandle p_nh_;
    
    ros::Subscriber left_event_sub_;
    ros::Subscriber right_event_sub_;

    // Message filters subscribers
    message_filters::Subscriber<dvs_msgs::EventArray> left_event_sub_2;
    message_filters::Subscriber<dvs_msgs::EventArray> right_event_sub_2;

    // Synchronizer
    typedef message_filters::sync_policies::ApproximateTime<dvs_msgs::EventArray, dvs_msgs::EventArray> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    boost::shared_ptr<Sync> sync_;

    dvs_msgs::EventArray left_events_;
    dvs_msgs::EventArray right_events_;

    std::chrono::time_point<std::chrono::high_resolution_clock> time_start_;
    std::chrono::high_resolution_clock::time_point loop_timer_;

    std::string left_event_dataset_file_path_;
    std::string right_event_dataset_file_path_;

    bool calc_disparity_ = false;
    bool first_loop_ = true;
    bool do_NN_ = false;
    bool publish_slice_;
    bool write_to_text_ = false;
    bool do_adaptive_window_ = true;

    int camera_height_ ;
    int camera_width_ ;
    int disparity_step_;
    double slice_rate_;
    double slice_start_time_;
    double slice_end_time_;
    int NN_block_size_;
    int NN_min_num_of_events_;  
    int left_event_index_ = 0;
    int right_event_index_ = 0;
    int left_empty_pixel_val_ = 127;
    int right_empty_pixel_val_ = 126;

    int crop_width = 1200;
    int crop_height = 630;
    int new_mid_x;
    int new_mid_y;

    // Adaptive Window
    int small_block_size_;
    int medium_block_size_;
    int large_block_size_;
    int threshold_edge_;

    double age_penalty_;
    int msg_queue_size_;

    cv_bridge::CvImage cv_image_disparity_;
    image_transport::Publisher estimated_disparity_pub_;
    int disparity_range_ ; // Maximum disparity
    int window_block_size_;  // must be odd
    int window_step_size_;
    int window_fill_size_;

    cv::Mat event_disparity_;
    cv::Mat disparity;

    cv::Mat event_image_left_;
    cv::Mat event_image_left_polarity_remmaped_;
    cv::Mat event_image_left_polarity_;
    
    cv::Mat event_image_right_;
    cv::Mat event_image_right_polarity_remmaped_;
    cv::Mat event_image_right_polarity_;
    cv::Mat disparity_color_map_;

    std::ofstream gt_disparity_left_file_;
    std::ofstream gt_disparity_right_file_;
    std::ofstream estimated_disparity_file_;

    // Adaptive window Matrices
    cv::Mat sobel_x_, sobel_y_, mean_, mean_sq_, gradient_sq_;
    cv::Mat window_size_map_;

    std::string estimated_disparity_path_ = "SAD_disparity_.txt";

    // Camera matrices (K) and distortion coefficients (dist)
    cv::Mat K_0 = (cv::Mat_<double>(3,3) << 571.48555589, 0, 636.00644236, 
                                           0, 572.41991416, 357.83088501, 
                                           0, 0, 1);
    cv::Mat dist_0 = (cv::Mat_<double>(1,4) << -0.02343015, 0.04860866, -0.00181647, 0.00308489);

    cv::Mat K_1 = (cv::Mat_<double>(3,3) << 570.9912814, 0, 664.05891687, 
                                           0, 571.99822399, 351.11646702, 
                                           0, 0, 1);
    cv::Mat dist_1 = (cv::Mat_<double>(1,4) << 0.02650264, -0.01766294, -0.0022541, 0.00644137);

    // Rotation and translation between the cameras
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F); // Assuming the rotation matrix R is identity for simplicity.
    cv::Mat T = (cv::Mat_<double>(3,1) << 0.03963375, -0.00024961, 0.00266686);

    // Stereo rectification variables
    cv::Mat R1, R2, P1, P2, Q;

    // Rectification maps
    cv::Mat map1_x, map1_y, map2_x, map2_y;

};
