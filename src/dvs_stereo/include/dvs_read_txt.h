#pragma once

// #define DEBUG_MODE

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
        cv::Mat &event_image_polarity);

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
    
    // Message filters subscribers
    message_filters::Subscriber<dvs_msgs::EventArray> left_event_sub_;
    message_filters::Subscriber<dvs_msgs::EventArray> right_event_sub_;

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
    bool rectify_ = true;
    bool checker_board_window_ = false;

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
    cv_bridge::CvImage cv_image_depth_;
    cv_bridge::CvImage left_pol_image_;
    cv_bridge::CvImage right_pol_image_;

    image_transport::Publisher estimated_disparity_pub_;
    image_transport::Publisher estimated_depth_pub_;
    image_transport::Publisher left_pol_image_pub_;
    image_transport::Publisher right_pol_image_pub_;

    int disparity_range_ ; // Maximum disparity
    int step_size_;
    int window_fill_size_;

    cv::Mat depth_map_;
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
    cv::Mat K_0 = (cv::Mat_<double>(3,3) << 570.9912814, 0, 664.05891687, // Slave left
                                           0, 571.99822399, 351.11646702, 
                                           0, 0, 1);
    cv::Mat dist_0 = (cv::Mat_<double>(1,4) << 0.02650264, -0.01766294, -0.0022541, 0.00644137);

//     Camera matrix Slave
// [572.4314787055476, 0, 655.5993194282812;
//  0, 572.4314787055476, 353.9229001533187;
//  0, 0, 1] 
// Distortion coefficients 
// [-0.08538147921551763, 0.1829366218062925, 0.0006811934446241191, 0.002427034158652812, 0] 

    cv::Mat K_1 = (cv::Mat_<double>(3,3) << 571.48555589, 0, 636.00644236, // Master, right
                                           0, 572.41991416, 357.83088501, 
                                           0, 0, 1);
    cv::Mat dist_1 = (cv::Mat_<double>(1,4) << -0.02343015, 0.04860866, -0.00181647, 0.00308489);

    // camera matrix Master
    // "camera_matrix": {
    //     "type_id": "opencv-matrix",
    //     "rows": 3,
    //     "cols": 3,
    //     "dt": "d",
    //     "data": [ 5.6397392716854040e+02, 0.0, 6.3196082987426553e+02,
    //         0.0, 5.6397392716854040e+02, 3.6335609870294326e+02, 0.0,
    //         0.0, 1.0 ]
    // },
    // "distortion_coefficients": {
    //     "type_id": "opencv-matrix",
    //     "rows": 1,
    //     "cols": 5,
    //     "dt": "d",
    //     "data": [ -8.1253226522012695e-02, 1.6508260669048941e-01,
    //         -2.1049100648449411e-04, 2.4157447475095357e-03, 0.0 ]

    // Rotation and translation between the cameras
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F); // Assuming the rotation matrix R is identity for simplicity.
    cv::Mat T = (cv::Mat_<double>(3,1) << 0.03963375, -0.00024961, 0.00266686);

    // Stereo rectification variables
    cv::Mat R0, R1, P0, P1, Q;

    // Rectification maps
    cv::Mat map_master_x, map_master_y, map_slave_x, map_slave_y;
};
