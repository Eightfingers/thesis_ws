#pragma once

#include <ros/ros.h>
#include <dvs_msgs/EventArray.h>
#include <dvs_msgs/Event.h>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

// Dynamic reconfigure
#include <dynamic_reconfigure/server.h>

using std::chrono::duration;

class DVSReadTxt
{
public:
    DVSReadTxt(ros::NodeHandle &nh, ros::NodeHandle p_nh);

    void readFile();
    void publishOnce(double start_time, double end_time);
    void loopOnce();

    virtual ~DVSReadTxt();

private:
    void readTimeSliceEventsVec(
        int &start_index,
        double start_time,
        double end_time,
        std::vector<dvs_msgs::Event> &event_vector,
        std::vector<double> &disparity_vector,
        dvs_msgs::EventArray &event_arr,
        cv::Mat &event_image_polarity,
        cv::Mat &event_image_disp_gt,
        cv::Mat &map1_x,
        cv::Mat &map1_y);

    void publishGTDisparity(
        cv::Mat &disparity_gt,
        image_transport::Publisher &disparity_gt_image_pub);

    void calcPublishDisparity(
        cv::Mat &event_image_polarity_left,
        cv::Mat &event_image_polarity_right,
        cv::Mat &left_gt_disparity,
        std::ofstream &file);

    void applyNearestNeighborFilter(cv::Mat &event_image_pol, int value_of_empty_cell);

    void createAdaptiveWindowMap(cv::Mat &event_image_pol,
                                        cv::Mat &sobel_x,
                                        cv::Mat &sobel_y,
                                        cv::Mat &mean,
                                        cv::Mat &mean_sq,
                                        cv::Mat &gradient_sq,
                                        cv::Mat &image_window_sizes);

    ros::NodeHandle nh_;
    ros::NodeHandle p_nh_;

    ros::Publisher left_event_arr_pub_;
    ros::Publisher right_event_arr_pub_;

    std::vector<dvs_msgs::Event> left_events_;
    std::vector<dvs_msgs::Event> right_events_;
    std::vector<double> left_disparity_values_;
    std::vector<double> right_disparity_values_;

    std::chrono::time_point<std::chrono::high_resolution_clock> time_start_;
    std::chrono::high_resolution_clock::time_point loop_timer_;

    image_transport::Publisher img_pub_disparity_gt_left_;
    image_transport::Publisher img_pub_disparity_gt_right_;
    image_transport::Publisher debug_image_pub_;

    std::string left_event_csv_file_path_;
    std::string right_event_csv_file_path_;
    std::string event_txt_file_path_;

    // Configs
    bool calc_disparity_ = false;
    bool first_loop_ = true;
    bool do_NN_ = false;
    bool publish_slice_ = false;
    bool file_is_txt_ = false;
    bool write_to_text_ = false;
    bool do_adaptive_window_ = true;
    int camera_height_;
    int camera_width_;
    double loop_rate_;
    double slice_start_time_;
    double slice_end_time_;

    int NN_block_size_;
    int NN_min_num_of_events_;
    int left_event_index_ = 0;
    int right_event_index_ = 0;

    int small_block_size_;
    int medium_block_size_;
    int large_block_size_;

    cv_bridge::CvImage cv_image_;
    cv_bridge::CvImage cv_image_disparity_;
    cv_bridge::CvImage debug_image_;

    image_transport::Publisher sad_disparity_pub_;
    int disparity_range_;   // Maximum disparity
    int window_block_size_; // window size (must be odd)

    cv::Mat disparity_;
    cv::Mat event_disparity_;
    cv::Mat color_map_;

    cv::Mat event_image_left_polarity_;
    cv::Mat disparity_gt_left_;

    cv::Mat event_image_right_polarity_;
    cv::Mat disparity_gt_right_;

    std::ofstream gt_disparity_left_file_;
    std::ofstream gt_disparity_right_file_;
    std::ofstream disparity_file_;

    // Adaptive winmdow
    cv::Mat sobel_x_, sobel_y_, mean_, mean_sq_, gradient_sq_;
    cv::Mat window_size_map_;
    int threshold_edge_;

    std::string gt_disparity_left_path_ = "gt_disparity_left.txt";
    std::string gt_disparity_right_path_ = "gt_disparity_right.txt";
    std::string disparity_path_ = "disparity_out_.txt";

    // Camera matrices (K) and distortion coefficients (dist)
    cv::Mat K_0 = (cv::Mat_<double>(3, 3) << 571.48555589, 0, 636.00644236,
                   0, 572.41991416, 357.83088501,
                   0, 0, 1);
    cv::Mat dist_0 = (cv::Mat_<double>(1, 4) << -0.02343015, 0.04860866, -0.00181647, 0.00308489);

    cv::Mat K_1 = (cv::Mat_<double>(3, 3) << 570.9912814, 0, 664.05891687,
                   0, 571.99822399, 351.11646702,
                   0, 0, 1);
    cv::Mat dist_1 = (cv::Mat_<double>(1, 4) << 0.02650264, -0.01766294, -0.0022541, 0.00644137);

    // Rotation and translation between the cameras
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F); // Assuming the rotation matrix R is identity for simplicity.
    cv::Mat T = (cv::Mat_<double>(3, 1) << 0.03963375, -0.00024961, 0.00266686);

    // Stereo rectification variables
    cv::Mat R1, R2, P1, P2, Q;

    // Rectification maps
    cv::Mat map1_x, map1_y, map2_x, map2_y;
};
