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

// #define USE_TS
// #define USE_NSAD

using std::chrono::duration;

class DVSReadTxt
{
public:
    DVSReadTxt(ros::NodeHandle &nh, ros::NodeHandle p_nh);
    virtual ~DVSReadTxt();

    void readFile();
    void readTimeSliceEventsVec(
        int &start_index,
        double start_time,
        double end_time,
        std::vector<dvs_msgs::Event> &event_vector,
        std::vector<double> &disparity_vector,
        dvs_msgs::EventArray &event_arr,
        cv::Mat &event_image_polarity,
        cv::Mat &event_image_sum,
        cv::Mat &event_image_disp_gt,
        cv::Mat &event_ts,
        cv::Mat &event_ts_prev,
        cv::Mat &map1_x,
        cv::Mat &map1_y);
    void applyNearestNeighborFilter(cv::Mat &event_image_pol, int value_of_empty_cell);
    void createBinaryEdgeMap(cv::Mat &event_image_pol,
                             cv::Mat &sobel_x,
                             cv::Mat &sobel_y,
                             cv::Mat &mean,
                             cv::Mat &mean_sq,
                             cv::Mat &gradient_sq,
                             cv::Mat &image_window_sizes);
    void publishGTDisparity(
        cv::Mat &disparity_gt,
        image_transport::Publisher &disparity_gt_image_pub);
    void calcPublishDisparity(
        cv::Mat &event_image_polarity_left,
        cv::Mat &event_image_polarity_right,
        cv::Mat &left_gt_disparity,
        std::ofstream &file);

    void sparseRemap(cv::Mat &event_image, cv::Mat &remapped_image, cv::Mat &map1_x, cv::Mat &map1_y, int empty_pixel_value);

    void publishOnce(double start_time, double end_time);
    void loopOnce();

private:
    // ROS Node Handles
    ros::NodeHandle nh_;
    ros::NodeHandle p_nh_;

    // Publishers
    ros::Publisher left_event_arr_pub_;
    ros::Publisher right_event_arr_pub_;
    image_transport::Publisher disparity_pub_;
    image_transport::Publisher img_pub_disparity_gt_left_;
    image_transport::Publisher img_pub_disparity_gt_right_;
    image_transport::Publisher debug_img_ts_left_pub_;
    image_transport::Publisher debug_img_ts_right_pub_;
    image_transport::Publisher sobel_img_pub_;
    image_transport::Publisher rect_left_image_pub_;
    image_transport::Publisher rect_right_image_pub_;

    // OpenCV Images
    cv_bridge::CvImage gt_cv_image_;
    cv_bridge::CvImage cv_image_disparity_;
    cv_bridge::CvImage cv_image_ts_;
    cv_bridge::CvImage sobel_debug_image_;
    cv_bridge::CvImage rect_left_image_;
    cv_bridge::CvImage rect_right_image_;

    // Event Data Storage
    std::vector<dvs_msgs::Event> left_events_;
    std::vector<dvs_msgs::Event> right_events_;
    std::vector<double> left_disparity_values_;
    std::vector<double> right_disparity_values_;
    int left_event_index_ = 0;
    int right_event_index_ = 0;

    // Timing Control
    std::chrono::time_point<std::chrono::high_resolution_clock> time_start_;
    std::chrono::high_resolution_clock::time_point loop_timer_;
    double loop_rate_;
    double slice_start_time_;
    double slice_end_time_;

    // Configuration Flags
    bool first_loop_ = true;
    bool do_disp_estimation_ = false;
    bool do_NN_ = false;
    bool publish_slice_ = false;
    bool file_is_txt_ = false;
    bool write_to_text_ = false;
    bool do_adaptive_window_ = true;
    bool rectify_ = false;

    // Camera and Cropping Parameters
    int camera_height_;
    int camera_width_;
    int crop_width = 1200;
    int crop_height = 630;
    int new_mid_x;
    int new_mid_y;
    int left_empty_pixel_val_ = 127;
    int right_empty_pixel_val_ = 126;

    // Algorithm Parameters
    // Nearest Neighbour
    int NN_block_size_;
    int NN_min_num_of_events_;
    // Adaptive Window
    int window_size_;
    int threshold_edge_;
    // Search range
    int disparity_range_;

    // Image Processing Matrices
    cv::Mat color_map_;
    cv::Mat event_image_left_polarity_;
    cv::Mat event_image_left_sum_;
    cv::Mat event_image_left_polarity_remmaped_;
    cv::Mat disparity_gt_left_;

    cv::Mat event_image_right_polarity_;
    cv::Mat event_image_right_polarity_remmaped_;
    cv::Mat event_image_right_sum_;
    cv::Mat disparity_gt_right_;
    cv::Mat event_image_left_ts_;
    cv::Mat event_image_left_ts_prev_;
    cv::Mat event_image_right_ts_;
    cv::Mat event_image_right_ts_prev_;

    // Adaptive window Matrices
    cv::Mat sobel_x_, sobel_y_, mean_, mean_sq_, gradient_sq_;
    cv::Mat image_binary_map_;

    cv::Mat K_0 = (cv::Mat_<double>(3, 3) << // slave left
                       560.12251945,0, 649.5897273,
                   0, 560.60013947, 350.82562497,
                   0, 0, 1);
    cv::Mat dist_0 = (cv::Mat_<double>(1, 4) << 0.01820616, -0.01350185, -0.00057661, -0.00074246);

    cv::Mat K_1 = (cv::Mat_<double>(3, 3) << // master right
                       563.87494477, 0, 626.17947286,
                   0, 564.16504935, 358.44549531,
                   0, 0, 1);
    cv::Mat dist_1 = (cv::Mat_<double>(1, 4) << -0.00760154, 0.0203495, -0.00069216, -0.00175436);

    // Rotation and translation between the cameras
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F); // Assuming the rotation matrix R is identity for simplicity.
    cv::Mat T = (cv::Mat_<double>(3, 1) << 0.0398738, -0.00020666, -0.00021764);

    // Stereo rectification variables
    cv::Mat R1, R2, P1, P2, Q;
    cv::Mat map_slave_x, map_slave_y, map_master_x, map_master_y;

    // File Paths
    std::ofstream disparity_file_;
    std::string left_event_csv_file_path_;
    std::string right_event_csv_file_path_;
    std::string event_txt_file_path_;
    std::string disparity_path_ = "disparity_out_.txt";

    int number_of_left_events_estimated_ = 0;
    int number_of_accurate_left_events_ = 0;
};
