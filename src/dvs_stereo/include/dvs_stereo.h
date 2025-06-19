#pragma once


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

// #define DEBUG_MODE
#define USE_TS2

using std::chrono::duration;

class DVSStereo
{
public:
    DVSStereo(ros::NodeHandle &nh, ros::NodeHandle p_nh);

    void publishOnce();
    void getSuitableSlice(dvs_msgs::EventArray &event_array, double start_time, double end_time);
    virtual ~DVSStereo();

private:
    void readEventArray(
        dvs_msgs::EventArray &event_array,
        cv::Mat &event_image_polarity,
        cv::Mat &event_ts);

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
        cv::Mat &event_image_polarity_right,
        cv::Mat &event_image_left_ts,
        cv::Mat &event_image_right_ts);
    
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

    bool calc_disparity_ = false;
    bool first_loop_ = true;
    bool do_NN_ = false;
    bool publish_slice_;
    bool write_to_text_ = false;
    bool do_adaptive_window_ = true;
    bool rectify_ = true;

    cv::Size image_size_;
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

    cv::Rect crop_region_;
    cv::Size cropped_image_size_;
    int crop_width_ = 1200;
    int crop_height_ = 630;
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
    cv_bridge::CvImage rect_left_image_;
    cv_bridge::CvImage rect_right_image_;
    cv_bridge::CvImage sobel_debug_image_;

    image_transport::Publisher estimated_disparity_pub_;
    image_transport::Publisher estimated_disparity_pub2_;
    image_transport::Publisher estimated_depth_pub_;
    image_transport::Publisher left_pol_image_pub_;
    image_transport::Publisher right_pol_image_pub_;
    image_transport::Publisher rect_left_image_pub_;
    image_transport::Publisher rect_right_image_pub_;
    image_transport::Publisher sobel_img_pub_;

    int disparity_range_ ; // Maximum disparity
    int step_size_;
    int window_fill_size_;

    cv::Mat depth_map_;
    cv::Mat disparity_;
    cv::Mat disparity2_;

    cv::Mat event_image_left_;
    cv::Mat event_image_left_polarity_remmaped_;
    cv::Mat event_image_left_polarity_;
    
    cv::Mat event_image_left_ts_;
    cv::Mat event_image_left_ts_remapped_;
    cv::Mat event_image_right_ts_;
    cv::Mat event_image_right_ts_remapped_;

    cv::Mat event_image_right_;
    cv::Mat event_image_right_polarity_remmaped_;
    cv::Mat event_image_right_polarity_;
    cv::Mat disparity_color_map_;

    std::ofstream gt_disparity_left_file_;
    std::ofstream gt_disparity_right_file_;
    std::ofstream estimated_disparity_file_;

    // Adaptive window Matrices
    cv::Mat sobel_x_, sobel_y_, mean_, mean_sq_, gradient_sq_;
    cv::Mat image_binary_map_;

    std::string estimated_disparity_path_ = "SAD_disparity_.txt";

    cv::Mat K_0 = (cv::Mat_<double>(3,3) << // slave left
        560.12251945, 0, 649.5897273,
        0, 560.60013947, 350.82562497,
        0, 0, 1);
    cv::Mat dist_0 = (cv::Mat_<double>(1,4) << 
        0.01820616, -0.01350185, -0.00057661, -0.00074246);

    cv::Mat K_1 = (cv::Mat_<double>(3,3) << // master right
        563.87494477, 0, 626.17947286,
        0, 564.16504935, 358.44549531,
        0, 0, 1);
    cv::Mat dist_1 = (cv::Mat_<double>(1,4) << 
        -0.00760154, 0.0203495, -0.00069216, -0.00175436);

// const double FOCAL_LENGTH = 0.0027; // f (from K_1)
// const double BASELINE = 0.03963375;       // B (from T_x)
    
    // Rotation and translation between the cameras
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F); // Assuming the rotation matrix R is identity for simplicity.

    cv::Mat T = (cv::Mat_<double>(3,1) << 0.0398738, -0.00020666, -0.00021764);

    // Stereo rectification variables
    cv::Mat R0, R1, P0, P1, Q;

    // Rectification maps
    cv::Mat map_master_x, map_master_y, map_slave_x, map_slave_y;
};
