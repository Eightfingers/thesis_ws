#include "dvs_stereo.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <omp.h>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

// #define DEBUG_MODE  // Uncomment this line to enable debug messages

DVSStereo::DVSStereo(ros::NodeHandle &nh, ros::NodeHandle nh_private) : nh_(nh), p_nh_(nh_private)
{

    nh_private.param<int>("camera_height", camera_height_, 11);
    nh_private.param<int>("camera_width", camera_width_, 11);

    nh_private.param<bool>("nearest_neighbour", do_NN_, false);
    nh_private.param<bool>("write_to_text", write_to_text_, false);
    nh_private.param<bool>("calc_disparity", calc_disparity_, false);
    nh_private.param<bool>("do_adaptive_window", do_adaptive_window_, true);
    nh_private.param<bool>("rectify", rectify_, true);

    nh_private.param<int>("large_block_size", large_block_size_, 11);
    nh_private.param<int>("threshold_edge", threshold_edge_, 300);

    nh_private.param<double>("age_penalty", age_penalty_, 0.1);
    nh_private.param<int>("msg_queue_size", msg_queue_size_, 1);

    nh_private.param<int>("disparity_range", disparity_range_, 40);
    nh_private.param<int>("disparity_step", disparity_step_, 1);
    nh_private.param<int>("step_size", step_size_, 2);
    nh_private.param<int>("window_fill_size", window_fill_size_, 1);
    nh_private.param<int>("NN_block_size", NN_block_size_, 10);
    nh_private.param<int>("NN_min_num_of_events", NN_min_num_of_events_, 3);

    // Initialize subscribers with message_filters
    left_event_sub_.subscribe(nh_, "/left_events", 10); // left is slave
    right_event_sub_.subscribe(nh_, "/right_events", 10); // right is master

    // Create an Exact Time Synchronizer
    // Sync policy of q size
    sync_.reset(new Sync(MySyncPolicy(msg_queue_size_), left_event_sub_, right_event_sub_));
    sync_->registerCallback(boost::bind(&DVSStereo::syncCallback, this, _1, _2));
    // sync_->setAgePenalty(age_penalty_);  // Reduce age penalty (100 microseconds)
    // sync_->setMaxIntervalDuration(ros::Duration(0.010));  // 10 millisecond tolerance

    // Stereo rectify
    cv::Size image_size(camera_width_, camera_height_);

    image_transport::ImageTransport it_(nh);
    estimated_disparity_pub_ = it_.advertise("left_disparity", 1);
    left_pol_image_pub_ = it_.advertise("left_pol_image", 1);
    right_pol_image_pub_ = it_.advertise("right_pol_image", 1);
    estimated_depth_pub_ = it_.advertise("left_depth_map", 1);

    rect_left_image_.encoding = "mono8";
    rect_left_image_.image = cv::Mat(image_size, CV_32F, cv::Scalar(0));
    rect_left_image_pub_ = it_.advertise("left_rect", 1);

    rect_right_image_.encoding = "mono8";
    rect_right_image_.image = cv::Mat(image_size, CV_32F, cv::Scalar(0));
    rect_right_image_pub_ = it_.advertise("right_rect", 1);

    cv_image_disparity_.encoding = "bgr8";
    cv_image_disparity_.image = cv::Mat(image_size, CV_8U, cv::Scalar(0));

    cv_image_depth_.encoding = "bgr8";
    cv_image_depth_.image = cv::Mat(image_size, CV_8U, cv::Scalar(0));

    left_pol_image_.encoding = "mono8";
    left_pol_image_.image = cv::Mat(image_size, CV_8U, cv::Scalar(0));
    right_pol_image_.encoding = "mono8";
    right_pol_image_.image = cv::Mat(image_size, CV_8U, cv::Scalar(0));

    event_image_left_ = cv::Mat(image_size, CV_8U, cv::Scalar(0));
    event_image_left_polarity_ = cv::Mat(image_size, CV_8U, cv::Scalar(left_empty_pixel_val_));
    event_image_left_polarity_remmaped_ = cv::Mat(image_size, CV_8U, cv::Scalar(left_empty_pixel_val_));

    event_image_right_ = cv::Mat(image_size, CV_8U, cv::Scalar(0));
    event_image_right_polarity_ = cv::Mat(image_size, CV_8U, cv::Scalar(right_empty_pixel_val_));
    event_image_right_polarity_remmaped_ = cv::Mat(image_size, CV_8U, cv::Scalar(right_empty_pixel_val_));

    disparity_color_map_ = cv::Mat(image_size, CV_8U, cv::Scalar(0));
    depth_map_ = cv::Mat(image_size, CV_32F, cv::Scalar(0));

    sobel_x_ = cv::Mat::zeros(image_size, CV_64F);
    sobel_y_ = cv::Mat::zeros(image_size, CV_64F);
    mean_ = cv::Mat::zeros(image_size, CV_64F);
    mean_sq_ = cv::Mat::zeros(image_size, CV_64F);
    gradient_sq_ = cv::Mat::zeros(image_size, CV_64F);
    image_binary_map_ = cv::Mat::zeros(image_size, CV_8U);

    cv::stereoRectify(K_0, dist_0, K_1, dist_1, image_size, R, T, R0, R1, P0, P1, Q, 0);
    cv::initUndistortRectifyMap(K_0, dist_0, R0, P0, image_size, CV_32FC1, map_slave_x, map_slave_y); // Slave, Left
    cv::initUndistortRectifyMap(K_1, dist_1, R1, P1, image_size, CV_32FC1, map_master_x, map_master_y); // Master, Right
    
    new_mid_x = (event_image_left_polarity_.cols - crop_width) / 2;  // Center horizontally
    new_mid_y = (event_image_left_polarity_.rows - crop_height) / 2; // Center vertically

    // cv::Rect crop_region(new_mid_x, new_mid_y, crop_width, crop_height);
    // // Crop the image
    // event_image_left_polarity_remmaped_  = event_image_left_polarity_remmaped_(crop_region);
    // event_image_right_polarity_remmaped_  = event_image_right_polarity_remmaped_(crop_region);

    ROS_INFO("NN_min_num_of_events_: %d", NN_min_num_of_events_);
    ROS_INFO("Do NN: %s", do_NN_ ? "true" : "false");
    ROS_INFO("Do calc disparity: %s", calc_disparity_ ? "true" : "false");
    ROS_INFO("Do adaptive window?: %s", do_adaptive_window_ ? "true" : "false");
    ROS_INFO("Finished init!");
    
    loop_timer_ = high_resolution_clock::now();
}

DVSStereo::~DVSStereo()
{
}

void DVSStereo::publishOnce()
{
    auto start_time_read = std::chrono::high_resolution_clock::now();

    event_image_left_polarity_.setTo(left_empty_pixel_val_);
    event_image_left_polarity_remmaped_.setTo(left_empty_pixel_val_);

    event_image_right_polarity_.setTo(right_empty_pixel_val_);
    event_image_right_polarity_remmaped_.setTo(right_empty_pixel_val_);

    depth_map_.setTo(0);
    // image_binary_map_.setTo(0);
    image_binary_map_ = cv::Mat::zeros(event_image_right_polarity_.size(), CV_8U);

    // Read, Write & Publish disparity ground truth values of the specified time slice

    readEventArray(left_events_,
                   event_image_left_polarity_,
                   map_slave_x,
                   map_slave_y);

    readEventArray(right_events_,
                   event_image_right_polarity_,
                   map_slave_x,
                   map_slave_y);

    auto end_time_read = std::chrono::high_resolution_clock::now();
    auto duration_read = std::chrono::duration_cast<std::chrono::microseconds>(end_time_read - start_time_read);
    std::cout << "Slicing left right event vector took: " << duration_read.count() / 1000.0 << " milliseconds" << std::endl;

    if (do_NN_)
    {
        applyNearestNeighborFilter(event_image_left_polarity_, left_empty_pixel_val_);
        applyNearestNeighborFilter(event_image_right_polarity_, right_empty_pixel_val_);
    }

    if (rectify_)
    {
        auto start_time_map = std::chrono::high_resolution_clock::now();

        // cv::undistort(event_image_right_polarity_, event_image_right_polarity_remmaped_, K_1, dist_1);
        // cv::undistort(event_image_left_polarity_, event_image_left_polarity_remmaped_, K_0, dist_0);

        cv::remap(event_image_right_polarity_, event_image_right_polarity_remmaped_, map_master_x, map_master_y, cv::INTER_NEAREST);
        cv::remap(event_image_left_polarity_, event_image_left_polarity_remmaped_, map_slave_x, map_slave_y, cv::INTER_NEAREST);
    
        cv::Rect crop_region(new_mid_x, new_mid_y, crop_width, crop_height);
        // Crop the image
        event_image_left_polarity_remmaped_  = event_image_left_polarity_remmaped_(crop_region);
        event_image_right_polarity_remmaped_  = event_image_right_polarity_remmaped_(crop_region);
        image_binary_map_  = image_binary_map_(crop_region);
        auto end_time_map = std::chrono::high_resolution_clock::now();
        auto duration_map = std::chrono::duration_cast<std::chrono::microseconds>(end_time_map - start_time_map);
        event_image_left_polarity_remmaped_.copyTo(rect_left_image_.image);
        rect_left_image_pub_.publish(rect_left_image_.toImageMsg());

        event_image_right_polarity_remmaped_.copyTo(rect_right_image_.image);
        rect_right_image_pub_.publish(rect_right_image_.toImageMsg());
        std::cout << "Rectify took: " << duration_map.count() / 1000.0 << " milliseconds" << std::endl;    
    }
    else
    {
        event_image_left_polarity_remmaped_ = event_image_left_polarity_;
        event_image_right_polarity_remmaped_ = event_image_right_polarity_;
    }


    auto start_time_adapt = std::chrono::high_resolution_clock::now();
    if (do_adaptive_window_)
    {
        createAdaptiveWindowMap(event_image_left_polarity_remmaped_,
                                sobel_x_,
                                sobel_y_,
                                mean_,
                                mean_sq_,
                                gradient_sq_,
                                image_binary_map_);
    }
    auto end_time_adapt = std::chrono::high_resolution_clock::now();
    auto duration_adapt = std::chrono::duration_cast<std::chrono::microseconds>(end_time_adapt - start_time_adapt);
    std::cout << "Adaptive took: " << duration_adapt.count() / 1000.0 << " milliseconds" << std::endl;

    if (calc_disparity_)
    {
        calcPublishDisparity(
            event_image_left_polarity_remmaped_,
            event_image_right_polarity_remmaped_);
    }
}

void DVSStereo::readEventArray(dvs_msgs::EventArray &event_array, cv::Mat &event_image_polarity, cv::Mat &map_x, cv::Mat &map_y)
{
    const size_t num_events = event_array.events.size();
    for (int i = 0; i < num_events; i++)
    {
        const dvs_msgs::Event& tmp = event_array.events[i]; 
        
        if (tmp.polarity){
            event_image_polarity.at<uchar>(tmp.y, tmp.x) = 255;
        }
        else
        {
            event_image_polarity.at<uchar>(tmp.y, tmp.x) = 0;
        }
    }
}

void DVSStereo::applyNearestNeighborFilter(cv::Mat &event_image_pol, int value_of_empty_cell)
{
    auto start_time_NN = std::chrono::high_resolution_clock::now();

    int final_row = event_image_pol.rows - 1;
    int final_col = event_image_pol.cols - 1;
    for (int y = NN_block_size_ / 2; y < final_row - NN_block_size_ / 2; y++)
    {
        for (int x = NN_block_size_ / 2; x < final_col - NN_block_size_ / 2; x++)
        {
            if (event_image_pol.at<uchar>(y, x) == value_of_empty_cell)
            {
                continue; // skip processing
            }
            else
            {
                bool noise = true;
                int num_of_events = 0;
                for (int wy = -NN_block_size_ / 2; wy <= NN_block_size_ / 2; wy++)
                {
                    for (int wx = -NN_block_size_ / 2; wx <= NN_block_size_ / 2; wx++)
                    {
                        // Skip the center pixel (the event we're checking)
                        if (wx == 0 && wy == 0)
                        {
                            continue;
                        }

                        if (event_image_pol.at<uchar>(y + wy, x + wx) != value_of_empty_cell)
                        {
                            num_of_events++;
                            if (num_of_events == NN_min_num_of_events_)
                            {
                                noise = false; // Unlikely to be a noise as another event exist around it
                                break;
                            }
                        }
                    }
                    if (!noise)
                        break;
                }

                if (noise) // remove that event
                {
                    event_image_pol.at<uchar>(y, x) = value_of_empty_cell;
                }
            }
        }
    }

    auto end_time_NN = std::chrono::high_resolution_clock::now();
    auto duration_NN = std::chrono::duration_cast<std::chrono::microseconds>(end_time_NN - start_time_NN);
    #ifdef DEBUG_MODE
    ROS_INFO("Nearest neighbor filtering took: %.2f milliseconds", duration_NN.count() / 1000.0);
    #endif
}

void DVSStereo::createAdaptiveWindowMap(cv::Mat &event_image_pol, cv::Mat &sobel_x, cv::Mat &sobel_y, cv::Mat &mean, cv::Mat &mean_sq, cv::Mat &gradient_sq, cv::Mat &image_window_sizes)
{
    cv::Sobel(event_image_pol, sobel_x, CV_64F, 1, 0, 3);
    cv::Sobel(event_image_pol, sobel_y, CV_64F, 0, 1, 3);

    cv::Mat gradient_magnitude;
    cv::magnitude(sobel_x, sobel_y, gradient_magnitude);

    // Convert to displayable range
    // cv::Mat tmp;
    // tmp = gradient_magnitude.clone();
    // cv::normalize(tmp, tmp, 0, 255, cv::NORM_MINMAX, CV_8U);

    for (int i = 0; i < event_image_pol.rows; i++)
    {
        for (int j = 0; j < event_image_pol.cols; j++)
        {
            if (event_image_pol.at<uchar>(i, j) == left_empty_pixel_val_)
            {
                continue; // skip processing
            }
            double grad_val = gradient_magnitude.at<double>(i, j);
            if (grad_val > threshold_edge_)
            {
                image_window_sizes.at<uchar>(i, j) = 1;
            }
        }
    }
    
}

void DVSStereo::calcPublishDisparity(
    cv::Mat &event_image_polarity_left,
    cv::Mat &event_image_polarity_right)
{
    // Window-based disparity calculation
    auto start_time_disp = std::chrono::high_resolution_clock::now();
    cv::Mat disparity(event_image_polarity_left.size(), CV_8U, cv::Scalar(0));
    
    const double FOCAL_LENGTH = 0.0027; // f (from K_1)
    const double BASELINE = 0.03963375;       // B (from T_x)
    const double MAX_DISTANCE = 22.65;
    const double MIN_DISTANCE = 0.354;

    int half_block = large_block_size_ / 2;
    double window_total_pixel = large_block_size_ * large_block_size_;
    
    const int total_rows = event_image_polarity_left.rows - half_block - 1;
    const int total_cols = event_image_polarity_left.cols - half_block - 1;
    
    if (image_binary_map_.rows != event_image_polarity_left.rows)
    {
        std::cout << "GG \n"; 
    }
    // if (!event_image_polarity_left.isContinuous() || !event_image_polarity_right.isContinuous())
    // {
    //     // Make it continous 
    //     event_image_polarity_left = event_image_polarity_left.clone();
    //     event_image_polarity_right = event_image_polarity_right.clone();
    // }

    for (int y = half_block; y < total_rows; y += step_size_)
    {
        for (int x = half_block; x < total_cols; x += step_size_)
        {
            // uint64_t idx = (y * event_image_polarity_left.cols) + x;
            // if ((uchar) event_image_polarity_left.data[idx] == left_empty_pixel_val_)
            // {
            //     continue; // skip processing
            // }

            // if (do_adaptive_window_ && (uchar) image_binary_map_.data[idx] == 0)
            // {
            //     continue;
            // }

            if (event_image_polarity_left.at<uchar>(y, x) == left_empty_pixel_val_)
            {
                continue; // skip processing
            }

            if (do_adaptive_window_ && image_binary_map_.at<uchar>(y, x) == 0)
            {
                continue;
            }

            // Window calculation
            int best_disparity = 0;
            double min_sad = 10000000;

            // Compare blocks at different disparities
            for (int d = 0; d < disparity_range_; d += disparity_step_)
            {
                if (x - half_block - d < 0)
                {
                    continue; // Skip
                }

                double cost = 0;
                double num_of_non_similar_pixels = 0;

                // Window for the current disparity
                for (int wy = -half_block; wy <= half_block; wy++)
                {
                    for (int wx = -half_block; wx <= half_block; wx++)
                    {                        
                        if ((event_image_polarity_left.at<uchar>(y + wy, x + wx) != event_image_polarity_right.at<uchar>(y + wy, x + wx - d)))
                        {
                            num_of_non_similar_pixels++;
                        }

                        // if ((event_image_polarity_left.data[ ((y + wy) * event_image_polarity_left.cols) + x + wx] != event_image_polarity_right.data[((y + wy) * event_image_polarity_left.cols) + x + wx - d]))
                        // {
                        //     num_of_non_similar_pixels++;
                        // }
                    }
                }

                cost = num_of_non_similar_pixels / window_total_pixel;
                if (cost < min_sad)
                {
                    min_sad = cost;
                    best_disparity = d;
                }
            }

            if (best_disparity != 0)
            {
                double disparity_out = (best_disparity * 255 / disparity_range_);
                // disparity.at<uchar>(y-1, x-1) = disparity_out;
                // disparity.at<uchar>(y-1, x) = disparity_out;
                // disparity.at<uchar>(y-1, x+1) = disparity_out;

                // disparity.at<uchar>(y, x-1) = disparity_out;
                // disparity.at<uchar>(y, x) = disparity_out;
                // disparity.at<uchar>(y, x+1) = disparity_out;

                // disparity.at<uchar>(y+1, x-1) = disparity_out;
                // disparity.at<uchar>(y+1, x) = disparity_out;
                // disparity.at<uchar>(y+1, x+1) = disparity_out;

                double depth = (FOCAL_LENGTH * BASELINE) / best_disparity; // Compute depth
                depth_map_.at<double>(y, x) = depth;
                
                // disparity around this pixel is the same.

                for (int i = y - window_fill_size_; i <= y + window_fill_size_; i++)
                {
                    for (int j = x - window_fill_size_; j <= x + window_fill_size_; j++)
                    {
                        disparity.at<uchar>(i, j) = disparity_out;
                    }
                }
            }
        }
    }

    #ifdef DEBUG_MODE
    event_image_left_polarity_remmaped_.copyTo(left_pol_image_.image);
    left_pol_image_pub_.publish(left_pol_image_.toImageMsg());
    event_image_right_polarity_remmaped_.copyTo(right_pol_image_.image);
    right_pol_image_pub_.publish(right_pol_image_.toImageMsg());
    #endif
    
    // cv::reprojectImageTo3D(disparity, depth_map_, Q);

    // cv::Mat depth_single_channel;
    // cv::extractChannel(depth_map_, depth_single_channel, 2); // Extract Z-depth
    // cv::normalize(depth_single_channel, depth_single_channel, 0, 255, cv::NORM_MINMAX, CV_8U);
    // depth_single_channel.copyTo(cv_image_depth_.image);    
    // estimated_depth_pub_.publish(cv_image_depth_.toImageMsg());

    // disparity = 255 - (disparity * 255 / disparity_range_);
    cv::applyColorMap(disparity, disparity_color_map_, cv::COLORMAP_JET); // convert to colour map
    cv::Mat mask = (disparity == 0); // Binary mask where disparity is 0
    disparity_color_map_.setTo(cv::Vec3b(0, 0, 0), mask); // Set to black using the mask
    
    disparity_color_map_.copyTo(cv_image_disparity_.image);
    estimated_disparity_pub_.publish(cv_image_disparity_.toImageMsg());

    // cv::normalize(depth_map_, depth_map_, 0, 255, cv::NORM_MINMAX, CV_8U);
    // depth_map_.copyTo(cv_image_depth_.image);
    // estimated_depth_pub_.publish(cv_image_depth_.toImageMsg());

    auto end_time_disp = std::chrono::high_resolution_clock::now();
    auto duration_disp = std::chrono::duration_cast<std::chrono::microseconds>(end_time_disp - start_time_disp);

    ROS_INFO("Disp estimation took: %.2f milliseconds", duration_disp.count() / 1000.0);
}

void DVSStereo::syncCallback(const dvs_msgs::EventArray::ConstPtr &msg1, const dvs_msgs::EventArray::ConstPtr &msg2)
{

    if (msg1->events.empty() || msg2->events.empty()) {
        ROS_WARN("One or both DVS event arrays are empty!");
        return;
    }

    left_events_ = *msg1;
    right_events_ = *msg2;

    double left_ts = left_events_.header.stamp.toSec() + (left_events_.header.stamp.toNSec() / 1e9);
    double right_ts = right_events_.header.stamp.toSec() + (right_events_.header.stamp.toNSec() / 1e9);
    double diff = std::abs(left_ts - right_ts) * 1000;

    dvs_msgs::Event left_start = left_events_.events.front();
    dvs_msgs::Event left_end = left_events_.events.back();
    double left_start_ts_ = left_start.ts.toSec() + (left_start.ts.toNSec() / 1e9);
    double left_end_ts_ = left_end.ts.toSec() + (left_end.ts.toNSec() / 1e9);

    dvs_msgs::Event right_start = right_events_.events.front();
    dvs_msgs::Event right_end = right_events_.events.back();
    double right_start_ts_ = right_start.ts.toSec() + (right_start.ts.toNSec() / 1e9);
    double right_end_ts_ = right_end.ts.toSec() + (right_end.ts.toNSec() / 1e9);

    double common_start = std::max(left_start_ts_, right_start_ts_);
    double common_end = std::min(left_end_ts_, right_end_ts_);
    double common_time_slice = common_end - common_start;

    if (!(common_start < common_end))
    {
        ROS_WARN("No common time slice found!");
        return;
    }

    #ifdef DEBUG_MODE
    ROS_INFO("Received %lu left events and %lu right events",
        left_events_.events.size(), right_events_.events.size());
    ROS_INFO("Timestamp left_event = %.6f, Timestamp Right event = %.6f, difference: %.2fms", 
             left_ts, right_ts, diff);
    ROS_INFO("left_start_ts_ = %.6f, left_end = %.6f", left_start_ts_, left_end_ts_);
    ROS_INFO("right_start_ts_ = %.6f, right_end_ts_ = %.6f", right_start_ts_, right_end_ts_);
    ROS_INFO("common_start: %.6f, common_end: %.6f, common_time_slice: %.6f", 
             common_start, common_end, common_time_slice);
    #endif

    auto algo_start = high_resolution_clock::now();

    getSuitableSlice(left_events_, common_start, common_end);
    getSuitableSlice(right_events_, common_start, common_end);
    publishOnce();

    // loop_timer_ = t;
    auto algo_end = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(algo_end - algo_start);

    ROS_INFO("Algorithm execution time: %f ms, right event size: %lu, left event size: %lu",
        duration.count() / 1000.0, left_events_.events.size(), right_events_.events.size());
}

void DVSStereo::getSuitableSlice(dvs_msgs::EventArray &event_array, double start_time, double end_time)
{
    auto &events = event_array.events;
    int original_size = events.size(); // Store the original size

    // Remove front events (before start_time)
    auto start_it = std::find_if(events.begin(), events.end(), [&](const dvs_msgs::Event &event)
                                 {
        double ts = event.ts.toSec()  + (event.ts.toNSec() / 1e9);
        return ts > start_time; });

    // Erase everything before the first event within the time range
    // Compute number of erased events from the front
    int front_erased = std::distance(events.begin(), start_it);
    events.erase(events.begin(), start_it);

    // Remove back events (after end_time)
    auto end_it = std::find_if(events.rbegin(), events.rend(), [&](const dvs_msgs::Event &event)
                               {
        double ts = event.ts.toSec()  + (event.ts.toNSec() / 1e9);
        return ts < end_time; });

    // Compute number of erased events from the back
    int back_erased = std::distance(end_it.base(), events.end());
    // Erase everything after the last event within the time range
    events.erase(end_it.base(), events.end());

    // Total number of erased events
    int total_erased = front_erased + back_erased;
    #ifdef DEBUG_MODE
    ROS_INFO("Total number of events erased: %d", total_erased);
    #endif
}
