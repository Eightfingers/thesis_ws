#include "dvs_read_txt.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

DVSStereo::DVSStereo(ros::NodeHandle &nh, ros::NodeHandle nh_private) : nh_(nh), p_nh_(nh_private)
{

    nh_private.param<int>("camera_height", camera_height_, 11);
    nh_private.param<int>("camera_width", camera_width_, 11);
    nh_private.param<bool>("nearest_neighbour", do_NN_, false);
    nh_private.param<bool>("write_to_text", write_to_text_, false);
    nh_private.param<bool>("calc_disparity", calc_disparity_, false);
    nh_private.param<bool>("do_adaptive_window", do_adaptive_window_, true);

    nh_private.param<int>("small_block_size", small_block_size_, 7);
    nh_private.param<int>("medium_block_size", medium_block_size_, 9);
    nh_private.param<int>("large_block_size", large_block_size_, 11);

    nh_private.param<double>("age_penalty", age_penalty_, 0.1);
    nh_private.param<int>("msg_queue_size", msg_queue_size_, 1);

    nh_private.param<int>("disparity_range", disparity_range_, 40);
    nh_private.param<int>("disparity_step", disparity_step_, 1);
    nh_private.param<int>("window_block_size", window_block_size_, 11);
    nh_private.param<int>("window_step", window_step_size_, 2);
    nh_private.param<int>("window_fill_size", window_fill_size_, 1);
    nh_private.param<int>("NN_block_size", NN_block_size_, 10);
    nh_private.param<int>("NN_min_num_of_events", NN_min_num_of_events_, 3);

    // Initialize subscribers with message_filters
    left_event_sub_2.subscribe(nh_, "/left_events", 10);
    right_event_sub_2.subscribe(nh_, "/right_events", 10);

    // Create an Exact Time Synchronizer
    // Sync policy of q size
    sync_.reset(new Sync(MySyncPolicy(msg_queue_size_), left_event_sub_2, right_event_sub_2));
    sync_->registerCallback(boost::bind(&DVSStereo::syncCallback, this, _1, _2));
    // sync_->setAgePenalty(age_penalty_);  // Reduce age penalty (100 microseconds)
    // sync_->setMaxIntervalDuration(ros::Duration(0.010));  // 10 millisecond tolerance

    std::cout << "NN_min_num_of_events_.. " << NN_min_num_of_events_ << std::endl;
    std::cout << "Do NN.. " << do_NN_ << std::endl;

    image_transport::ImageTransport it_(nh);
    estimated_disparity_pub_ = it_.advertise("left_disparity", 1);

    cv_image_disparity_.encoding = "bgr8";
    cv_image_disparity_.image = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(0));

    event_image_left_ = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(0));
    event_image_left_polarity_ = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(left_empty_pixel_val_));
    event_image_left_polarity_remmaped_ = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(left_empty_pixel_val_));

    event_image_right_ = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(0));
    event_image_right_polarity_ = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(right_empty_pixel_val_));
    event_image_right_polarity_remmaped_ = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(right_empty_pixel_val_));

    disparity_color_map_ = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(0));

    sobel_x_ = cv::Mat::zeros(event_image_left_polarity_.size(), CV_64F);
    sobel_y_ = cv::Mat::zeros(event_image_left_polarity_.size(), CV_64F);
    mean_ = cv::Mat::zeros(event_image_left_polarity_.size(), CV_64F);
    mean_sq_ = cv::Mat::zeros(event_image_left_polarity_.size(), CV_64F);
    gradient_sq_ = cv::Mat::zeros(event_image_left_polarity_.size(), CV_64F);
    window_size_map_ = cv::Mat::zeros(event_image_left_polarity_.size(), CV_8U);

    // Stereo rectify
    cv::Size imageSize(camera_width_, camera_height_);

    cv::stereoRectify(K_0, dist_0, K_1, dist_1, imageSize, R, T, R1, R2, P1, P2, Q, 0);
    cv::initUndistortRectifyMap(K_0, dist_0, R1, P1, imageSize, CV_32FC1, map1_x, map1_y);
    cv::initUndistortRectifyMap(K_1, dist_1, R2, P2, imageSize, CV_32FC1, map2_x, map2_y);

    new_mid_x = (event_image_left_polarity_.cols - crop_width) / 2;  // Center horizontally
    new_mid_y = (event_image_left_polarity_.rows - crop_height) / 2; // Center vertically

    std::cout << "Finished init! " << std::endl;
    loop_timer_ = high_resolution_clock::now();
}

DVSStereo::~DVSStereo()
{
}

void DVSStereo::publishOnce(double start_time, double end_time)
{
    event_image_left_polarity_.setTo(left_empty_pixel_val_);
    event_image_left_polarity_remmaped_.setTo(left_empty_pixel_val_);

    event_image_right_polarity_.setTo(right_empty_pixel_val_);
    event_image_right_polarity_remmaped_.setTo(right_empty_pixel_val_);

    // Read, Write & Publish disparity ground truth values of the specified time slice
    auto start_time_read = std::chrono::high_resolution_clock::now();

    readEventArray(left_events_2,
                   event_image_left_polarity_,
                   map1_x,
                   map1_y);

    readEventArray(right_events_2,
                   event_image_right_polarity_,
                   map2_x,
                   map2_y);

    auto end_time_read = std::chrono::high_resolution_clock::now();
    auto duration_read = std::chrono::duration_cast<std::chrono::microseconds>(end_time_read - start_time_read);

    // std::cout << "Slicing left right event vector took: " << duration_read.count() / 1000.0 << " milliseconds" << std::endl;

    if (do_NN_)
    {
        applyNearestNeighborFilter(event_image_left_polarity_, left_empty_pixel_val_);
        applyNearestNeighborFilter(event_image_right_polarity_, right_empty_pixel_val_);
    }

    cv::remap(event_image_left_polarity_, event_image_left_polarity_remmaped_, map1_x, map1_y, cv::INTER_NEAREST);
    cv::remap(event_image_right_polarity_, event_image_right_polarity_remmaped_, map2_x, map2_y, cv::INTER_NEAREST);
    cv::Rect crop_region(new_mid_x, new_mid_y, crop_width, crop_height);

    // Crop the image
    event_image_left_polarity_remmaped_  = event_image_left_polarity_remmaped_(crop_region);
    event_image_right_polarity_remmaped_  = event_image_right_polarity_remmaped_(crop_region);
    
    if (do_adaptive_window_)
    {
        createAdaptiveWindowMap(event_image_left_polarity_remmaped_,
                                sobel_x_,
                                sobel_y_,
                                mean_,
                                mean_sq_,
                                gradient_sq_,
                                window_size_map_);
    }

    if (calc_disparity_)
    {
        calcPublishDisparity(
            event_image_left_polarity_remmaped_,
            event_image_right_polarity_remmaped_);
    }
}

void DVSStereo::readEventArray(dvs_msgs::EventArray &event_array, cv::Mat &event_image_polarity, cv::Mat &map1_x, cv::Mat &map1_y)
{
    for (int i = 0; i < event_array.events.size(); i++)
    {
        dvs_msgs::Event tmp = event_array.events.at(i);
        double current_ts = tmp.ts.toSec() + (tmp.ts.toNSec() * 1e-9);
        int row = tmp.y;
        int col = tmp.x;
        bool polarity = tmp.polarity;

        event_image_polarity.at<uchar>(row, col) = (uchar)polarity;
    }

    auto start_time_NN = std::chrono::high_resolution_clock::now();
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
    std::cout << "Nearest neighbor filtering took: " << duration_NN.count() / 1000.0 << " milliseconds" << std::endl;
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

    // tmp.copyTo(debug_image_.image);
    // debug_image_pub_.publish(debug_image_.toImageMsg());

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
                image_window_sizes.at<uchar>(i, j) = small_block_size_;
            }
            else
            {
                image_window_sizes.at<uchar>(i, j) = large_block_size_;
            }
        }
    }
}

void DVSStereo::calcPublishDisparity(
    cv::Mat &event_image_polarity_left,
    cv::Mat &event_image_polarity_right)
{
    cv::Mat disparity(event_image_polarity_left.size(), CV_64F, cv::Scalar(0));

    // Window-based disparity calculation
    auto start_time_disp = std::chrono::high_resolution_clock::now();

    int half_block = window_block_size_ / 2;
    double total_pixel = window_block_size_ * window_block_size_;

    int total_rows = event_image_polarity_left.rows - half_block - 1;
    int total_cols = event_image_polarity_left.cols - half_block - 1;

    for (int y = half_block; y < total_rows; y += window_step_size_)
    {
        for (int x = half_block; x < total_cols; x += window_step_size_)
        {
            if (event_image_polarity_left.at<uchar>(y, x) == left_empty_pixel_val_)
            {
                continue; // skip processing
            }

            // Window calculation
            int best_disparity = 0;
            double min_sad = 10000000;            
            if (do_adaptive_window_)
            {
                int window_size = window_size_map_.at<uchar>(y, x);
                half_block = window_size / 2;
                total_pixel = window_size * window_size;
            }

            // Compare blocks at different disparities
            for (int d = 0; d < disparity_range_; d += disparity_step_)
            {

                double sad = 0;
                double num_of_non_similar_pixels = 0;

                // Window for the current disparity
                for (int wy = -half_block; wy <= half_block; wy++)
                {
                    for (int wx = -half_block; wx <= half_block; wx++)
                    {
                        int left_polarity = event_image_polarity_left.at<uchar>(y + wy, x + wx);
                        int right_polarity = event_image_right_polarity_.at<uchar>(y + wy, x + wx - d);
                        if ((right_polarity != left_polarity))
                        {
                            num_of_non_similar_pixels++;
                        }
                    }
                }

                sad = num_of_non_similar_pixels / total_pixel;

                if (sad < min_sad)
                {
                    min_sad = sad;
                    best_disparity = d;
                }
            }
            if (best_disparity != 0)
            {
                double disparity_out = best_disparity * 255 / disparity_range_;
                // disparity around this pixel is the same.
                for (int i = y - window_fill_size_; i < y + window_fill_size_; i++)
                {
                    for (int j = x - window_fill_size_; j < x + window_fill_size_; j++)
                    {
                        disparity.at<double>(i, j) = disparity_out;
                    }
                }
            }
        }
    }

    auto end_time_disp = std::chrono::high_resolution_clock::now();
    auto duration_disp = std::chrono::duration_cast<std::chrono::microseconds>(end_time_disp - start_time_disp);
    std::cout << "Disp estimation took: " << duration_disp.count() / 1000.0 << " milliseconds" << std::endl;

    disparity.convertTo(disparity, CV_8U);                                // Convert from 64F to 8U
    cv::applyColorMap(disparity, disparity_color_map_, cv::COLORMAP_JET); // convert to colour map
    disparity_color_map_.copyTo(cv_image_disparity_.image);
    estimated_disparity_pub_.publish(cv_image_disparity_.toImageMsg());
}

void DVSStereo::syncCallback(const dvs_msgs::EventArray::ConstPtr &msg1, const dvs_msgs::EventArray::ConstPtr &msg2)
{
    left_events_2 = *msg1;
    right_events_2 = *msg2;

    double left_ts = left_events_2.header.stamp.toSec() + (left_events_2.header.stamp.toNSec() / 1e9);
    double right_ts = right_events_2.header.stamp.toSec() + (right_events_2.header.stamp.toNSec() / 1e9);
    double diff = std::abs(left_ts - right_ts) * 1000;

    dvs_msgs::Event left_start = left_events_2.events.front();
    dvs_msgs::Event left_end = left_events_2.events.back();
    double left_start_ts_ = left_start.ts.toSec() + (left_start.ts.toNSec() / 1e9);
    double left_end_ts_ = left_end.ts.toSec() + (left_end.ts.toNSec() / 1e9);

    dvs_msgs::Event right_start = right_events_2.events.front();
    dvs_msgs::Event right_end = right_events_2.events.back();
    double right_start_ts_ = right_start.ts.toSec() + (right_start.ts.toNSec() / 1e9);
    double right_end_ts_ = right_end.ts.toSec() + (right_end.ts.toNSec() / 1e9);

    double common_start = std::max(left_start_ts_, right_start_ts_);
    double common_end = std::min(left_end_ts_, right_end_ts_);
    double common_time_slice = common_end - common_start;

    if (!(common_start < common_end))
    {
        std::cout << "!!!!!!!!!!!!!!!!!!!!!No common slice found!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        return;
    }

    std::cout << "Timestamp left_event = " << left_ts << " , Timestamp Right event = " << right_ts << ", difference: " << diff << "ms" << std::endl;
    std::cout << "left_start_ts_ = " << left_start_ts_ << " , left_end = " << left_end_ts_ << std::endl;
    std::cout << "right_start_ts_ = " << right_start_ts_ << " , right_end_ts_ = " << right_end_ts_ << std::endl;
    std::cout << "common_start: " << common_start << ", common_end:" << common_end << ", common_time_slice:" << common_time_slice << std::endl;

    auto algo_start = high_resolution_clock::now();

    getSuitableSlice(left_events_2, common_start, common_end);
    getSuitableSlice(right_events_2, common_start, common_end);
    publishOnce(0, 1);

    // loop_timer_ = t;
    auto algo_end = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(algo_end - algo_start);

    // std::cout << "Algorithm execution time: " << duration.count() / 1000.0 << " milliseconds" << std::endl;
    // std::cout << "Left event size: " << left_events_2.events.size() << " , Right event size: " << right_events_2.events.size() << std::endl;
}

void DVSStereo::loopOnce()
{
    if (left_events_2.events.size() > 1 && right_events_2.events.size() > 1)
    {
        auto algo_start = high_resolution_clock::now();

        publishOnce(0, 1);
        // loop_timer_ = t;
        auto algo_end = high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(algo_end - algo_start);

        std::cout << "Algorithm execution time: " << duration.count() / 1000.0 << " milliseconds" << std::endl;
        std::cout << "Left event size: " << left_events_2.events.size() << " , Right event size: " << right_events_2.events.size() << std::endl;
    }
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
    std::cout << "Total events erased: " << total_erased << std::endl;
}
