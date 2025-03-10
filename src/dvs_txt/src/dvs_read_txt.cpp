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

DVSReadTxt::DVSReadTxt(ros::NodeHandle &nh, ros::NodeHandle nh_private) : nh_(nh), p_nh_(nh_private)
{
    left_event_arr_pub_ = nh_.advertise<dvs_msgs::EventArray>("/left_events", 1000);
    right_event_arr_pub_ = nh_.advertise<dvs_msgs::EventArray>("/right_events", 1000);

    nh_private.param<bool>("publish_slice", publish_slice_, false);
    nh_private.param<bool>("nearest_neighbour", do_NN_, false);
    nh_private.param<bool>("write_to_text", write_to_text_, false);
    nh_private.param<bool>("file_is_txt", file_is_txt_, true);
    nh_private.param<bool>("do_adaptive_window", do_adaptive_window_, true);
    nh_private.param<bool>("rectify", rectify_, true);

    nh_private.param<int>("camera_height", camera_height_, 11);
    nh_private.param<int>("camera_width", camera_width_, 11);

    nh_private.param<std::string>("file_path", event_txt_file_path_, "/home/asus/thesis_ws/src/dvs_txt/resource/box2.txt");
    nh_private.param<std::string>("left_csv_file_path", left_event_csv_file_path_, "/home/asus/thesis_ws/src/dvs_txt/resource/master.csv");
    nh_private.param<std::string>("right_csv_file_path", right_event_csv_file_path_, "/home/asus/thesis_ws/src/dvs_txt/resource/slave.csv");

    nh_private.param<double>("loop_rate", loop_rate_, 0.10);
    nh_private.param<double>("slice_start_time", slice_start_time_, 0.0);
    nh_private.param<double>("slice_end_time", slice_end_time_, 0.1);
    nh_private.param<int>("disparity_range", disparity_range_, 40);
    nh_private.param<int>("NN_block_size", NN_block_size_, 11);
    nh_private.param<int>("NN_min_num_of_events", NN_min_num_of_events_, 3);

    nh_private.param<int>("threshold_edge", threshold_edge_, 50);

    nh_private.param<int>("small_block_size", small_block_size_, 7);
    nh_private.param<int>("medium_block_size", medium_block_size_, 9);
    nh_private.param<int>("large_block_size", large_block_size_, 11);

    std::cout << "NN_min_num_of_events_.. " << NN_min_num_of_events_ << std::endl;
    std::cout << "do_NN_.. " << do_NN_ << std::endl;

    std::cout << "Reading text file.. " << event_txt_file_path_ << std::endl;
    std::cout << "Reading csv file: " << left_event_csv_file_path_ << ", right_file_path:" << right_event_csv_file_path_ << std::endl;
    std::cout << "loop_rate_.. " << loop_rate_ << std::endl;

    image_transport::ImageTransport it_(nh);
    img_pub_disparity_gt_left_ = it_.advertise("disparity_gt_left_", 1);
    img_pub_disparity_gt_right_ = it_.advertise("disparity_gt_right_", 1);
    debug_image_pub_ = it_.advertise("debug_img", 1);

    gt_cv_image_.encoding = "bgr8";
    gt_cv_image_.image = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(0));

    debug_image_.encoding = "mono8";
    debug_image_.image = cv::Mat(camera_height_, camera_width_, CV_32F, cv::Scalar(0));

    rect_left_image_.encoding = "mono8";
    rect_left_image_.image = cv::Mat(camera_height_, camera_width_, CV_32F, cv::Scalar(0));
    rect_left_image_pub_ = it_.advertise("left_rect", 1);

    rect_right_image_.encoding = "mono8";
    rect_right_image_.image = cv::Mat(camera_height_, camera_width_, CV_32F, cv::Scalar(0));
    rect_right_image_pub_ = it_.advertise("right_rect", 1);

    cv_image_disparity_.encoding = "bgr8";
    cv_image_disparity_.image = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(0));
    disparity_pub_ = it_.advertise("left_disparity", 1);

    event_image_left_polarity_ = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(left_empty_pixel_val_));
    event_image_left_polarity_remmaped_ = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(left_empty_pixel_val_));
    disparity_gt_left_ = cv::Mat(camera_height_, camera_width_, CV_64F, cv::Scalar(0));

    event_image_right_polarity_ = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(right_empty_pixel_val_));
    event_image_right_polarity_remmaped_ = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(right_empty_pixel_val_));
    disparity_gt_right_ = cv::Mat(camera_height_, camera_width_, CV_64F, cv::Scalar(0));

    color_map_ = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(0));

    sobel_x_ = cv::Mat::zeros(event_image_left_polarity_.size(), CV_64F);
    sobel_y_ = cv::Mat::zeros(event_image_left_polarity_.size(), CV_64F);
    mean_ = cv::Mat::zeros(event_image_left_polarity_.size(), CV_64F);
    mean_sq_ = cv::Mat::zeros(event_image_left_polarity_.size(), CV_64F);
    gradient_sq_ = cv::Mat::zeros(event_image_left_polarity_.size(), CV_64F);
    window_size_map_ = cv::Mat::zeros(event_image_left_polarity_.size(), CV_8U);

    gt_disparity_left_file_.open(gt_disparity_left_path_);
    gt_disparity_right_file_.open(gt_disparity_right_path_);
    disparity_file_.open(disparity_path_);

    if (rectify_)
    {
        cv::Size imageSize(camera_width_, camera_height_);

        cv::stereoRectify(K_0, dist_0, K_1, dist_1, imageSize, R, T, R1, R2, P1, P2, Q, 0);
        cv::initUndistortRectifyMap(K_0, dist_0, R1, P1, imageSize, CV_32FC1, map1_x, map1_y);
        cv::initUndistortRectifyMap(K_1, dist_1, R2, P2, imageSize, CV_32FC1, map2_x, map2_y);
        new_mid_x = (event_image_left_polarity_.cols - crop_width) / 2;  // Center horizontally
        new_mid_y = (event_image_left_polarity_.rows - crop_height) / 2; // Center vertically
    }

    if (!gt_disparity_left_file_.is_open() || !gt_disparity_right_file_.is_open() || !disparity_file_.is_open())
    {
        ROS_ERROR("Failed to open one or more output files");
    }
}

DVSReadTxt::~DVSReadTxt()
{
    gt_disparity_left_file_.close();
    gt_disparity_right_file_.close();
    disparity_file_.close();
}

void DVSReadTxt::readFile()
{
    // e(ts,x,y,p,l,Gt_d), ts is timestamps, x and y are the image coodrinates of the events,p is the polarity, l is the left/right label(0 is left and 1 is right),Gt_d is the ground truth of disparity
    if (file_is_txt_)
    {
        std::ifstream event_dataset_file(event_txt_file_path_);
        if (!event_dataset_file.is_open())
        {
            std::cerr << "Error: Unable to open file: " << event_txt_file_path_ << std::endl;
        }

        std::string line;
        while (std::getline(event_dataset_file, line))
        {
            std::istringstream iss(line);
            int ts;
            int x;
            int y;
            int polarity;
            int left_right; // 0 is left, 1 is right
            double gt;
            if (!(iss >> ts >> x >> y >> polarity >> left_right >> gt))
            {
                std::cerr << "Error: Malformed line in file: " << line << std::endl;
                continue; // Skip malformed lines
            }
            else
            {
                if (left_right == 0)
                {
                    dvs_msgs::Event event;
                    ros::Time ros_ts;
                    double seconds = ts / 1e6; // Convert to seconds as double
                    // Create a ros::Time object
                    ros_ts.fromSec(seconds);
                    event.ts = ros_ts;
                    event.x = x;
                    event.y = y;
                    event.polarity = (bool)polarity;
                    left_events_.emplace_back(event);
                    left_disparity_values_.emplace_back(gt);
                }
                else
                {
                    dvs_msgs::Event event;
                    ros::Time ros_ts;
                    double seconds = ts / 1e6; // Convert to seconds as double
                    ros_ts.fromSec(seconds);
                    event.ts = ros_ts;
                    event.x = x;
                    event.y = y;
                    event.polarity = (bool)polarity;
                    right_events_.emplace_back(event);
                    right_disparity_values_.emplace_back(gt);
                }
            }
        }
        event_dataset_file.close();
        std::cout << "Finished reading, left events size:" << left_events_.size() << ", right events size:" << right_events_.size() << std::endl;
    }
    else
    {
        std::ifstream left_event_dataset_file(left_event_csv_file_path_);
        if (!left_event_dataset_file.is_open())
        {
            std::cerr << "Error: Unable to open file: " << left_event_csv_file_path_ << std::endl;
        }

        std::string line;
        while (std::getline(left_event_dataset_file, line))
        {
            std::istringstream iss(line);
            int ts;
            int x;
            int y;
            int polarity;
            char comma;
            if (!(iss >> x >> comma >> y >> comma >> polarity >> comma >> ts))
            {
                std::cerr << "Error: Malformed line in file: " << line << std::endl;
                continue; // Skip malformed lines
            }
            else
            {
                dvs_msgs::Event event;
                ros::Time ros_ts;
                double seconds = ts / 1e6; // Convert to seconds as double
                ros_ts.fromSec(seconds);
                event.ts = ros_ts;
                event.x = x;
                event.y = y;
                event.polarity = (bool)polarity;
                left_events_.emplace_back(event);
                // Meta vision DVS does not have disparity values
                left_disparity_values_.emplace_back(0);
            }
        }
        left_event_dataset_file.close();

        std::ifstream right_event_dataset_file(right_event_csv_file_path_);
        if (!right_event_dataset_file.is_open())
        {
            std::cerr << "Error: Unable to open file: " << right_event_csv_file_path_ << std::endl;
        }

        std::string line2;
        while (std::getline(right_event_dataset_file, line2))
        {
            std::istringstream iss(line2);
            int ts;
            int x;
            int y;
            int polarity;
            char comma;
            if (!(iss >> x >> comma >> y >> comma >> polarity >> comma >> ts))
            {
                std::cerr << "Error: Malformed line in file: " << line << std::endl;
                continue; // Skip malformed lines
            }
            else
            {
                dvs_msgs::Event event;
                ros::Time ros_ts;
                double seconds = ts / 1e6; // Convert to seconds as double
                ros_ts.fromSec(seconds);
                event.ts = ros_ts;
                event.x = x;
                event.y = y;
                event.polarity = (bool)polarity;
                right_events_.emplace_back(event);
                // Meta vision DVS does not have disparity values
                right_disparity_values_.emplace_back(0);
            }
        }

        right_event_dataset_file.close();
        std::cout << "Finished reading, left events size:" << left_events_.size() << ", right events size:" << right_events_.size() << std::endl;
    }
    loop_timer_ = high_resolution_clock::now();
}

void DVSReadTxt::readTimeSliceEventsVec(
    int &start_index,
    double start_time,
    double end_time,
    std::vector<dvs_msgs::Event> &event_vector,
    std::vector<double> &disparity_vector,
    dvs_msgs::EventArray &event_arr,
    cv::Mat &event_image_polarity,
    cv::Mat &event_image_disp_gt,
    cv::Mat &map1_x,
    cv::Mat &map1_y)
{
    int num_same_pixels = 0 ;
    // int start_index2 = 0;
    // bool found = false;

    for (int i = start_index; i < event_vector.size(); i++)
    {
        dvs_msgs::Event tmp = event_vector.at(i);
        double current_ts = tmp.ts.toSec() + (tmp.ts.toNSec() / 1e9);
        double disparity = disparity_vector.at(i);

        if (current_ts > start_time)
        {
            // if (!found)
            // {
            //     start_index2 = i;
            //     found = true;
            // }
            
            
            // found the time start
            event_arr.events.emplace_back(tmp);

            int row = tmp.y;
            int col = tmp.x;
            uchar polarity;
            if (tmp.polarity)
            {
                polarity = 255;
            }
            else
            {
                polarity = 0;
            }

            event_image_disp_gt.at<double>(row, col) = disparity;

            if (event_image_polarity.at<uchar>(row, col) == 255 || event_image_polarity.at<uchar>(row, col) == 0)
            {
                num_same_pixels += 1;
            }
            event_image_polarity.at<uchar>(row, col) = polarity;
            if (current_ts > end_time)
            {
                start_index = i;
                // exit loop as this event has exceeded end time
                break;
            }
        }
    }
    // std::cout << "Total number of pixels :" << start_index - start_index2 << std::endl;
    // std::cout << "num_same_pixels:" << num_same_pixels << std::endl;
}

void DVSReadTxt::applyNearestNeighborFilter(cv::Mat &event_image_pol, int value_of_empty_cell)
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

void DVSReadTxt::createAdaptiveWindowMap(cv::Mat &event_image_pol,
                                         cv::Mat &sobel_x,
                                         cv::Mat &sobel_y,
                                         cv::Mat &mean,
                                         cv::Mat &mean_sq,
                                         cv::Mat &gradient_sq,
                                         cv::Mat &image_window_sizes)
{
    cv::Sobel(event_image_pol, sobel_x, CV_64F, 1, 0, 3);
    cv::Sobel(event_image_pol, sobel_y, CV_64F, 0, 1, 3);

    cv::Mat gradient_magnitude;
    cv::magnitude(sobel_x, sobel_y, gradient_magnitude);

    // Convert to displayable range
    cv::Mat tmp;
    tmp = gradient_magnitude.clone();
    cv::normalize(tmp, tmp, 0, 255, cv::NORM_MINMAX, CV_8U);

    tmp.copyTo(debug_image_.image);
    debug_image_pub_.publish(debug_image_.toImageMsg());

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

void DVSReadTxt::publishGTDisparity(cv::Mat &disparity_gt, image_transport::Publisher &disparity_gt_image_pub)
{
    cv::Mat tmp = disparity_gt.clone();
    tmp = (tmp / 30) * 255;
    tmp.convertTo(tmp, CV_8U); // Convert double to integer

    cv::applyColorMap(tmp, color_map_, cv::COLORMAP_JET); // Convert to colour map
    color_map_.copyTo(gt_cv_image_.image);
    disparity_gt_image_pub.publish(gt_cv_image_.toImageMsg());
}

void DVSReadTxt::calcPublishDisparity(
    cv::Mat &event_image_polarity_left,
    cv::Mat &event_image_polarity_right,
    cv::Mat &left_gt_disparity,
    std::ofstream &file)
{
    cv::Mat disparity(event_image_polarity_left.size(), CV_64F, cv::Scalar(0));

    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open disparity file for writing!" << std::endl;
    }

    int num_of_polarities = 0;
    int wtf = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    int half_block = large_block_size_ / 2;
    double total_pixel = large_block_size_ * large_block_size_;
    int total_rows = event_image_polarity_left.rows - half_block -1;
    int total_cols = event_image_polarity_left.cols - half_block -1;

    // for (int y = 0; y < event_image_polarity_left.rows; y++)
    // {
    //     for (int x = 0; x < event_image_polarity_left.cols; x++)
    //     {
    //         if (event_image_polarity_left.at<uchar>(y, x) == left_empty_pixel_val_)
    //         {
    //             continue; // skip processing
    //         }
    //         num_of_polarities ++;
    //     }
    // }

    for (int y = half_block; y < total_rows; y++)
    {
        for (int x = half_block; x < total_cols; x++)
        {
            if (event_image_polarity_left.at<uchar>(y, x) == left_empty_pixel_val_)
            {
                continue; // skip processing
            }
            
            if (event_image_polarity_left.at<uchar>(y, x) != 255 &&  event_image_polarity_left.at<uchar>(y, x) != 0)
            {
                std::cout << "ERRROR!!! left event_pol_image should only contain 255 or 0 or 127:" << event_image_polarity_left.at<uchar>(y, x) <<std::endl;
            }

            // Window calculation
            int best_disparity = 0;
            double min_cost = 10000000;

            if (do_adaptive_window_)
            {
                int window_size = window_size_map_.at<uchar>(y, x);
                half_block = window_size / 2;
                total_pixel = window_size * window_size;
            }

            // Compare blocks at different disparities
            for (int d = 0; d < disparity_range_; d++)
            {
                double cost = 0;
                double num_of_similar_pixels = 0;
                double num_of_non_similar_pixels = 0;

                // Compute costs for the current disparity
                for (int wy = -half_block; wy <= half_block; wy++)
                {
                    for (int wx = -half_block; wx <= half_block; wx++)
                    {
                        int left_polarity = event_image_polarity_left.at<uchar>(y + wy, x + wx);
                        int right_polarity = event_image_right_polarity_.at<uchar>(y + wy, x + wx - d);
                        // if (left_polarity != right_polarity)
                        // {
                        //     num_of_non_similar_pixels++;
                        // }

                        if (left_polarity != left_empty_pixel_val_ && right_polarity != right_empty_pixel_val_ )
                        {
                            num_of_non_similar_pixels++;
                        }
                    }
                }

                cost = (total_pixel - num_of_non_similar_pixels) / total_pixel;
                // if (y == 14 && x == 56)
                // {
                //     std::cout << "cost:" << cost << ", num_of_non_similar_pixels:" << num_of_non_similar_pixels << ", total_pixel" << total_pixel << std::endl;
                // }
                if (cost < min_cost)
                {
                    min_cost = cost;
                    best_disparity = d;
                }
            }

            if (best_disparity != 0)
            {
                double gt_disparity = left_gt_disparity.at<double>(y, x);
                file << y << "," << x << "," << best_disparity << "," << gt_disparity << "," << min_cost << "\n";
                disparity.at<double>(y, x) = (best_disparity * 255 / disparity_range_);
            }
            // else
            // {
            //     std::cout << "best_disparity:" << best_disparity << " at: " << y << "," << x << std::endl;
            //     wtf++;
            // }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Disp calculation took: " << duration.count() / 1000.0 << " milliseconds" << std::endl;

    // std::cout << "num of polarities: " << num_of_polarities << std::endl;
    // std::cout << "wtf: " << wtf << std::endl;

    disparity.convertTo(disparity, CV_8U);                      // Convert from 64F to 8U
    cv::applyColorMap(disparity, color_map_, cv::COLORMAP_JET); // convert to colour map
    color_map_.copyTo(cv_image_disparity_.image);
    disparity_pub_.publish(cv_image_disparity_.toImageMsg());
}

void DVSReadTxt::sparseRemap(cv::Mat &event_image, cv::Mat &remapped_image, cv::Mat &map1_x, cv::Mat &map1_y, int empty_pixel_value)
{
    for (int i = 0; i < event_image.rows; i++)
    {
        for (int j = 0; j < event_image.cols; j++)
        {
            if (event_image.at<uchar>(i, j) == empty_pixel_value)
            {
                continue; // skip processing
            }

            float col_dst = map1_x.at<float>(i, j);
            float row_dst = map1_y.at<float>(i, j);

            // Ensure destination coordinates are within the image bounds
            if (col_dst >= 0 && col_dst < event_image.cols && row_dst >= 0 && row_dst < event_image.rows)
            {
                // Perform interpolation (nearest neighbor in this case)
                int col_new = static_cast<int>(col_dst);
                int row_new = static_cast<int>(row_dst);

                // Assign the new pixel value
                remapped_image.at<uchar>(row_new, col_new) = event_image.at<uchar>(i, j);
            }
        }
    }
}

void DVSReadTxt::publishOnce(double start_time, double end_time)
{
    dvs_msgs::EventArray left_arr;
    dvs_msgs::EventArray right_arr;

    event_image_left_polarity_.setTo(left_empty_pixel_val_);
    disparity_gt_left_.setTo(0);

    event_image_right_polarity_.setTo(right_empty_pixel_val_);
    disparity_gt_right_.setTo(0);

    if (left_events_.size() > 1 && right_events_.size() > 1)
    {
        readTimeSliceEventsVec(
            left_event_index_,
            start_time,
            end_time,
            left_events_,
            left_disparity_values_,
            left_arr,
            event_image_left_polarity_,
            disparity_gt_left_,
            map1_x,
            map1_y);

        readTimeSliceEventsVec(
            right_event_index_,
            start_time,
            end_time,
            right_events_,
            right_disparity_values_,
            right_arr,
            event_image_right_polarity_,
            disparity_gt_right_,
            map2_x,
            map2_x);

        // Apply Nearest Neighbor filtering
        if (do_NN_)
        {
            applyNearestNeighborFilter(event_image_left_polarity_, left_empty_pixel_val_);
            applyNearestNeighborFilter(event_image_right_polarity_, right_empty_pixel_val_);
        }

        auto rect_start_time = std::chrono::high_resolution_clock::now();
        if (rectify_)
        {
            // For some reason this causes an error in the resultant cropped image ?
            // cv::remap(event_image_left_polarity_, event_image_left_polarity_, map1_x, map1_y, cv::INTER_NEAREST);
            // cv::remap(event_image_right_polarity_, event_image_right_polarity_, map2_x, map2_y, cv::INTER_NEAREST);

            cv::remap(event_image_left_polarity_, event_image_left_polarity_remmaped_, map1_x, map1_y, cv::INTER_NEAREST);
            cv::remap(event_image_right_polarity_, event_image_right_polarity_remmaped_, map2_x, map2_y, cv::INTER_NEAREST);
            cv::Rect crop_region(new_mid_x, new_mid_y, crop_width, crop_height);

            // Crop the image
            event_image_left_polarity_remmaped_  = event_image_left_polarity_remmaped_(crop_region);
            event_image_right_polarity_remmaped_  = event_image_right_polarity_remmaped_(crop_region);

            event_image_left_polarity_remmaped_.copyTo(rect_left_image_.image);
            rect_left_image_pub_.publish(rect_left_image_.toImageMsg());
            event_image_right_polarity_remmaped_.copyTo(rect_right_image_.image);
            rect_right_image_pub_.publish(rect_right_image_.toImageMsg());
        }
        else
        {
            // Nothing happens
            event_image_left_polarity_remmaped_= event_image_left_polarity_;
            event_image_right_polarity_remmaped_ = event_image_right_polarity_;
        }

        auto rect_end_time = std::chrono::high_resolution_clock::now();
        auto rect_duration = std::chrono::duration_cast<std::chrono::microseconds>(rect_end_time - rect_start_time);
        std::cout << "Rectify calculation took: " << rect_duration.count() / 1000.0 << " milliseconds" << std::endl;

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

        publishGTDisparity(
            disparity_gt_left_,
            img_pub_disparity_gt_left_);

        publishGTDisparity(
            disparity_gt_right_,
            img_pub_disparity_gt_right_);

        calcPublishDisparity(
            event_image_left_polarity_remmaped_,
            event_image_right_polarity_remmaped_,
            disparity_gt_left_,
            disparity_file_);
    }
    
    // Publish event array data
    left_arr.header.stamp = ros::Time::now();
    right_arr.header.stamp = ros::Time::now();

    left_arr.width = camera_width_;
    left_arr.height = camera_height_;

    right_arr.width = camera_width_;
    right_arr.height = camera_height_;

    left_event_arr_pub_.publish(left_arr);
    right_event_arr_pub_.publish(right_arr);

}

void DVSReadTxt::loopOnce()
{
    auto t = high_resolution_clock::now();
    if (first_loop_)
    {
        first_loop_ = false;
        time_start_ = t;
    }

    duration<double> time_elapsed = t - loop_timer_;
    duration<double> total_time_elapsed = t - time_start_;

    if (time_elapsed.count() > loop_rate_)
    {
        loop_timer_ = t;
        dvs_msgs::Event tmp = left_events_.at(0);
        double start_offset = tmp.ts.toSec() + (tmp.ts.toNSec() / 1e9);
        auto algo_start = high_resolution_clock::now();
        if (publish_slice_)
        {
            // need to reset index ...
            left_event_index_ = 0;
            right_event_index_ = 0;
            publishOnce(start_offset + slice_start_time_, start_offset + slice_end_time_);
        }
        else
        {
            publishOnce(start_offset + total_time_elapsed.count(), start_offset + total_time_elapsed.count() + loop_rate_);
        }
        auto algo_end = high_resolution_clock::now();
        duration<double> algo_duration = algo_end - algo_start;

        std::cout << "Algorithm execution time: " << algo_duration.count() * 1000 << " milliseconds" << std::endl;
    }
}
