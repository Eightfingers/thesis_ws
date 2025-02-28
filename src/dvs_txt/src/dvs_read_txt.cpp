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
    left_event_arr_pub_ = nh_.advertise<dvs_msgs::EventArray>("/republished_left/events_dvs", 1000);
    right_event_arr_pub_ = nh_.advertise<dvs_msgs::EventArray>("/republished_right/events_dvs", 1000);

    nh_private.param<int>("camera_height", camera_height_, 11);
    nh_private.param<int>("camera_width", camera_width_, 11);
    nh_private.param<bool>("publish_slice", publish_slice_, false);
    nh_private.param<bool>("nearest_neighbour", do_NN_, false);
    nh_private.param<bool>("write_to_text", write_to_text_, false);
    nh_private.param<bool>("calc_disparity", calc_disparity_, false);
    nh_private.param<bool>("file_is_txt", file_is_txt_, true);

    nh_private.param<std::string>("file_path", event_txt_file_path_, "/home/asus/thesis_ws/src/dvs_txt/resource/box2.txt");
    nh_private.param<std::string>("left_csv_file_path", left_event_csv_file_path_, "/home/asus/thesis_ws/src/dvs_txt/resource/master.csv");
    nh_private.param<std::string>("right_csv_file_path", right_event_csv_file_path_, "/home/asus/thesis_ws/src/dvs_txt/resource/slave.csv");

    nh_private.param<double>("loop_rate", loop_rate_, 0.10);
    nh_private.param<double>("slice_start_time", slice_start_time_, 0.0);
    nh_private.param<double>("slice_end_time", slice_end_time_, 0.1);
    nh_private.param<int>("disparity_range", disparity_range_, 40);
    nh_private.param<int>("window_block_size", window_block_size_, 11);
    nh_private.param<int>("NN_block_size", NN_block_size_, 11);
    nh_private.param<int>("NN_min_num_of_events", NN_min_num_of_events_, 3);

    std::cout << "NN_min_num_of_events_.. " << NN_min_num_of_events_ << std::endl;

    std::cout << "Reading text file.. " << event_txt_file_path_ << std::endl;
    std::cout << "Reading csv file: " << left_event_csv_file_path_ << ", right_file_path:" << right_event_csv_file_path_ << std::endl;
    std::cout << "loop_rate_.. " << loop_rate_ << std::endl;

    image_transport::ImageTransport it_(nh);
    img_pub_disparity_gt_left_ = it_.advertise("disparity_gt_left_", 1);
    img_pub_disparity_gt_right_ = it_.advertise("disparity_gt_right_", 1);

    cv_image_.encoding = "bgr8";
    cv_image_.image = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(0));

    cv_image_disparity_.encoding = "bgr8";
    cv_image_disparity_.image = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(0));
    sad_disparity_pub_ = it_.advertise("left_disparity", 1);

    event_image_left_ = cv::Mat(camera_height_, camera_width_, CV_64F, cv::Scalar(0));
    event_image_left_polarity_ = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(8));
    event_image_left_disparity_gt_ = cv::Mat(camera_height_, camera_width_, CV_64F, cv::Scalar(0));
    disparity_gt_left_  = cv::Mat(camera_height_, camera_width_, CV_64F, cv::Scalar(0));

    event_image_right_ = cv::Mat(camera_height_, camera_width_, CV_64F, cv::Scalar(0));
    event_image_right_polarity_ = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(9));
    event_image_right_disparity_gt_ = cv::Mat(camera_height_, camera_width_, CV_64F, cv::Scalar(0));
    disparity_gt_right_  = cv::Mat(camera_height_, camera_width_, CV_64F, cv::Scalar(0));

    color_map_ = cv::Mat(camera_height_, camera_width_, CV_8U, cv::Scalar(0));

    gt_disparity_left_file_.open(gt_disparity_left_path_);
    gt_disparity_right_file_.open(gt_disparity_right_path_);
    disparity_file_.open(disparity_path_);

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

void DVSReadTxt::publishOnce(double start_time, double end_time)
{
    dvs_msgs::EventArray left_arr;
    dvs_msgs::EventArray right_arr;

    event_image_left_polarity_.setTo(8);
    event_image_left_.setTo(0);
    disparity_gt_left_.setTo(0);

    event_image_right_polarity_.setTo(9);
    event_image_right_.setTo(0);
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
            event_image_left_,
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
            event_image_right_,
            event_image_right_polarity_,
            disparity_gt_right_,
            map2_x,
            map2_x);

        publishGTDisparity(
            disparity_gt_left_,
            img_pub_disparity_gt_left_);

        publishGTDisparity(
            disparity_gt_right_,
            img_pub_disparity_gt_right_);

        calcPublishDisparity(
            event_image_left_,
            event_image_left_polarity_,
            event_image_right_,
            event_image_right_polarity_,
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

void DVSReadTxt::readTimeSliceEventsVec(
    int &start_index,
    double start_time,
    double end_time,
    std::vector<dvs_msgs::Event> &event_vector,
    std::vector<double> &disparity_vector,
    dvs_msgs::EventArray &event_arr,
    cv::Mat &event_image,
    cv::Mat &event_image_polarity,
    cv::Mat &event_image_disp_gt,
    cv::Mat &map1_x,
    cv::Mat &map1_y)
{
    for (int i = start_index; i < event_vector.size(); i++)
    {
        dvs_msgs::Event tmp = event_vector.at(i);
        double current_ts = tmp.ts.toSec();
        double disparity = disparity_vector.at(i);

        if (current_ts > start_time)
        {
            // found the time start
            event_arr.events.emplace_back(tmp);

            int row = tmp.y;
            int col = tmp.x;
            bool polarity = tmp.polarity;

            event_image.at<double>(row, col) += 1;
            event_image_disp_gt.at<double>(row, col) = disparity;
            event_image_polarity.at<uchar>(row, col) = (uchar)polarity;
            if (current_ts > end_time)
            {
                // exit loop as this event has exceeded end time
                break;
            }
        }
    }

    // Nearest neighbour filter
    auto start_time_NN = std::chrono::high_resolution_clock::now();
    for (int y = NN_block_size_ / 2; y < event_image.rows - NN_block_size_ / 2; y++)
    {
        for (int x = NN_block_size_ / 2; x < event_image.cols - NN_block_size_ / 2; x++)
        {
            if (event_image.at<double>(y, x) == 0)
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

                        if (event_image.at<double>(y + wy, x + wx) != 0)
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
                    event_image.at<double>(y, x) = 0;
                    // std::cout << "Removed isolated event at, (" << x << "," << y << ")" << std::endl;
                }
                else
                {
                    // left_events_.emplace_back(msg->events[i]);
                }
            }
        }
    }

    auto end_time_NN = std::chrono::high_resolution_clock::now();
    auto duration_NN = std::chrono::duration_cast<std::chrono::microseconds>(end_time_NN - start_time_NN);
    std::cout << "Nearest neighbor filtering took: " << duration_NN.count() / 1000.0 << " milliseconds" << std::endl;
}

void DVSReadTxt::publishGTDisparity(cv::Mat &disparity_gt, image_transport::Publisher &disparity_gt_image_pub)
{
    cv::Mat tmp = disparity_gt.clone();
    tmp = (tmp / 30) * 255;
    tmp.convertTo(tmp, CV_8U); // Convert double to integer

    cv::applyColorMap(tmp, color_map_, cv::COLORMAP_JET); // Convert to colour map
    color_map_.copyTo(cv_image_.image);
    disparity_gt_image_pub.publish(cv_image_.toImageMsg());
}

void DVSReadTxt::calcPublishDisparity(
    cv::Mat &event_image_left,
    cv::Mat &event_image_polarity_left,
    cv::Mat &event_image_right,
    cv::Mat &event_image_polarity_right,
    cv::Mat &left_gt_disparity,
    std::ofstream &file)
{
    cv::Mat disparity(event_image_left_.size(), CV_64F, cv::Scalar(0));

    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open disparity file for writing!" << std::endl;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    const int half_block = window_block_size_ / 2;
    const double total_pixel = window_block_size_ * window_block_size_;

    for (int y = half_block; y < event_image_left.rows - half_block; y++)
    {
        for (int x = half_block; x < event_image_left.cols - half_block; x++)
        {
            if (event_image_left.at<double>(y, x) == 0)
            {
                continue; // skip processing
            }

            // Window calculation
            int best_disparity = 0;
            double min_cost = 10000000;
            bool out_of_bounds = false;

            // Compare blocks at different disparities
            for (int d = 0; d < disparity_range_; d++)
            {
                if (x - d - half_block < 0)
                {
                    out_of_bounds = true;
                    break; // Avoid accessing out-of-bound pixels
                }

                double cost = 0;
                double num_of_similar_pixels = 0;
                double num_of_non_similar_pixels = 0;

                // Compute SAD for the current disparity
                for (int wy = -half_block; wy <= half_block; wy++)
                {
                    const uchar *left_row_ptr = event_image_polarity_left.ptr<uchar>(y + wy);
                    const uchar *right_row_ptr = event_image_right_polarity_.ptr<uchar>(y + wy);

                    for (int wx = -half_block; wx <= half_block; wx++)
                    {
                        if (left_row_ptr[x + wx] != right_row_ptr[x + wx - d])
                        {
                            num_of_non_similar_pixels++;
                        }
                    }
                }

                cost = num_of_non_similar_pixels / total_pixel;
                if (cost < min_cost)
                {
                    min_cost = cost;
                    best_disparity = d;
                }
            }
            if (!out_of_bounds & best_disparity != 0)
            {
                double gt_disparity = left_gt_disparity.at<double>(y, x);
                file << y << "," << x << "," << best_disparity << "," << gt_disparity << "," << min_cost << "\n";
                disparity.at<double>(y, x) = (best_disparity * 255 / disparity_range_);
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Disp calculation took: " << duration.count() / 1000.0 << " milliseconds" << std::endl;

    disparity.convertTo(disparity, CV_8U); // Convert from 64F to 8U
    cv::applyColorMap(disparity, color_map_, cv::COLORMAP_JET); // convert to colour map
    color_map_.copyTo(cv_image_disparity_.image);
    sad_disparity_pub_.publish(cv_image_disparity_.toImageMsg());
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
        double start_offset = tmp.ts.toSec();
        auto algo_start = high_resolution_clock::now();
        if (publish_slice_)
        {
            publishOnce(start_offset + slice_start_time_, start_offset + slice_end_time_);
        }
        else
        {
            publishOnce(start_offset + total_time_elapsed.count(), start_offset + total_time_elapsed.count() + loop_rate_);
        }
        auto algo_end = high_resolution_clock::now();
        duration<double> algo_duration = algo_end - algo_start;
        std::cout << "Algorithm execution time: " << algo_duration.count()*1000 << " milliseconds" << std::endl;
    }
}
