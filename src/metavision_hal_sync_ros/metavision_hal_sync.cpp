/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A.                                                                                       *
 *                                                                                                                    *
 * Licensed under the Apache License, Version 2.0 (the "License");                                                    *
 * you may not use this file except in compliance with the License.                                                   *
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0                                 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed   *
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                      *
 * See the License for the specific language governing permissions and limitations under the License.                 *
 **********************************************************************************************************************/

#include <exception>
#include <iostream>
#include <boost/program_options.hpp>
#include <type_traits>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>

#include <metavision/hal/facilities/i_camera_synchronization.h>
#include <metavision/hal/facilities/i_trigger_in.h>
#include <metavision/hal/facilities/i_trigger_out.h>
#include <metavision/hal/facilities/i_ll_biases.h>
#include <metavision/hal/facilities/i_monitoring.h>
#include <metavision/hal/facilities/i_event_rate_activity_filter_module.h>
#include <metavision/hal/facilities/i_geometry.h>
#include <metavision/hal/facilities/i_events_stream_decoder.h>
#include <metavision/hal/facilities/i_event_decoder.h>
#include <metavision/hal/device/device.h>
#include <metavision/hal/device/device_discovery.h>
#include <metavision/hal/facilities/i_events_stream.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/events/event_ext_trigger.h>
#include <metavision/sdk/base/utils/error_utils.h>
#include <metavision/hal/facilities/i_event_trail_filter_module.h>

// ROS
#include <ros/ros.h>
#include <dvs_msgs/EventArray.h>
#include <dvs_msgs/Event.h>

class EventAnalyzer
{
public:
    EventAnalyzer(ros::NodeHandle &nh, ros::NodeHandle &p_nh, std::string topic_name)
    {
        nh_ = nh;
        nh_private_ = p_nh;
        topic_name_ = topic_name;
    };

    cv::Mat img, img_swap;
    dvs_msgs::EventArray event_arr;
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    std::string topic_name_;
    std::mutex m;
    ros::Publisher event_array_publisher;
    bool is_waiting_for_master = false;
    // int event_sequence;

    // Display colors
    cv::Vec3b color_bg = cv::Vec3b(52, 37, 30);
    cv::Vec3b color_on = cv::Vec3b(236, 223, 216);
    cv::Vec3b color_off = cv::Vec3b(201, 126, 64);

    void setup_display(const int width, const int height)
    {
        img = cv::Mat(height, width, CV_8UC3);
        img_swap = cv::Mat(height, width, CV_8UC3);
        img.setTo(color_bg);

        event_arr.height = height;
        event_arr.width = width;
        event_array_publisher = nh_.advertise<dvs_msgs::EventArray>(topic_name_, 1000000);
    }

    // Called from main Thread
    void get_display_frame(cv::Mat &display)
    {
        // Swap images
        {
            std::unique_lock<std::mutex> lock(m);
            event_arr.header.stamp = ros::Time::now();
            event_array_publisher.publish(event_arr);
            event_arr.events.clear();
            std::swap(img, img_swap);
            img.setTo(color_bg);
        }
        img_swap.copyTo(display);
    }

    // Called from decoding Thread
    void process_events(const Metavision::EventCD *begin, const Metavision::EventCD *end)
    {
        // acquire lock
        {
            std::unique_lock<std::mutex> lock(m);

            // check if master is started
            // the slave camera will get events with ts==0 until the master is started
            if (begin->t == 0 && (end - 1)->t == 0)
            {
                if (!is_waiting_for_master)
                { // print it only the first time
                    std::cout << "===========================" << std::endl;
                    std::cout << "Waiting for master to start" << std::endl;
                    std::cout << "===========================" << std::endl;
                }
                is_waiting_for_master = true;
            }
            else
            {
                is_waiting_for_master = false;
            }

            dvs_msgs::Event tmp;
            if (!is_waiting_for_master)
            { // no image display if the camera is not ready
                for (auto it = begin; it != end; ++it)
                {
                    img.at<cv::Vec3b>(it->y, it->x) = (it->p) ? color_on : color_off;
                    tmp.y = it->y;
                    tmp.x = it->x;
                    tmp.polarity = it->p;
                    ros::Time ros_ts;
                    double seconds = it->t * 1e-6; // Convert to seconds as double
                    ros_ts.fromSec(seconds);
                    tmp.ts = ros_ts;
                    event_arr.events.emplace_back(tmp);
                }
            }
        }
    }
};

namespace po = boost::program_options;
int main(int argc, char *argv[])
{
    // Setup ros
    ros::init(argc, argv, "metavision_ros_driver");

    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    std::string out_raw_file_path;
    std::string serial;
    std::string event_topic_name;
    bool mode_master = false;
    bool mode_slave = false;
    int fps_param = 25;
    int bias_diff_off = 0;
    int bias_diff_on = 0;
    int bias_hpf = 0;
    int event_trail_filter_threshold = 0;

    nh_private.param<std::string>("camera_serial", serial, "");
    nh_private.param<std::string>("out_raw_file_path", out_raw_file_path, "");
    nh_private.param<std::string>("event_topic_name", event_topic_name, "");

    nh_private.param<bool>("mode_master", mode_master, false);
    nh_private.param<bool>("mode_slave", mode_slave, true);

    nh_private.param<int>("fps", fps_param, 0);
    nh_private.param<int>("bias_diff_off", bias_diff_off, 0);
    nh_private.param<int>("bias_diff_on", bias_diff_on, 0);
    nh_private.param<int>("bias_hpf", bias_hpf, 0);
    nh_private.param<int>("event_trail_filter_threshold", event_trail_filter_threshold, 0);

    if (mode_master && mode_slave)
    {
        std::cerr << "A camera must be either Master or Slave, but it cannot be both." << std::endl;
        return 1;
    }

    std::cout << "Camera serial: " << serial << std::endl;
    std::cout << "Publishing to topic: " << event_topic_name << std::endl;

    if (mode_master)
    {
        std::cout << "Camera set as mode master " << std::endl;
    }
    else if (mode_slave)
    {
        std::cout << "Camera set as mode slave " << std::endl;
    }

    // Open the device
    std::cout << "Opening camera..." << std::endl;
    std::unique_ptr<Metavision::Device> device;
    try
    {
        device = Metavision::DeviceDiscovery::open(serial);
    }
    catch (Metavision::BaseException &e)
    {
        std::cerr << "Error exception: " << e.what() << std::endl;
    }

    if (!device)
    {
        std::cerr << "Camera opening failed." << std::endl;
        return 1;
    }
    std::cout << "Camera open." << std::endl;

    // do we want to record data?
    Metavision::I_EventsStream *i_eventsstream = device->get_facility<Metavision::I_EventsStream>();
    if (i_eventsstream)
    {
        if (out_raw_file_path != "")
        {
            i_eventsstream->log_raw_data(out_raw_file_path);
        }
    }
    else
    {
        std::cerr << "Could not initialize events stream." << std::endl;
        return 3;
    }

    // set master/slave mode
    Metavision::I_CameraSynchronization *i_camera_synchronization =
        device->get_facility<Metavision::I_CameraSynchronization>();

    if (mode_master)
    {
        if (i_camera_synchronization->set_mode_master())
        {
            std::cout << "Set mode Master successful. Remember to start the slave first." << std::endl;
        }
        else
        {
            std::cerr << "Could not set Master mode. Master/slave might not be supported by your camera" << std::endl;
            return 3;
        }
    }

    if (mode_slave)
    {
        if (i_camera_synchronization->set_mode_slave())
        {
            std::cout << "Set mode Slave successful." << std::endl;
        }
        else
        {
            std::cerr << "Could not set Slave mode. Master/slave might not be supported by your camera" << std::endl;
            return 3;
        }
    }

    // Set the biases
    Metavision::I_LL_Biases *i_biases = device->get_facility<Metavision::I_LL_Biases>();
    i_biases->set("bias_diff_off", bias_diff_off);
    i_biases->set("bias_diff_on", bias_diff_on);
    i_biases->set("bias_hpf", bias_hpf);

    // Set the trail filter
    Metavision::I_EventTrailFilterModule *event_trail_filter =
        device->get_facility<Metavision::I_EventTrailFilterModule>();

    event_trail_filter->enable(true);
    event_trail_filter->set_type(Metavision::I_EventTrailFilterModule::Type::TRAIL);
    // Set a specific filter time constant (example value: 5000 µs)
    event_trail_filter->set_threshold(event_trail_filter_threshold);

    i_eventsstream->start();
    // get camera geometry
    Metavision::I_Geometry *i_geometry = device->get_facility<Metavision::I_Geometry>();
    if (!i_geometry)
    {
        std::cerr << "Could not retrieve geometry." << std::endl;
        return 4;
    }

    // Instantiate framer object
    EventAnalyzer event_analyzer(nh, nh_private, event_topic_name);
    event_analyzer.setup_display(i_geometry->get_width(), i_geometry->get_height());

    // Get the handler of CD events
    Metavision::I_EventDecoder<Metavision::EventCD> *i_cddecoder =
        device->get_facility<Metavision::I_EventDecoder<Metavision::EventCD>>();

    if (i_cddecoder)
    {
        // Register a lambda function to be called on every CD events
        i_cddecoder->add_event_buffer_callback(
            [&event_analyzer](const Metavision::EventCD *begin, const Metavision::EventCD *end)
            {
                event_analyzer.process_events(begin, end);
            });
    }

    std::cout << "Camera started." << std::endl;

    // Get the decoder of events & start decoding thread
    Metavision::I_Decoder *i_decoder = device->get_facility<Metavision::I_EventsStreamDecoder>();
    bool stop_decoding = false;
    bool stop_application = false;
    std::thread decoding_loop([&]()
                              {
        while (!stop_decoding) {
            short ret = i_eventsstream->poll_buffer();

            // Here we polled data, so we can launch decoding
            auto raw_data = i_eventsstream->get_latest_raw_data();

            // This will trigger callbacks set on decoders: in our case EventAnalyzer.process_events
            if (raw_data) {
                i_decoder->decode(raw_data->data(), raw_data->data() + raw_data->size());
            }
        } });

    // Prepare OpenCV window
    // event-based cameras do not have a frame rate, but we need one for visualization
    const int wait_time = static_cast<int>(std::round(1.f / fps_param * 1000)); // how much we should wait between two frames
    
    cv::Mat display;                                                            // frame where events will be accumulated
    const std::string window_name = (mode_master) ? "Metavision HAL Sync - Master " : "Metavision HAL Sync - Slave ";
    cv::namedWindow(window_name, cv::WINDOW_GUI_EXPANDED);
    cv::resizeWindow(window_name, i_geometry->get_width(), i_geometry->get_height());

    // Now let's create the loop of main thread
    while (!stop_application)
    {
        event_analyzer.get_display_frame(display);

        if (!display.empty())
        {
            cv::imshow(window_name, display);
        }

        // If user presses `q` key, exit loop and stop application
        int key = cv::waitKey(wait_time);
        if ((key & 0xff) == 'q')
        {
            stop_application = true;
            stop_decoding = true;
            std::cout << "q pressed, exiting." << std::endl;
        }
    }

    // Wait end of decoding loop
    decoding_loop.join();

    return 0;
}
