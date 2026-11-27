// SPDX-License-Identifier: GPL-3.0-or-later

#include "Tracking.h"
#include "Settings.h"
#include "System.h"
#include "Thirdparty/DBoW2/DUtils/Random.h"

#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ORB_SLAM3;

cv::Mat ReadImage(const cv::FileNode& path, int mode)
{
    cv::Mat image = cv::imread(static_cast<std::string>(path), mode);
    if (image.empty())
        throw std::runtime_error("Cannot read " + static_cast<std::string>(path));
    return image;
}

Sophus::SE3f WorldFromCamera(const cv::FileNode& node)
{
    std::vector<float> q, p;
    node["quaternion"] >> q;
    node["position"] >> p;
    if (q.size() != 4 || p.size() != 3)
        throw std::runtime_error("Invalid map keyframe pose");
    return Sophus::SE3f(Eigen::Quaternionf(q[3], q[0], q[1], q[2]).normalized(),
                        Eigen::Vector3f(p[0], p[1], p[2]));
}

class FixedMapTracker : public Tracking {
public:
    FixedMapTracker(ORBVocabulary& vocabulary, Atlas& atlas, KeyFrameDatabase& database, Settings& settings)
        : Tracking(nullptr, &vocabulary, nullptr, nullptr, &atlas, &database, "",
                   System::IMU_STEREO, &settings),
          atlas_(atlas), database_(database), camera_(settings.camera1())
    {}

    ~FixedMapTracker()
    {
        delete mpORBextractorLeft;
        delete mpORBextractorRight;
        delete mpImuPreintegratedFromLastKF;
        delete mpImuCalib;
    }

    void Import(const cv::FileNode& node, unsigned index, std::ostream& inventory)
    {
        Frame frame = MakeFrame(node, index);
        if (frame.N == 0)
            throw std::runtime_error("No reference features");
        const auto world_from_camera = WorldFromCamera(node);
        frame.SetPose(world_from_camera.inverse());
        cv::Mat depth = ReadImage(node["depth"], cv::IMREAD_UNCHANGED);
        if (depth.type() != CV_32FC1 || depth.size() != image_size_)
            throw std::runtime_error("Incorrect optical depth image");
        Map* map = atlas_.GetCurrentMap();
        auto keyframe = std::make_unique<KeyFrame>(frame, map, &database_);
        unsigned count = 0;
        for (int i = 0; i < frame.N; ++i)
        {
            const cv::Point2f uv = frame.mvKeysUn[i].pt;
            const float z = depth.at<float>(cvRound(uv.y), cvRound(uv.x));
            if (!std::isfinite(z) || z <= 0)
                throw std::runtime_error("Invalid optical depth");
            const Eigen::Vector3f ray((uv.x - frame.cx) / frame.fx, (uv.y - frame.cy) / frame.fy, 1);
            auto point = std::make_unique<MapPoint>(world_from_camera * (ray * z), keyframe.get(), map);
            point->AddObservation(keyframe.get(), i);
            keyframe->AddMapPoint(point.get(), i);
            point->ComputeDistinctiveDescriptors();
            point->UpdateNormalAndDepth();
            atlas_.AddMapPoint(point.get());
            landmarks_.push_back(std::move(point));
            ++count;
        }
        keyframe->ComputeBoW();
        atlas_.AddKeyFrame(keyframe.get());
        database_.add(keyframe.get());
        inventory << keyframe->mnId << ',' << static_cast<std::string>(node["name"]) << ','
                  << static_cast<std::string>(node["region"]) << ',' << count << '\n';
        keyframes_.push_back(std::move(keyframe));
    }

    bool Query(const cv::FileNode& node, const std::string& incident, unsigned index, std::ostream& results)
    {
        const std::string stamp = static_cast<std::string>(node["timestamp_ns"]);
        mCurrentFrame = MakeFrame(node, std::stoll(stamp) / 1e9);
        // Invoke the unmodified upstream LOST-state relocalization routine against the frozen atlas.
        const bool accepted = mCurrentFrame.N != 0 && Relocalization();
        results << incident << ',' << index << ',' << stamp << ',' << accepted;
        if (accepted)
        {
            const auto pose = mCurrentFrame.GetPose().inverse();
            const auto p = pose.translation();
            const auto q = pose.unit_quaternion();
            results << ',' << p.x() << ',' << p.y() << ',' << p.z() << ',' << q.x() << ','
                    << q.y() << ',' << q.z() << ',' << q.w();
        }
        else
            results << ",,,,,,,";
        results << '\n';
        return accepted;
    }

private:
    Frame MakeFrame(const cv::FileNode& node, double timestamp)
    {
        const cv::Mat left = ReadImage(node["left"], cv::IMREAD_GRAYSCALE);
        const cv::Mat right = ReadImage(node["right"], cv::IMREAD_GRAYSCALE);
        if (left.size() != right.size() || (image_size_.area() != 0 && left.size() != image_size_))
            throw std::runtime_error("Inconsistent stereo image size");
        image_size_ = left.size();
        return Frame(left, right, timestamp, mpORBextractorLeft, mpORBextractorRight,
                     mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, camera_, nullptr, *mpImuCalib);
    }

    Atlas& atlas_;
    KeyFrameDatabase& database_;
    GeometricCamera* camera_;
    cv::Size image_size_;
    std::vector<std::unique_ptr<KeyFrame>> keyframes_;
    std::vector<std::unique_ptr<MapPoint>> landmarks_;
};

} // namespace

int main(int argc, char** argv)
{
    if (argc != 4 && !(argc == 5 && std::string(argv[4]) == "--realtime"))
    {
        std::cerr << "Usage: orb_fixed_map VOCABULARY SETTINGS REQUEST.json [--realtime]\n";
        return 2;
    }
    const bool realtime = argc == 5;
    try
    {
        cv::setNumThreads(1);
        cv::FileStorage input(argv[3], cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
        if (!input.isOpened())
            throw std::runtime_error("Cannot open replay request");
        ORB_SLAM3::ORBVocabulary vocabulary;
        if (!vocabulary.loadFromTextFile(argv[1]))
            throw std::runtime_error("Cannot load ORB vocabulary");
        ORB_SLAM3::Settings settings(argv[2], ORB_SLAM3::System::IMU_STEREO);
        ORB_SLAM3::Atlas atlas(0);
        ORB_SLAM3::KeyFrameDatabase database(vocabulary);
        FixedMapTracker tracker(vocabulary, atlas, database, settings);
        std::ofstream inventory("atlas.csv"), results("native-frames.csv");
        if (!inventory || !results)
            throw std::runtime_error("Cannot write baseline traces");
        inventory << "keyframe_id,name,region,landmarks\n";
        results << "incident_id,frame,timestamp_ns,published,px,py,pz,qx,qy,qz,qw\n" << std::setprecision(9);
        std::ofstream timing;
        if (realtime)
        {
            timing.open("timing.csv");
            if (!timing)
                throw std::runtime_error("Cannot write timing.csv");
            timing << "incident_id,frame,timestamp_ns,scheduled_ns,started_ns,finished_ns,cpu_ns\n";
        }
        unsigned index = 0;
        for (const auto& keyframe : input["keyframes"])
            tracker.Import(keyframe, index++, inventory);
        if (index == 0)
            throw std::runtime_error("Empty fixed map");
        for (const auto& test : input["incidents"])
        {
            DUtils::Random::SeedRand(42);
            const std::string incident = static_cast<std::string>(test["id"]);
            const auto frames = test["frames"];
            if (frames.empty())
                throw std::runtime_error("Empty incident " + incident);
            using Clock = std::chrono::steady_clock;
            using Nanos = std::chrono::nanoseconds;
            const auto epoch = Clock::now();
            const auto resume = std::stoll(static_cast<std::string>(frames[0]["timestamp_ns"]));
            unsigned frame = 0;
            for (const auto& observation : frames)
            {
                const auto stamp = std::stoll(static_cast<std::string>(observation["timestamp_ns"]));
                const auto scheduled = stamp - resume;
                if (realtime)
                    std::this_thread::sleep_until(epoch + Nanos(scheduled));
                const auto started = Clock::now();
                const auto cpu_start = realtime ? std::clock() : 0;
                const bool accepted = tracker.Query(observation, incident, frame, results);
                const auto finished = Clock::now();
                if (realtime)
                {
                    const auto cpu_ns = std::llround((std::clock() - cpu_start) * (1e9 / CLOCKS_PER_SEC));
                    timing << incident << ',' << frame << ',' << stamp << ',' << scheduled << ','
                           << std::chrono::duration_cast<Nanos>(started - epoch).count() << ','
                           << std::chrono::duration_cast<Nanos>(finished - epoch).count() << ',' << cpu_ns << '\n';
                }
                ++frame;
                if (accepted)
                    break;
            }
        }
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
