
#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <numeric>      // for std::accumulate
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

// Enumeration for the descriptor type, which can be
// either histogram of orineted gradients (HOG) or
// binary.
// An overbiew about which is which can be found for instance here:
// https://www.uio.no/studier/emner/matnat/its/TEK5030/v19/lect/lecture_4_1_feature_descriptors.pdf
// HOG: SIFT, SURF
// Binary: BRIEF, ORB, BRISK, FREAK, AKAZE
enum class DescriptorType {
    HOG,
    BINARY
};

struct LidarPoint { // single lidar point in space
    double x,y,z,r; // x,y,z in [m], r is point reflectivity
};

struct BoundingBox { // bounding box around a classified object (contains both 2D and 3D data)
    
    int boxID; // unique identifier for this bounding box
    int trackID; // unique identifier for the track to which this bounding box belongs
    
    cv::Rect roi; // 2D region-of-interest in image coordinates
    int classID; // ID based on class file provided to YOLO framework
    double confidence; // classification trust

    std::vector<LidarPoint> lidarPoints; // Lidar 3D points which project into 2D image roi
    std::vector<cv::KeyPoint> keypoints; // keypoints enclosed by 2D roi
    std::vector<cv::DMatch> kptMatches; // keypoint matches enclosed by 2D roi
};

struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
    std::vector<LidarPoint> lidarPoints;

    std::vector<BoundingBox> boundingBoxes; // ROI around detected objects in 2D image coordinates
    std::map<int,int> bbMatches; // bounding box matches between previous and current frame
};


//
// Websites which have been used as a reference for implementing this ringbuffer:
// https://stackoverflow.com/questions/8054273/how-to-implement-an-stl-style-iterator-and-avoid-common-pitfalls
// https://stackoverflow.com/questions/7758580/writing-your-own-stl-container/7759622#7759622
// https://www.ibm.com/support/knowledgecenter/ssw_ibm_i_74/rzarg/cplr330.htm
//

// Ringbuffer class.
// T is the data type, and Capacity is the capacity of the ringbuffer.
template<typename T, int Capacity>
class RingBuffer {

    // The actual data.
    T *data_[Capacity];
    // The capacity of the ringbuffer -- stored as a variable for easy access.
    const int capacity_{Capacity};
    // Current number of entries in the ringbuffer.
    int size_{0};
    // Index of the first element in the buffer.
    int first_{0};

    public:
    // Constructor
    RingBuffer() {
        for (int ii=0; ii<capacity_; ii++) {
            data_[ii] = nullptr;
        }
    }
    // Destructor
    virtual ~RingBuffer() {
        // Free all remaining entries.
        for (int ii=0; ii<capacity_; ii++) {
            if (data_[ii] != nullptr) {
                delete data_[ii];
                data_[ii] = nullptr;
            }
        }
    }

    // Non-const iterator for this ringbuffer (not a complete implementation as required by the C++ standard).
    class iterator {
        // Type of the associated ringbuffer.
        using RingBufferType = RingBuffer<T, Capacity>;
        // Pointer to the associated ringbuffer object.
        RingBufferType *rb_;
        // Index in the ringbuffer where this iterator is pointing to.
        // This is not the actual index in the data_ member of the ringbuffer.
        int index_;

        public:
        // constructor
        iterator(RingBufferType *rb, int idx) : rb_(rb), index_(idx) {

        }

        // Prefix increment
        iterator& operator++() {
            // index_ = (index_ + 1) % rb_->capacity_;
            index_++;
            return *this;
        }

        // Postfix increment
        iterator operator++(int) {
            // index_ = (index_ + 1) % rb_->capacity_;
            index_++;
            return *this;
        }

        // Addition and subtraction
        iterator operator+(int offset) {
            iterator it(rb_, index_ + offset);
            return it;
        }
        iterator operator-(int offset) {
            iterator it(rb_, index_ - offset);
            return it;
        }

        // Check for equality
        bool operator==(const iterator& it) { return this->index_ == it.index_; }
        bool operator!=(const iterator& it) { return !(this->operator==(it)); }

        // Access to reference.
        T& operator*() {
            // return *(rb_->data_[index_]);
            return *(rb_->data_[(rb_->first_ + index_) % rb_->capacity_]);
        }

        // Access to pointer.
        T* operator->() {
            // return rb_->data_[index_];
            return rb_->data_[(rb_->first_ + index_) % rb_->capacity_];
        }
    };

    // Add an element to the end.
    // If the ring buffer is full, the oldest element is deleted.
    void push_back(const T& new_entry) {
        // First check if the buffer is full.
        if (size_ == capacity_) {
            pop_front();
        }

        // Index where the next element is to be put.
        int next = (first_ + size_) % capacity_;
        data_[next] = new T(new_entry);
        size_++;
    }

    // Remove the oldest element from the front.
    void pop_front() {
        // Make sure there is a valid entry at the front
        if (data_[first_] == nullptr) {
            return;
        }
        // Pop it away.
        delete data_[first_];
        data_[first_] = nullptr;
        first_ = (first_ + 1) % capacity_;
        size_--;
    }

    iterator begin() {
        // iterator it(this, first_);
        iterator it(this, 0);
        return it;
    }

    iterator end() {
        // iterator it(this, (first_ + size_) % capacity_);
        iterator it(this, size_);
        return it;
    }

    // Access operator.
    T& operator[](int idx) { return *data_[(first_ + idx) % capacity_]; }
    const T& operator[](int idx) const { return *data_[(first_ + idx) % capacity_]; }

    // Get the size
    int size() const { return size_; }
    

};

// In order to work on the performance evaluation tasks MP.7 - MP.9
// in a clean and efficient manner, a class is created which
// encapsulates the interesting quantities.
class PerformanceEvaluation {
    // Total number of images from which the keypoints or matches originate.
    int image_count_{0};
    // Detector type
    std::string detector_type_;
    // Descriptor type
    std::string descriptor_type_;
    // All keypoints ('all' meaning: collected over the images)
    std::vector<cv::KeyPoint> all_keypoints_;
    // Total number of matched keypoints.
    int total_number_matched_keypoints_{0};
    // Total processing time (in seconds) for the keypoint detection.
    double total_detector_time_{0.0};
    // Total processing time (in seconds) for the descriptor computation.
    double total_descriptor_time_{0.0};

    public:
        int imageCount() const { return image_count_; }
        void imageCount(int ic) { image_count_ = ic; }

        const std::string& detectorType() { return detector_type_; }
        void detectorType(const std::string& dt) { detector_type_ = dt; }

        const std::string& descriptorType() { return descriptor_type_; }
        void descriptorType(const std::string& dt) { descriptor_type_ = dt; }

        // Add keypoints to the vector of keypoints
        void addKeypoints(const std::vector<cv::KeyPoint> &keypoints) {
            for (const cv::KeyPoint &kpt : keypoints) {
                all_keypoints_.push_back(kpt);
            }
        }

        // Add a number of matched keypoints (do not add the actual matches, only their count).
        void addMatchedKeypoints(int count) {
            total_number_matched_keypoints_ += count;
        }

        // Add processing time for the keypoint detection.
        void addDetectorTime(double time_in_sec) {
            total_detector_time_ += time_in_sec;
        }

        // Add processing time for the descriptor computation.
        void addDescriptorTime(double time_in_sec) {
            total_descriptor_time_ += time_in_sec;
        }

        // Print the statistics to the console.
        // You can use this output for easy 'grep'-ing the data
        // for the writeup/readme.
        void printStatistics() const {

            std::cout << "\n===== Performance statistics summary =====\n";
            std::cout << "There have been " << image_count_ << " images.\n";
            std::cout << "Detector: " << detector_type_ << ", Descriptor: " << descriptor_type_ << "\n";

            std::cout << "\nMP.7 Performance Evaluation 1:\n";
            float avg_num_keypoints_per_image = (float) all_keypoints_.size() / ((float) image_count_);
            std::cout << "Average number of keypoints (on the vehicle) per image: " << avg_num_keypoints_per_image << "\n";
            // Compute the mean of the keypoint size.
            float avg_size = std::accumulate(all_keypoints_.begin(), all_keypoints_.end(), 0.0f, [](const float &val, const cv::KeyPoint &kpt) {
                    return (val + kpt.size);
                }) / ((float) all_keypoints_.size());
            // Compute the sample standard deviation of the keypoint size. See for instance here:
            // https://www.statisticshowto.com/probability-and-statistics/descriptive-statistics/sample-variance/
            float std_dev_size = std::accumulate(all_keypoints_.begin(), all_keypoints_.end(), 0.0f, [&avg_size](const float &val, const cv::KeyPoint &kpt) {
                    return (val + (kpt.size - avg_size) * (kpt.size - avg_size));
                }) / ((float) (all_keypoints_.size() - 1));
            std_dev_size = std::sqrt(std_dev_size);
            std::cout << "Average size of the keypoints (on the vehicle): " << avg_size << "\n";
            std::cout << "Size of the keypoints (on the vehicle) standard deviation: " << std_dev_size << "\n";

            std::cout << "\nMP.8 Performance Evaluation 2:\n";
            std::cout << "Total number of matched keypoints: " << total_number_matched_keypoints_ << "\n";

            std::cout << "\nMP.9 Performance Evaluation 3:\n";
            std::cout << "Average processing time for keypoint detection (per image): " << 1000.0 * total_detector_time_ / ((float) image_count_) << " ms\n";
            std::cout << "Average processing time for descriptor computation (per image): " << 1000.0 * total_descriptor_time_ / ((float) image_count_) << " ms\n";
            std::cout << "Combined average processing time (per image): " << 1000.0 * (total_detector_time_+total_descriptor_time_) / ((float) image_count_) << " ms\n";
        }

        // Write a JPG image of the detected keypoints within the image.
        // Path is the path of the folder where the file is stored.
        // The file itself is called <detector_type_>.jpg, for instance, HARRIS.jpg.
        void writeImage(const std::string& path, const cv::Mat &img, const std::vector<cv::KeyPoint> &keypoints, bool show_window = false) const {
            cv::Mat visImage = img.clone();
            cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            cv::imwrite(path + detector_type_ + ".jpg", visImage);
            if (show_window) {
                std::string windowName = detector_type_ + " Results";
                cv::namedWindow(windowName, 6);
                imshow(windowName, visImage);
                cv::waitKey(0);
            }
        }

};

#endif /* dataStructures_h */
