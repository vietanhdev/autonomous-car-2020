#include "traffic_sign_detector_2.h"

using namespace cv;
using namespace std;
using namespace cv::ml;

TrafficSignDetector2::TrafficSignDetector2() {

    config = Config::getDefaultConfigInstance();
    debug_flag = config->get<bool>("debug_sign_detector");

    // Init debug image publishers
    if (debug_flag) {
        debug_img_publisher = createImagePublisher("trafficsign/debug_img", 1);
        debug_img_publisher_inrange = createImagePublisher("trafficsign/debug_img_inrange", 1);
    }

    color_file = ros::package::getPath(config->getROSPackage()) + config->get<std::string>("traffic_sign_detector_2_colorfile");
    svm_file = ros::package::getPath(config->getROSPackage()) + config->get<std::string>("traffic_sign_detector_2_svmfile");


    cout << "color_file: " << color_file << endl;
    cout << "svm_file: " << svm_file << endl;

    // Init HOG Descriptor config
    hog = cv::HOGDescriptor(
        cv::Size(32,32), //winSize
        cv::Size(8,8), //blocksize
        cv::Size(4,4), //blockStride,
        cv::Size(8,8), //cellSize, 
        9, //nbins,
        1, //derivAper,
        -1, //winSigma,
        0, //histogramNormType,
        0.2, //L2HysThresh,
        0,//gammal correction,
        64,//nlevels=64
        1
    );

    readColorFile();

    cout << "Done loading color file." << endl;

    svm = svm->load(svm_file);
    cout << "Done loading detectors." << endl;
    getSVMParams(svm);
    std::cout << "var_count = " << svm->getVarCount() << endl;

}


/*!
 * \brief Enlarge an ROI rectangle by a specific amount if possible 
 * \param frm The image the ROI will be set on
 * \param boundingBox The current boundingBox
 * \param padding The amount of padding around the boundingbox
 * \return The enlarged ROI as far as possible
 */
cv::Rect TrafficSignDetector2::enlargeROI(cv::Mat frm, cv::Rect boundingBox, int padding) {
    cv::Rect returnRect = cv::Rect(boundingBox.x - padding, boundingBox.y - padding, boundingBox.width + (padding * 2), boundingBox.height + (padding * 2));
    if (returnRect.x < 0) returnRect.x = 0;
    if (returnRect.y < 0) returnRect.y = 0;
    if (returnRect.x+returnRect.width >= frm.cols)
        returnRect.width = frm.cols-returnRect.x;
    if (returnRect.y+returnRect.height >= frm.rows)
        returnRect.height = frm.rows-returnRect.y;
    return returnRect;
}

void TrafficSignDetector2::mergeOverlappingBoxes(std::vector<cv::Rect> &inputBoxes, cv::Mat &image, std::vector<cv::Rect> &outputBoxes)
{
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1); // Mask of original image
    cv::Size scaleFactor(2,2); // To expand rectangles, i.e. increase sensitivity to nearby rectangles. Doesn't have to be (10,10)--can be anything
    for (int i = 0; i < inputBoxes.size(); i++)
    {
        cv::Rect box = inputBoxes.at(i) + scaleFactor;
        cv::rectangle(mask, box, cv::Scalar(255), CV_FILLED); // Draw filled bounding boxes on mask
    }

    std::vector<std::vector<cv::Point>> contours;
    // Find contours in mask
    // If bounding boxes overlap, they will be joined by this function call
    cv::findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    for (int j = 0; j < contours.size(); j++)
    {
        outputBoxes.push_back(cv::boundingRect(contours.at(j)));
    }
}


void TrafficSignDetector2::getSVMParams(cv::ml::SVM *svm)
{
	std::cout << "Kernel type     : " << svm->getKernelType() << std::endl;
	std::cout << "Type            : " << svm->getType() << std::endl;
	std::cout << "C               : " << svm->getC() << std::endl;
	std::cout << "Degree          : " << svm->getDegree() << std::endl;
	std::cout << "Nu              : " << svm->getNu() << std::endl;
	std::cout << "Gamma           : " << svm->getGamma() << std::endl;
}


// Receive list of detected objects and classify them
// Return: sign type id. Return -1 if the object is not a traffic sign
// hogDetectorsPath: path to HOG detectors
void TrafficSignDetector2::classify(cv::Mat & img,                 std::vector<Rect>& boundaries,
            cv::Ptr<cv::ml::SVM> & svm,
            std::vector<TrafficSign> & classification_results
    ) {


    cv::Mat cropImg;

    classification_results.clear();

    //cv::namedWindow( "Crop");
    for (std::vector<Rect>::iterator boundary = boundaries.begin(); boundary != boundaries.end(); boundary++) {
                
        cv::Mat croppedRef = img(*boundary);
        //cv::Mat croppedRef(img, boundary->r);


        if (croppedRef.empty()) {
            break;
        }

        // crop the area which containing traffic sign
        croppedRef.copyTo(cropImg);
        // Obtain a grayscale matrix
        cv::cvtColor(cropImg, cropImg, CV_BGR2GRAY);


        // resize
        cv::Mat resizedImg;
        cv::resize(cropImg, resizedImg, cv::Size(32, 32));


        std::vector<float> descriptors;
        hog.compute(resizedImg, descriptors);
        cv::Mat Hogfeat(1, descriptors.size(), CV_32FC1);
        for (size_t i = 0; i < descriptors.size(); i++)
        {
            Hogfeat.at<float>(0, i) = descriptors.at(i);
        }
        Hogfeat.reshape(1, 1); //flattened to a single row


        cv::Mat testResponse;
        float retConf = svm->predict(Hogfeat, testResponse, true);
                
        std::string label;
        int id = -1;
        for (int i = 0; i < testResponse.rows; i++) {

            id = static_cast<int>(testResponse.at<float>(i, 0));

            if (id == TrafficSign::SignType::NO_SIGN) break;
            else if (id == TrafficSign::SignType::TURN_LEFT) label = "RE TRAI";
            else if (id == TrafficSign::SignType::TURN_RIGHT) label = "RE PHAI";

            int baseline = 0;
            rectangle(img, *boundary, Scalar(0,0,255), 2);
            cv::Size text = cv::getTextSize(label, CV_FONT_HERSHEY_PLAIN, 1, 1, &baseline);
            cv::rectangle(img, boundary->tl() + cv::Point(0, baseline), boundary->tl() + cv::Point(text.width, -text.height), CV_RGB(0,255,0), CV_FILLED);
            cv::putText(img, label, boundary->tl(), CV_FONT_HERSHEY_PLAIN, 1, Scalar(0,0,0));
       
            if (id == TrafficSign::SignType::TURN_LEFT || id == TrafficSign::SignType::TURN_RIGHT) {
                classification_results.push_back(TrafficSign(id, *boundary));
            
            }
       
       
        }

    }

}


std::vector<cv::Rect> TrafficSignDetector2::detect(const cv::Mat & input) {

    cv::Mat img = input.clone();

    // Applying a Median filter
    // For a better result in segmentation
	cv::medianBlur(img, img, 3);

    // Convert input image to HSV
	cv::Mat hsv_image;
	cv::cvtColor(img, hsv_image, cv::COLOR_BGR2HSV);

    cv::Mat hue_range; // filter matrix for both major and minor color
    cv::Mat result_img; // result matrix for both major and minor color


    // Process the first color range first
    if (color_ranges.size() == 0) {
        std::cout << "Color range is empty. Break!" << std::endl;
        std::vector<cv::Rect> empty_set;
        return empty_set;
    }

    cv::inRange(hsv_image, color_ranges[0].begin, color_ranges[0].end, hue_range);
    cv::addWeighted(hue_range, 1, hue_range, 1, 0.0, result_img);

    result_img.copyTo(result_img);

    // Loop for all color ranges
    for (size_t i = 1; i < color_ranges.size(); i++) {
        cv::inRange(hsv_image, color_ranges[i].begin, color_ranges[i].end, hue_range);
        cv::addWeighted(result_img, 1, hue_range, 1, 0.0, result_img);
    }
    
    // Blur the result
    cv::GaussianBlur(result_img, result_img, cv::Size(1, 1), 2, 2);

    int morph_size = 3;
    cv::Mat close_kernel = getStructuringElement( cv::MORPH_RECT, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );


    //Open the area - Remove noise and expand the detected area
    cv::morphologyEx( result_img, result_img, cv::MORPH_CLOSE, close_kernel, cv::Point(-1,-1), 1 );   

    cv::Mat openKernel = getStructuringElement( cv::MORPH_RECT, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );
    cv::morphologyEx( result_img, result_img, cv::MORPH_OPEN, openKernel, cv::Point(-1,-1), 1 );   

    // Threshold
    cv::threshold(result_img, result_img, 1, 255, cv::THRESH_BINARY);

    // Publish debug image
    if (debug_flag) {
        publishImage(debug_img_publisher_inrange, result_img);
    }
    

    /// Find contours
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours( result_img, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );


    /// Find the boundary
    
    std::vector<cv::Rect> bound_rects( contours.size() );
    cv::Rect rect;
    

    for(size_t i = 0; i < contours.size(); i++ ) {
        rect = cv::boundingRect( cv::Mat(contours[i]) );
        // rect = enlargeROI(img, rect, 2);

        float area = rect.area();
        if (area > 10 && area < 4000) {
            bound_rects.push_back(rect);
        }
        
    }

    std::vector<cv::Rect> merged_bound_rects;
    mergeOverlappingBoxes(bound_rects, img, merged_bound_rects);

    
    return merged_bound_rects;

}


void TrafficSignDetector2::recognize(cv::Mat & input, std::vector<TrafficSign> & classification_results) {

    cv::Mat draw = input.clone();

    std::vector<cv::Rect> detection_results = detect(input);

    classify(draw, detection_results, svm, classification_results);

    for(size_t i = 0; i < detection_results.size(); i++ ) {
        rectangle(draw, detection_results[i], Scalar(0, 255, 0), 1);
    }

    if (debug_flag) {
        // cv::imshow("Traffic Sign Recognition", draw);
        // cv::waitKey(1);
        publishImage(debug_img_publisher, draw);
    }


}

void TrafficSignDetector2::readColorFile() {
    // Load color file
    std::string line;
    std::ifstream color_file_stream (color_file);
    if (!color_file_stream.is_open()) {
        std::cout << "Cannot open file: " << "blue.color" << std::endl;
        return;
    }

    color_ranges.clear();

    // Skip first 5 lines (comments)
    std::getline (color_file_stream, line);
    std::getline (color_file_stream, line);
    std::getline (color_file_stream, line);
    std::getline (color_file_stream, line);
    std::getline (color_file_stream, line);


    // HSV color range (from Scalar(h1, s1, v1) to Scalar(h2, s2, v2))
    double h1, s1, v1;
    double h2, s2, v2;
    
    while (color_file_stream) {
        std::getline (color_file_stream, line);
        if (line == "" || line == "END") break;
        std::stringstream line_stream(line);
        line_stream >> h1 >> s1 >> v1 >> h2 >> s2 >> v2;
        color_ranges.push_back(ColorRange(h1,s1,v1,h2,s2,v2));
    }
    
}