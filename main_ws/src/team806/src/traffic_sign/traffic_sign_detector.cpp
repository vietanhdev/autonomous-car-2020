#include "traffic_sign_detector.h"

// ==================================================
// ********** INITIALIZE **********
// ==================================================
TrafficSignDetector::TrafficSignDetector(){


    config = Config::getDefaultConfigInstance();
    config_trafficsign = Config::getNewConfigInstance("config_trafficsign.yaml");
    debug_flag = config->get<bool>("debug_sign_detector");

    // Init debug image publishers
    if (debug_flag) {
        debug_img_publisher = createImagePublisher("trafficsign/debug_img", 1);
        debug_img_publisher_inrange = createImagePublisher("trafficsign/debug_img_inrange", 1);
    }
    

    std::string model_file = ros::package::getPath(config->getROSPackage()) + config_trafficsign->get<std::string>("traffic_sign_detector_svmfile");
    model = cv::Algorithm::load<cv::ml::SVM>(model_file);

    low_HSV = config_trafficsign->getScalar3("low_HSV");
    high_HSV = config_trafficsign->getScalar3("high_HSV");

    size = config_trafficsign->get<int>("crop_size");
    eps_diff = config_trafficsign->get<float>("eps_diff");

    // For filtering contours with area
    min_area_contour = config_trafficsign->get<float>("min_area_contour");
    max_area_contour = config_trafficsign->get<float>("max_area_contour");

    // For filtering bouding rects, compare high with min_accepted_size, ratio = high/width
    min_accepted_size = config_trafficsign->get<float>("min_accepted_size");
    max_accepted_size = config_trafficsign->get<float>("max_accepted_size");
    min_accepted_ratio = config_trafficsign->get<float>("min_accepted_ratio");
    max_accepted_ratio = config_trafficsign->get<float>("max_accepted_ratio");

    // Number of labeled bouding rects in previous frames is stored in 'record'
    // If there are enough similarity labeled bouding rects in 'record', we can label the current rects as them
    min_prev_check = config_trafficsign->get<int>("min_prev_check");
    min_prob = config_trafficsign->get<float>("min_prob");

    // Init HOG Descriptor config
    hog = cv::HOGDescriptor(
        cv::Size(size,size),    //winSize
        cv::Size(8, 8),         //blocksize
        cv::Size(4, 4),         //blockStride,
        cv::Size(8, 8),         //cellSize,
        9,                      //nbins,
        1,                      //derivAper,
        -1,                     //winSigma,
        0,                      //histogramNormType,
        0.2,                    //L2HysThresh,
        1,                      //gamma correction,
        64,                     //nlevels=64
        1                       //_signedGradient = true
    );
};


// ==================================================
// ********** HELPER **********
// ==================================================

void TrafficSignDetector::createHOG(cv::HOGDescriptor &hog, std::vector<std::vector<float>> &HOG, std::vector<cv::Mat> &cells){
    for(size_t i=0; i<cells.size(); i++){
        std::vector<float> descriptors;
        hog.compute(cells[i], descriptors);
        HOG.push_back(descriptors);
    }
}

void TrafficSignDetector::cvtVector2Matrix(std::vector<std::vector<float>> &HOG, cv::Mat &mat){
    int descriptor_size = HOG[0].size();

    for(size_t i=0; i<HOG.size(); i++){
        for(size_t j=0; j<descriptor_size; j++){
            mat.at<float>(i, j) = HOG[i][j];
        }
    }
}


// ==================================================
// ********** SVM **********
// ==================================================

void TrafficSignDetector::svmPredict(cv::Ptr<cv::ml::SVM> svm, cv::Mat &response, cv::Mat &mat){
    svm->predict(mat, response);
}


// ==================================================
// ********** CLASSIFY **********
// ==================================================

int TrafficSignDetector::classifySVM(cv::HOGDescriptor &hog, cv::Ptr<cv::ml::SVM> &model, cv::Mat &img){

	std::vector<cv::Mat> cells;
    cvtColor(img, img, cv::COLOR_BGR2GRAY);
	cells.push_back(img);

	std::vector<std::vector<float>> HOG;
    createHOG(hog, HOG, cells);

    cv::Mat mat(HOG.size(), HOG[0].size(), CV_32FC1);

    cvtVector2Matrix(HOG, mat);

    cv::Mat response;
    svmPredict(model, response, mat);

    return (int)(response.at<float>(0, 0));
}


// ==================================================
// ********** TRAFFIC SIGN DETECTOR **********
// ==================================================

void TrafficSignDetector::BrightnessAndContrastAuto(cv::Mat src, cv::Mat &dst, bool clipHistPercent=false){
    int histSize = 256;
    float alpha, beta;
    double minGray = 0, maxGray = 0;

    cv::Mat gray;

    cvtColor(src, gray, CV_BGR2GRAY);

    if (clipHistPercent == true) {
        // keep full available range
        cv::minMaxLoc(gray, &minGray, &maxGray);
    } else {
        cv::Mat hist; //the grayscale histogram

        float range[] = { 0, 256 };
        const float* histRange = { range };
        bool uniform = true;
        bool accumulate = false;
        calcHist(&gray, 1, 0, cv::Mat (), hist, 1, &histSize, &histRange, uniform, accumulate);

        // calculate cumulative distribution from the histogram
        std::vector<float> accumulator(histSize);
        accumulator[0] = hist.at<float>(0);
        for (int i = 1; i < histSize; i++)
        {
            accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
        }

        // locate points that cuts at required value
        float max = accumulator.back();
        clipHistPercent *= (max / 100.0); //make percent as absolute
        clipHistPercent /= 2.0; // left and right wings

        // locate left cut
        minGray = 0;
        while (accumulator[minGray] < clipHistPercent)
            minGray++;

        // locate right cut
        maxGray = histSize - 1;
        while (accumulator[maxGray] >= (max - clipHistPercent))
            maxGray--;
    }

    float input_range = maxGray - minGray;
    float output_range = 255;
    alpha = output_range/input_range;

    beta = -minGray * alpha;

    src.convertTo(dst, -1, alpha, beta);
}

void TrafficSignDetector::inRangeHSV(cv::Mat &bin_img){
	cv::Mat img_HSV, img_threshold;

	// Convert color from BGR to HSV color space
	cvtColor(img, img_HSV, cv::COLOR_BGR2HSV);

	// Mark out all points in range, return binary image
	inRange(img_HSV, low_HSV, high_HSV, bin_img);
    
    if(debug_flag == true){
        // imshow("inRangeHSV", bin_img);
        // cv::waitKey(1);
        publishImage(debug_img_publisher_inrange, bin_img);
    }
}

void TrafficSignDetector::boundRectBinImg(cv::Mat bin_img, std::vector<cv::Rect> &bound_rects){

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	findContours(bin_img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	std::vector<std::vector<cv::Point>> contours_poly( contours.size() );
	cv::Rect rect;

	for(size_t i=0; i<contours.size(); i++){
		float contour_area = contourArea(contours[i]);
        if(contour_area > min_area_contour && contour_area < max_area_contour){
            approxPolyDP(contours[i], contours_poly[i], 3, true);
            rect = boundingRect(contours_poly[i]);

            bound_rects.push_back(rect);
        }
	}
}

void TrafficSignDetector::boundRectByColor(std::vector<cv::Rect> &bound_rects){
	// Apply threshold for BGR image
	cv::Mat bin_img;
	inRangeHSV(bin_img);

	// Get all bound rects
	boundRectBinImg(bin_img, bound_rects);
}

void TrafficSignDetector::mergeRects(std::vector<cv::Rect> &bound_rects){
	int findIntersection;

	do{
		findIntersection = false;

		for(auto it_1 = bound_rects.begin(); it_1 != bound_rects.end(); it_1++){
			for(auto it_2 =it_1+1; it_2 != bound_rects.end();){
				if( ((*it_1) & (*it_2)).area() > 0 ){
					findIntersection = true;
					*it_1 = ((*it_1) | (*it_2));
					bound_rects.erase(it_2);
				} else {
					it_2++;
				}
			}
		}
	} while(findIntersection == true);
}

void TrafficSignDetector::extendRect(cv::Rect &rect, int extend_dist){
	int tl_x = (rect.tl().x - extend_dist > 0) ? rect.tl().x - extend_dist : rect.tl().x;
	int tl_y = (rect.tl().y - extend_dist > 0) ? rect.tl().y - extend_dist : rect.tl().y;
	int br_x = (rect.br().x + extend_dist < width) ? rect.br().x + extend_dist : rect.br().x;
	int br_y = (rect.br().y + extend_dist < height) ? rect.br().y + extend_dist : rect.br().y;
	rect.x = tl_x;
	rect.y = tl_y;
	rect.width = br_x - tl_x;
	rect.height = br_y - tl_y;
}

bool TrafficSignDetector::checkSimilarityRect(cv::Rect A, cv::Rect B){
	float x = (float)(A|B).area();
	float y = (float)(A&B).area();
	float ratio = x / y;
	if( ratio > eps_diff ){
		return true;
	}
	return false;
}

void TrafficSignDetector::classifyCurrRect(){

	for(int i=0; i<record.curr_rects.size(); i++){

        // we use classifySVM
        cv::Mat roi_img = img(record.curr_rects[i].rect);
        resize(roi_img, roi_img, cv::Size(size, size));
        int id = classifySVM(hog, model, roi_img);
        record.curr_rects[i].id = id;
        
	}
}

void TrafficSignDetector::updatePrevRect(){
    auto it = std::begin(record.prev_rects);

    // delete expired rects
    while (it != std::end(record.prev_rects)) {
        long time_pass = Timer::calcTimePassed( (*it).observe_time );
        if ( time_pass > 3000 ) {
            it = record.prev_rects.erase(it);
        } else {
            ++it;
        }
    }

    // add new classified rects
    for(size_t i=0; i<record.curr_rects.size(); i++){
		record.prev_rects.push_back(record.curr_rects[i]);
	}

}

void TrafficSignDetector::recognize(const cv::Mat & input, std::vector<TrafficSign> &traffic_signs, cv::Mat & draw, bool draw_result){

    cv::Mat frame = input.clone();

	// Preprocessing, auto adjust brightness and contrast image
	BrightnessAndContrastAuto(frame, img, 1);
    width = img.cols;
    height = img.rows;

	// Clear old curr_rects to record new one
	record.curr_rects.clear();

	std::vector<cv::Rect> bound_rects;

	// Detect all bound_rects by color and contour area
    boundRectByColor(bound_rects);

    // Merge all rects have intersection greater than 0
    mergeRects(bound_rects);

    // Merge all neighbor rects by extending their size in 4 directions, then merge again by mergeRects function
    for(size_t i=0; i<bound_rects.size(); i++){
        extendRect(bound_rects[i], 1);
    }
    mergeRects(bound_rects);

    // Filter bound_rects by height and width and add to current rects
    for(size_t i=0; i<bound_rects.size(); i++){
    	float width = bound_rects[i].width;
    	float height = bound_rects[i].height;
    	float ratio = height/width;

    	if(height > min_accepted_size && height < max_accepted_size
            && ratio >= min_accepted_ratio && ratio < max_accepted_ratio){
    		record.curr_rects.push_back(TrafficSign(0, bound_rects[i]));
    	}
    }

	classifyCurrRect();

    // Return value to traffic_signs
    traffic_signs.clear();
    for(size_t i=0; i<record.curr_rects.size(); i++) {
        int total, same_id;
        TrafficSign curr = record.curr_rects[i];
        
        for(size_t j=0; j<record.prev_rects.size(); j++){
            TrafficSign prev = record.prev_rects[j];
            if( checkSimilarityRect(curr.rect, prev.rect) ) {
                total++;
                if(curr.id == prev.id){
                    same_id++;
                }
            }
        }

        // std::cout << "same result " << same_id<< " and total " << total << std::endl;
        int min_right = (int) (total * min_prob + 1);
        // std::cout << "min right " << min_right << std::endl;

        if(total >= min_prev_check && same_id >= min_right) {
            traffic_signs.push_back(curr);
        }
    }

    updatePrevRect();


    // show result
    if(draw_result){
        for(size_t i=0; i<traffic_signs.size(); i++){
            if(traffic_signs[i].id != 0){
                int x = traffic_signs[i].rect.tl().x;
                int y = traffic_signs[i].rect.tl().y;
                std::string text = traffic_signs[i].id == 1? "turn_left":"turn_right";
                putText(draw, text, cv::Point(x, y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,255), 2.0);
                // std::cout << text << " at [" << x << ", " << y << "] with area " << traffic_signs[i].rect.area() << std::endl;
            }
            rectangle(draw, traffic_signs[i].rect.tl(), traffic_signs[i].rect.br(), CV_RGB(255,0,255), 1, 8, 0);
        }
    }

    if (debug_flag) {
        for(size_t i=0; i<record.prev_rects.size(); i++){
            if(record.prev_rects[i].id == 0){
                rectangle(img, record.prev_rects[i].rect.tl(), record.prev_rects[i].rect.br(), CV_RGB(255,0,0), 1, 8, 0);
            } else if(record.prev_rects[i].id == 1){
                rectangle(img, record.prev_rects[i].rect.tl(), record.prev_rects[i].rect.br(), CV_RGB(0,255,0), 1, 8, 0);
            } else {
                rectangle(img, record.prev_rects[i].rect.tl(), record.prev_rects[i].rect.br(), CV_RGB(0,0,255), 1, 8, 0);
            }
        }
        publishImage(debug_img_publisher, img);
    }
}



void TrafficSignDetector::recognize(const cv::Mat & input, std::vector<TrafficSign> &traffic_signs) {

    cv::Mat dummy_img;
    recognize(input, traffic_signs, dummy_img, false);

}