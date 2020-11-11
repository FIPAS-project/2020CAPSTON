#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const int LOOP_NUM = 10;
const int GOOD_PTS_MAX = 50;
const float GOOD_PORTION = 0.15f;

struct SURFDetector 
{
	Ptr<Feature2D> surf;
	SURFDetector(double hessian = 800.0)
	{
		surf = SURF::create(hessian);
	}

	template<class Type>
	void operator()(const Type& in, const Type& mask, vector<cv::KeyPoint>& pts, Type& descriptors, bool useProvided = false)
	{
		//여기서 detect랑 descriptor를 동시에 진행
		surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
	}
};

template<class KPMatcher>
struct SURFMatcher
{
	KPMatcher matcher;

	template<class T>
	void match(const T& in1, const T& in2, vector<DMatch>& matches) 
	{
		matcher.match(in1, in2, matches);
	}
};


void getGoodMatches(vector<DMatch>& matches, vector< DMatch >& good_matches,
	const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2, vector<Point2f>& img_1, vector<Point2f>& img_2)
{
	sort(matches.begin(), matches.end());

	//15%의 좋은 특징점들을 저장
	const int ptsPairs = min(GOOD_PTS_MAX, (int)(matches.size() * GOOD_PORTION));
	for (int i = 0; i < ptsPairs; i++)
	{
		good_matches.push_back(matches[i]);
	}

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		//good_match로부터 good features 얻어옴
		img_1.push_back(keypoints1[good_matches[i].queryIdx].pt);
		img_2.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}
}

static Mat drawGoodMatches(const Mat& img1, const Mat& img2, 
	const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
	vector<DMatch>& good_matches, vector<Point2f>& img2_corners_, Mat& H)
{
	//img1에서 corner를 얻어옴
	vector<Point2f> img1_corners(4);
	img1_corners[0] = Point(0, 0);
	img1_corners[1] = Point(img1.cols, 0);
	img1_corners[2] = Point(img1.cols, img1.rows);
	img1_corners[3] = Point(0, img1.rows);
	
	vector<Point2f> img2_corners(4);
	
	Mat img_matches;
	drawMatches(img1, keypoints1, img2, keypoints2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//Draw lines between the corners
	img2_corners_ = img2_corners;
	perspectiveTransform(img2_corners, img1_corners, H);

	for (int i = 0; i < 4; i++)
	{
		line(img_matches,
			img2_corners[i] + Point2f((float)img1.cols, 0), img2_corners[(i + 1) % 4] + Point2f((float)img1.cols, 0),
			Scalar(0, 255, 0), 2, LINE_AA);
	}

	return img_matches;
}

///img1과 img2가 stitching된 결과 output
static Mat stitchingImage(const Mat& img1, const Mat& img2, const Mat& HMatrix)
{
	Mat Stitched;
	warpPerspective(img2, Stitched, HMatrix, Size(img2.cols * 2, img2.rows), INTER_CUBIC);
	Mat matROI(Stitched, Rect(0, 0, img1.cols, img1.rows));
	img1.copyTo(matROI);

	return Stitched;
}

///src 이미지의 모든 특징점을 print
static Mat drawAllKeypoints(const Mat& src, const vector<KeyPoint>& srcKeypoints)
{
	Mat printKeyPoint;
	drawKeypoints(src, srcKeypoints, printKeyPoint, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	return printKeyPoint;
}

int main() 
{
	VideoCapture video1("left.mp4");
	VideoCapture video2("right.mp4");

	if (!video1.isOpened())
	{
		cout << "video1 is not opened" << endl;
		return 0;
	}
	if (!video2.isOpened()) 
	{
		cout << "video2 is not opened" << endl;
		return 0;
	}

	int v_width = video1.get(cv::CAP_PROP_FRAME_WIDTH);
	int v_height = video1.get(cv::CAP_PROP_FRAME_HEIGHT);
	int frames_count1 = video1.get(cv::CAP_PROP_FRAME_COUNT);
	int frames_count2 = video2.get(cv::CAP_PROP_FRAME_COUNT);

	cout << "VIDEO WIDTH: " << v_width << endl;
	cout << "VIDEO HEIGTH: " << v_height << endl;
	cout << "VIDEO1/VIDEO2 FRAME COUNT: " << frames_count1 << "/" << frames_count2 << endl;

	//stitching된 결과를 저장하는 videoWriter
	VideoWriter stitched_writer;
	stitched_writer.open("resultVideo.avi", CV_FOURCC('D', 'I', 'V', 'X'), 24, Size(v_width * 2, v_height), true);
	
	if (!stitched_writer.isOpened())
	{
		cout << "writer is not opened" << endl;
		return 0;
	}

	//feature들이 매칭된 결과를 저장하는 videoWriter
	VideoWriter feature_mathing_writer;
	feature_mathing_writer.open("featureMatched.avi", CV_FOURCC('D', 'I', 'V', 'X'), 24, Size(v_width * 2, v_height), true);
	
	if (!feature_mathing_writer.isOpened())
	{
		cout << "feature_mathing_writer is not opened" << endl;
		return 0;
	}

	vector<KeyPoint> keypoints1, keypoints2;
	vector<DMatch> bf_matches;

	Mat descriptors1, descriptors2;

	SURFDetector surf;
	SURFMatcher<BFMatcher> bf_matcher; 

	int cnt = 0; //진행되고 있는 frame 개수 확인
	
	while (true)
	{
		UMat frame1, frame2;
		video1 >> frame1;
		video2 >> frame2;

		if (frame1.empty() || frame2.empty())
			break;

		UMat frame1_gray, frame2_gray;
		cvtColor(frame1, frame1_gray, CV_BGR2GRAY);
		cvtColor(frame2, frame2_gray, CV_BGR2GRAY);

		++cnt;
		cout << "=================" << cnt << "=================" << endl;

		//start stitching
		for (int i = 0; i < LOOP_NUM; i++)
		{
			//surf로 feature detect and descriptor 생성
			surf(frame1_gray.getMat(ACCESS_RW), Mat(), keypoints1, descriptors1);
			surf(frame2_gray.getMat(ACCESS_RW), Mat(), keypoints2, descriptors2);

			//feature들 matching
			bf_matcher.match(descriptors1, descriptors2, bf_matches);
		}

		cout << "FOUND: " << keypoints1.size() << " keypoints on left image" << endl;
		cout << "FOUND: " << keypoints2.size() << " keypoints on right image" << endl;

		//matching되는 feature들을 한눈에 보는 함수
		vector<DMatch> good_matches;
		vector<Point2f> img_1, img_2;
		vector<Point2f> corner;

		getGoodMatches(bf_matches, good_matches, keypoints1, keypoints2, img_1, img_2);
		
		Mat H = findHomography(img_2, img_1, CV_RANSAC);

		Mat frame1_kp = drawAllKeypoints(frame1_gray.getMat(ACCESS_RW), keypoints1);
		Mat frame2_kp = drawAllKeypoints(frame2_gray.getMat(ACCESS_RW), keypoints2);

		Mat img_matches_bf = drawGoodMatches(frame1.getMat(ACCESS_RW), frame2.getMat(ACCESS_RW), keypoints1, keypoints2, good_matches, corner, H);
		Mat stitched = stitchingImage(frame1.getMat(ACCESS_RW), frame2.getMat(ACCESS_RW), H);
		
		//frame별로 확인하고 싶으시면 주석푸시고 확인해보세요~!
		//imshow("frame1_kp", frame1_kp);
		//imshow("frame2_kp", frame2_kp);
		//imshow("matched", img_matches_bf);
		//imshow("stitched", stitched);
		//waitKey(0);

		feature_mathing_writer.write(img_matches_bf);
		stitched_writer.write(stitched);
	}

	return 0;
}