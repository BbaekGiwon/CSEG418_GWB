#include <iostream>
#include <opencv2\opencv.hpp>

cv::Mat problem_a_rotate_forward(cv::Mat img, double angle){
	cv::Mat output;
	//////////////////////////////////////////////////////////////////////////////
	//                         START OF YOUR CODE                               //
	//////////////////////////////////////////////////////////////////////////////
	
	// 반환 메트릭스 초기화
	output = cv::Mat::zeros(img.size(), img.type());
	// 가로를 x, 세로를 y로 설정
	// 가로, 세로의 절반 -> 이 점을 중심으로 회전
	int x_half = img.cols / 2;
	int y_half = img.rows / 2;

	// 삼각함수
	double radian = angle * CV_PI / 180.0; // cos, sin 모두 라디안으로 계산한다고 함
	double cos_rad = cos(radian);
	double sin_rad = sin(radian);

	/* 회전시 좌표 이동 방법
	* 1. 회전하는 중심을 기준으로 상대위치로 변경 -> x' = x - x_half
	* 2. 회전 -> x'_rotate = x'sin() - y'cos()
	* 3. 원래 위치로 복원 x_rotate = x'_rotate + x_half	
	*/

	// 사진의 모든 점
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			// 좌표 계산 후 정수로 변환해준다
			int x_moved = static_cast<int>((x - x_half) * cos_rad - (y - y_half) * sin_rad + x_half);
			int y_moved = static_cast<int>((x - x_half) * sin_rad + (y - y_half) * cos_rad + y_half);
			// 범위 안에 있는 것만 넣기
			if (x_moved >= 0 && x_moved < img.cols && y_moved >= 0 && y_moved < img.rows) {
				output.at<cv::Vec3b>(y_moved, x_moved) = img.at<cv::Vec3b>(y, x);
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////////
	//                          END OF YOUR CODE                                //
	//////////////////////////////////////////////////////////////////////////////
	cv::imshow("a_output", output); cv::waitKey(0);
	return output;
}

cv::Mat problem_b_rotate_backward(cv::Mat img, double angle){
	cv::Mat output;

	//////////////////////////////////////////////////////////////////////////////
	//                         START OF YOUR CODE                               //
	//////////////////////////////////////////////////////////////////////////////
	// 반환 메트릭스 초기화
	output = cv::Mat::zeros(img.size(), img.type());
	// 가로를 x, 세로를 y로 설정
	// 가로, 세로의 절반 -> 이 점을 중심으로 회전
	int x_half = img.cols / 2;
	int y_half = img.rows / 2;

	// 삼각함수
	double radian = -angle * CV_PI / 180.0; // backward 이므로 반대로 돌린다
	double cos_rad = cos(radian);
	double sin_rad = sin(radian);

	// 사진의 모든 점
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			// x, y로 이동 될 점들
			int x_will_move = static_cast<int>((x - x_half) * cos_rad - (y - y_half) * sin_rad + x_half);
			int y_will_move = static_cast<int>((x - x_half) * sin_rad + (y - y_half) * cos_rad + y_half);
			// 범위 안에 있는 것만
			if (x_will_move >= 0 && x_will_move < img.cols && y_will_move >= 0 && y_will_move < img.rows) {
				output.at<cv::Vec3b>(y, x) = img.at<cv::Vec3b>(y_will_move, x_will_move); // 얘는 backward 방식이라 forward와 반대로
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////////
	//                          END OF YOUR CODE                                //
	//////////////////////////////////////////////////////////////////////////////

	cv::imshow("b_output", output); cv::waitKey(0);

	return output;
}

cv::Mat problem_c_rotate_backward_interarea(cv::Mat img, double angle){
	cv::Mat output;

	//////////////////////////////////////////////////////////////////////////////
	//                         START OF YOUR CODE                               //
	//////////////////////////////////////////////////////////////////////////////
	// 반환 메트릭스 초기화
	output = cv::Mat::zeros(img.size(), img.type());
	// 가로를 x, 세로를 y로 설정
	// 가로, 세로의 절반 -> 이 점을 중심으로 회전
	int x_half = img.cols / 2;
	int y_half = img.rows / 2;

	// 삼각함수
	double radian = -angle * CV_PI / 180.0; // backward 이므로 반대로 돌린다.
	double cos_rad = cos(radian);
	double sin_rad = sin(radian);

	// 사진내의 모든 점
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			// x,y로 이동 될 점들
			double x_will_move = (x - x_half) * cos_rad - (y - y_half) * sin_rad + x_half;
			double y_will_move = (x - x_half) * sin_rad + (y - y_half) * cos_rad + y_half;

			// 정수로 캐스팅
			int x1 = (int)x_will_move;
			int y1 = (int)y_will_move;
			// x1, y1과 가장 가까운 neighbor
			int x2 = x1 + 1;
			int y2 = y1 + 1;
			// 정수로 캐스팅하면서 오차값
			double a = x_will_move - x1;
			double b = y_will_move - y1;

			// 사진 범위 내에서만
			if (x1 >= 0 && x2 < img.cols && y1 >= 0 && y2 < img.rows) {
				for (int c = 0; c < img.channels(); c++) {
					// bilinear interpolation
					double val = (1-a)*(1-b)*img.at<cv::Vec3b>(y1,x1)[c] + a*(1-b)*img.at<cv::Vec3b>(y1,x2)[c] + (1-a)*b*img.at<cv::Vec3b>(y2,x1)[c] + a*b*img.at<cv::Vec3b>(y2,x2)[c];
					output.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(val);
				}
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////////
	//                          END OF YOUR CODE                                //
	//////////////////////////////////////////////////////////////////////////////

	cv::imshow("c_output", output); cv::waitKey(0);

	return output;
}

cv::Mat Example_change_brightness(cv::Mat img, int num, int x, int y) {
	/*
	img : input image
	num : number for brightness (increase or decrease)
	x : x coordinate of image (for square part)
	y : y coordinate of image (for square part)

	*/
	cv::Mat output = img.clone();
	int size = 100;
	int height = (y + 100 > img.cols) ? img.cols : y + 100;
	int width = (x + 100 > img.rows) ? img.rows : x + 100;

	for (int i = x; i < width; i++)
	{
		for (int j = y; j < height; j++)
		{
			for (int c = 0; c < img.channels(); c++)
			{
				int t = img.at<cv::Vec3b>(i, j)[c] + num;
				output.at<cv::Vec3b>(i, j)[c] = t > 255 ? 255 : t < 0 ? 0 : t;
			}
		}

	}
	cv::imshow("output1", img);
	cv::imshow("output2", output);
	cv::waitKey(0);
	return output;
}

int main(void){

	double angle = -15.0f;

	cv::Mat input = cv::imread("lena.jpg");
	//Fill problem_a_rotate_forward and show output
	problem_a_rotate_forward(input, angle);
	//Fill problem_b_rotate_backward and show output
	problem_b_rotate_backward(input, angle);
	//Fill problem_c_rotate_backward_interarea and show output
	problem_c_rotate_backward_interarea(input, angle);
	//Example how to access pixel value, change params if you want
	Example_change_brightness(input, 100, 50, 125);
}