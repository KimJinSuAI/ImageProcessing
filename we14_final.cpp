#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace cv;
using namespace std;

//�⺻ �̹��� ����
Mat I;
Mat menu;
Mat mTmp, mTmpForBrightness;

//�޴� ��ġ ���� 
Rect rMenu0(500, 800, 160, 50);     //Save
Rect rMenu1(700, 800, 160, 50);     //Back
Rect rMenu2(20, 20, 160, 50);       //Add Noise
Rect rMenu3(180, 20, 160, 50);      //My Test
Rect rMenu4(340, 20, 160, 50);      //Brightness
Rect rMenu5(500, 20, 160, 50);      //elaborate
Rect rRpt(500, 90, 160, 50);        //repeat
Rect rFlip(500, 140, 160, 50);      //flip
Rect rMenu6(660, 20, 160, 50);      //skin detect
Rect rMenu7(20, 200, 160, 50);      //flatify
Rect rMenu8(180, 200, 160, 50);     //blurA
Rect rMenu9(340, 200, 160, 50);     //Greyscale
Rect rMenu10(500, 200, 160, 50);    //DetectEdge
Rect rMenu11(660, 200, 160, 50);    //RmvNoise
Rect rMenu12(20, 380, 160, 50);     //Reduce
Rect rMenu13(180, 380, 160, 50);    //Expand
Rect rMenu14(340, 380, 160, 50);    //FFT

Mat readImage();                                //�̹��� �ҷ�����
int bInsideRect(int x, int y, Rect rect);       //���� �簢�� �ȿ� �ִ��� üũ
void addNoise(int nNoise);                      //������ �߰�
void onMouse(int event, int mx, int my, int flag, void* param);
void onChange(int value, void* userdata);       //�������
int brightMode = 0;                             //����������
int elaborateMode = 0; int flip_flag = 1;       //�ռ�, ��������
void isSkinArea(Mat ycrcb[], Mat& Skin_img);    //�Ǻΰ���
void drawSelectedPoints();                      //�簢�� �׸���
void onMouse2(int event, int x, int y, int flag, void*); //���ȭ�� ���� �̺�Ʈ ó��
Mat getPerspectiveMatrix();                     //���ȭ �޼ҵ�
void blurring(Mat img);                         //��ó��
void detectEdge();                              //��������
void rmvNoise();                                //����������
void reduce();                                  //���
void expand();                                  //Ȯ��
void fft();                                  //ȸ��
Point2f p[4];
int nPoint = 0;

int main()
{
    //�⺻ �̹��� �ҷ�����
    I = readImage();
    mTmp = I.clone();

    //�޴� �̹��� �����
    menu = Mat::zeros(900, 900, CV_8UC1);
    menu += 255;
    rectangle(menu, rMenu0, Scalar(0), 1);
    rectangle(menu, rMenu1, Scalar(0), 1);
    rectangle(menu, rMenu2, Scalar(0), 1);
    rectangle(menu, rMenu3, Scalar(0), 1);
    rectangle(menu, rMenu4, Scalar(0), 1);
    rectangle(menu, rMenu5, Scalar(0), 1);
    rectangle(menu, rMenu6, Scalar(0), 1);
    rectangle(menu, rMenu7, Scalar(0), 1);
    rectangle(menu, rMenu8, Scalar(0), 1);
    rectangle(menu, rMenu9, Scalar(0), 1);
    rectangle(menu, rMenu10, Scalar(0), 1);
    rectangle(menu, rMenu11, Scalar(0), 1);
    rectangle(menu, rMenu12, Scalar(0), 1);
    rectangle(menu, rMenu13, Scalar(0), 1);
    rectangle(menu, rMenu14, Scalar(0), 1);
    putText(menu, "Save", Point(rMenu0.x + 40, rMenu0.y + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
    putText(menu, "Back", Point(rMenu1.x + 40, rMenu1.y + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
    putText(menu, "Add Noise", Point(rMenu2.x+15, rMenu2.y+30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
    putText(menu, "My Test", Point(rMenu3.x + 15, rMenu3.y + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
    putText(menu, "Brightness", Point(rMenu4.x + 15, rMenu4.y + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
    putText(menu, "Elaborate", Point(rMenu5.x + 15, rMenu5.y + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
    putText(menu, "Skin detect", Point(rMenu6.x + 10, rMenu6.y + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
    putText(menu, "Flatify", Point(rMenu7.x + 40, rMenu7.y + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
    putText(menu, "Blur", Point(rMenu8.x + 55, rMenu8.y + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
    putText(menu, "Greyscale", Point(rMenu9.x + 20, rMenu9.y + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
    putText(menu, "DetectEdge", Point(rMenu10.x + 10, rMenu10.y + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
    putText(menu, "RmvNoise", Point(rMenu11.x + 15, rMenu11.y + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
    putText(menu, "Reduce", Point(rMenu12.x + 30, rMenu12.y + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
    putText(menu, "Expand", Point(rMenu13.x + 30, rMenu13.y + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
    putText(menu, "FFT", Point(rMenu14.x + 30, rMenu14.y + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);

    //�޴� â�� ����� ���콺 �̺�Ʈ �߰�
    namedWindow("Menu");
    setMouseCallback("Menu", onMouse, &menu);

    imshow("src", I);
    imshow("Menu", menu);
    waitKey();

    return 0;
}

Mat readImage() {
    Mat I = imread("G.jpg",IMREAD_COLOR);
    if (I.empty()) {
        cout << "Error opening image" << endl;
        exit(EXIT_FAILURE);
    }
    return I;
}

//���� �簢�� �ȿ� �ִ��� üũ: �ȿ� ������ 1, �ۿ� ������ 0
int bInsideRect(int x, int y, Rect rect) {
    if (x >= rect.x && x < rect.x + rect.width && y >= rect.y && y < rect.y + rect.height)
        return 1;
    else
        return 0;
}

//������ �߰��ϴ� �Լ�
void addNoise(int nNoise) {
    int nGenPoints = 0;
    while (nGenPoints < nNoise) {
        int x = (int)(((float)rand() / RAND_MAX) * mTmp.cols);
        int y = (int)(((float)rand() / RAND_MAX) * mTmp.rows);
        if (x >= mTmp.cols || y >= mTmp.rows)
            continue;
        if ((float)rand() / RAND_MAX > 0.5) {
            mTmp.at<Vec3b>(y, x)[0] = 0;
            mTmp.at<Vec3b>(y, x)[1] = 0;
            mTmp.at<Vec3b>(y, x)[2] = 0;
        }
        else {
            mTmp.at<Vec3b>(y, x)[0] = 255;
            mTmp.at<Vec3b>(y, x)[1] = 255;
            mTmp.at<Vec3b>(y, x)[2] = 255;
        }
        nGenPoints++;
    }
}

//���콺 �̺�Ʈ �ڵ鷯
void onMouse(int event, int mx, int my, int flag, void* param) {
    switch (event) {  
    case EVENT_LBUTTONDOWN:
        if(brightMode==1){                                                  //��⸦ �������� ���� �ٸ�Ŭ���̺�Ʈó���� ���Ѵ�.
            if (bInsideRect(mx, my, rMenu4)) {
                rectangle(menu, rMenu4, Scalar(255), -1);
                rectangle(menu, rMenu4, Scalar(0), 1);
                putText(menu, "Brightness", Point(rMenu4.x + 25, rMenu4.y + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
                destroyWindow("Menu");
                namedWindow("Menu");
                setMouseCallback("Menu", onMouse, &menu);
                mTmp = mTmpForBrightness;
                imshow("Menu", menu);
                imshow("src", mTmp);
                brightMode = 0;
            }
        }
        else {
            if (bInsideRect(mx, my, rMenu0)) {                              //Save
                I = mTmp;
                destroyWindow("Menu");
                namedWindow("Menu");
                setMouseCallback("Menu", onMouse, &menu);
                imshow("Menu", menu);
            }
            if (bInsideRect(mx, my, rMenu1)) {                              //Back
                mTmp = I.clone();
                imshow("src", mTmp);
            }
            else if (bInsideRect(mx, my, rMenu2)) {                         //AddNoise
                addNoise(100);
                imshow("src", mTmp);
            }
            else if (bInsideRect(mx, my, rMenu3)) {                         //My Test
                putText(mTmp, "This is The final Exam!", Point(100, 100), FONT_HERSHEY_SIMPLEX, 1.3, Scalar(255, 255, 255), 4);
                putText(mTmp, "Add your functions!", Point(100, 200), FONT_HERSHEY_SIMPLEX, 1.3, Scalar(255, 255, 255), 4);
                imshow("src", mTmp);
            }
            else if (bInsideRect(mx, my, rMenu4)) {                         //brightness
                mTmpForBrightness = mTmp.clone();
                int val = 128;
                rectangle(menu, rMenu4, Scalar(255), -1);
                rectangle(menu, rMenu4, Scalar(0), 1);
                putText(menu, "Confirm", Point(rMenu4.x + 25, rMenu4.y + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
                createTrackbar("���", "Menu", &val, 255, onChange, &mTmp);
                imshow("Menu", menu);
                brightMode = 1;
            }
            else if (bInsideRect(mx, my, rMenu5)) {                         //elaborate
                if (elaborateMode == 0) {
                    rectangle(menu, rRpt, Scalar(0), 1);
                    rectangle(menu, rFlip, Scalar(0), 1);
                    putText(menu, "Repeat", Point(rRpt.x + 35, rRpt.y + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
                    putText(menu, "Flip", Point(rFlip.x + 60, rFlip.y + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
                    imshow("Menu", menu);
                    elaborateMode = 1;
                }
                else {
                    rectangle(menu, rRpt, Scalar(255), -1);
                    rectangle(menu, rFlip, Scalar(255), -1);
                    imshow("Menu", menu);
                    elaborateMode = 0;
                }
            }
            else if (bInsideRect(mx, my, rRpt) * elaborateMode) {           //repeat
                Mat tmp;
                repeat(mTmp, 1, 2, tmp);
                mTmp = tmp;
                imshow("src", mTmp);
            }
            else if (bInsideRect(mx, my, rFlip) * elaborateMode) {          //flip
                flip_flag++;
                if (flip_flag == 2) flip_flag = 0;
                flip(mTmp, mTmp, flip_flag);
                imshow("src", mTmp);
            }
            else if (bInsideRect(mx, my, rMenu6)) {         //skin detect

                Mat HSV_img, bgr[3], ycrcb[3], Skin_img, Skin_img_color;

                cvtColor(mTmp, HSV_img, CV_BGR2YCrCb);

                split(HSV_img, ycrcb);
                split(HSV_img, ycrcb);
                split(mTmp, bgr);
                isSkinArea(ycrcb, Skin_img);

                bgr[0] = bgr[0].mul(Skin_img / 255);
                bgr[1] = bgr[1].mul(Skin_img / 255);
                bgr[2] = bgr[2].mul(Skin_img / 255);
                merge(bgr, 3, Skin_img_color);
                imshow("src", Skin_img_color);
                mTmp = Skin_img_color;
            }
            else if (bInsideRect(mx, my, rMenu7)) {         //flatify
                setMouseCallback("src", onMouse2, 0);
            }
            else if (bInsideRect(mx, my, rMenu8)) {         //blur
                blurring(mTmp);
            }
            else if (bInsideRect(mx, my, rMenu9)) {         //greyscale
                cvtColor(mTmp, mTmp, COLOR_BGR2GRAY);
                cvtColor(mTmp, mTmp, COLOR_GRAY2BGR);
                imshow("src", mTmp);
            }
            else if (bInsideRect(mx, my, rMenu10)) {         //DetectEdge
                detectEdge();
            }
            else if (bInsideRect(mx, my, rMenu11)) {         //rmvNoise
                rmvNoise();
            }
            else if (bInsideRect(mx, my, rMenu12)) {         //reduce
                reduce();
            }
            else if (bInsideRect(mx, my, rMenu13)) {         //expand
                expand();
            }
            else if (bInsideRect(mx, my, rMenu14)) {         //FFT
                fft();
            }
        }
        break;
    }
}

void onMouse2(int event, int x, int y, int flag, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        p[nPoint].x = x; p[nPoint].y = y;
        nPoint = (nPoint + 1) > 4 ? 4 : (nPoint + 1);
        drawSelectedPoints();
        if (nPoint == 4) {
            Mat m = getPerspectiveMatrix();
            //warpAffine(src, dst, m, Size(src.cols, src.rows));
            Mat tmp;
            warpPerspective(mTmp, tmp, m, Size(mTmp.cols, mTmp.rows));
            mTmp = tmp;
            imshow("src", mTmp);
            nPoint = 0;
        }
    }
}

void drawSelectedPoints() {
    int size_srect = 30;
    Mat disply_image = mTmp.clone();
    for (int i = 0; i < nPoint; i++) {
        Rect rect(p[i].x - size_srect / 2, p[i].y - size_srect / 2, size_srect, size_srect);
        rectangle(disply_image, rect, Scalar(0, 0, 0), 8);
    }
    imshow("src", disply_image);
}

Mat getPerspectiveMatrix() {                        //���ȭ �޼ҵ�
    Mat m;
    if (nPoint == 4) {
        int L1 = sqrt(pow(p[0].x - p[1].x, 2) + pow(p[0].y - p[1].x, 2));
        int L2 = sqrt(pow(p[2].x - p[1].x, 2) + pow(p[2].y - p[1].x, 2));
        Point2f p_new[4];
        p_new[1] = p[1];
        p_new[0].x = p[1].x;
        p_new[0].y = p[1].y - L1;
        p_new[2].x = p[1].x + L2;
        p_new[2].y = p[1].y;
        p_new[3].x = p[1].x + L2;
        p_new[3].y = p[1].y - L1;
        m = getPerspectiveTransform(p, p_new);
    }

    return m;
}

void onChange(int value, void* userdata) {
    Mat* m = (Mat*)userdata;
    Mat m2 = *m - Scalar(128, 128, 128) + Scalar(value, value, value);
    mTmpForBrightness = m2;
    imshow("src", mTmpForBrightness);
}

void isSkinArea(Mat ycrcb[], Mat& Skin_img) {
    Skin_img = Mat(ycrcb[0].rows, ycrcb[0].cols, CV_8U);
    for (int i = 0; i < ycrcb[0].rows; i++) {
        for (int j = 0; j < ycrcb[0].cols; j++) {
            if (ycrcb[0].at<uchar>(i, j) > 80 &&
                ycrcb[1].at<uchar>(i, j) > 135 &&
                ycrcb[1].at<uchar>(i, j) < 180 &&
                ycrcb[2].at<uchar>(i, j) > 85 &&
                ycrcb[2].at<uchar>(i, j) < 135)
                Skin_img.at<uchar>(i, j) = 255;
            else
                Skin_img.at<uchar>(i, j) = 0;
        }
    }
}

void blurring(Mat img) {
    GaussianBlur(img, img, Size(9, 9), 1.0);
    imshow("src", img);
}

void detectEdge() {
    cvtColor(mTmp, mTmp, COLOR_BGR2GRAY);
    cvtColor(mTmp, mTmp, COLOR_GRAY2BGR);
    Mat dst_x, dst_y;
    Matx<int, 3, 3> mask1(				//�ι��� ����ũ
        -1, 0, 0,
        0, 1, 0,
        0, 0, 0);
    Matx<int, 3, 3> mask2(				//�ι��� ����ũ
        0, 0, -1,
        0, 1, 0,
        0, 0, 0);
    filter2D(mTmp, dst_x, CV_32F, mask1);
    filter2D(mTmp, dst_y, CV_32F, mask2);
    convertScaleAbs(dst_x, dst_x);
    convertScaleAbs(dst_y, dst_y);
    mTmp = max(dst_x, dst_y);
    imshow("src", mTmp);
}

void rmvNoise() {
    Matx<uchar, 3, 3> mask;
    mask << 0, 1, 0,
        1, 1, 1,
        0, 1, 0;
    morphologyEx(mTmp, mTmp, MORPH_CLOSE, mask);    //������ ���ֱ�
    morphologyEx(mTmp, mTmp, MORPH_OPEN, mask);     //�Ͼ��� ���ֱ�

    imshow("src", mTmp);
}

void reduce() {
    resize(mTmp, mTmp, Size((int)(mTmp.cols * 0.9), (int)(mTmp.rows * 0.9)), 0, 0, INTER_CUBIC);
    imshow("src", mTmp);
}

void expand() {
    resize(mTmp, mTmp, Size((int)(mTmp.cols * 1.1), (int)(mTmp.rows * 1.1)), 0, 0, INTER_CUBIC);
    imshow("src", mTmp);
}

void getZeroPaddedImage(Mat& src, Mat& dst) {
    int m = getOptimalDFTSize(src.rows);
    int n = getOptimalDFTSize(src.cols);
    copyMakeBorder(src, dst, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));
}

//�Ǽ��ο� ����� �����κ��� ũ��� ������ ����ϰ� ũ�⿡ log ���Ͽ� �����ִ� �Լ�
void getLogMag(Mat planes[], Mat& magI, Mat& angI) {
    cartToPolar(planes[0], planes[1], magI, angI, true);
    magI += Scalar::all(1);
    log(magI, magI);

    //����Ʈ���̹����� ũ��(�ʺ�, ����)�� Ȧ���� ��� ���� (�߶�)
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
}

//�� ��и��� ��ġ�� �������Ͽ� ��迭
void rearrange(Mat magI) {
    //rearrange the quadrants of Fourier image so that the origin is at the image center
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;
    Mat q0(magI, Rect(0, 0, cx, cy));	//Top-Left - Create a ROI oer quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));	//Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));	//Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy));	//Bottom-Right
    Mat tmp;							//swap quadrants (Top-Left with Bottom-Left)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);						//swap quadrant ( Top-Right with Bottom - Left)
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void onMouse3(int event, int x, int y, int flag, void* param) {
    if (event == EVENT_LBUTTONDOWN) {
        Mat* m = (Mat*)param;
        circle(*m, Point(x, y), 5, Scalar(0), -1);
        imshow("filtered magnitude", *m);
    }
}

void fft() {
    //zero �е�
    cvtColor(mTmp, mTmp, COLOR_BGR2GRAY);
    Mat padded;
    getZeroPaddedImage(mTmp, padded);

    //dft�� ���� 2ä�� ��Ʈ���� ����
    Mat planes[] = { Mat_<float>(padded),Mat::zeros(padded.size(),CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);

    //dft �����ϰ� ä�� �и� planes[0]: �Ǽ���, planes[1]: �����
    dft(complexI, complexI);
    split(complexI, planes);

    //�Ǽ��ο� ����� �����κ��� ũ�� ����ϰ� log ���Ͽ� ����ȭ, ������ �Բ� ���
    Mat magI, angI;
    getLogMag(planes, magI, angI);

    //����ȯ�� ���� �ִ� ġ�Ұ� ���
    double min_mag, max_mag, min_ang, max_ang;
    minMaxLoc(magI, &min_mag, &max_mag);
    minMaxLoc(angI, &min_ang, &max_ang);

    //����ȭ
    normalize(magI, magI, 0, 1, NORM_MINMAX);		//0~1 ���̷� ����ȭ
    normalize(angI, angI, 0, 1, NORM_MINMAX);		//0~1 ���̷� ����ȭ

    //�� ��и��� ��ġ�� ������ (1~3��и� ��ȯ, 2~4��и� ��ȯ)
    rearrange(magI);
    rearrange(angI);

    imshow("src", mTmp);

    //���͸�
    imshow("filtered magnitude", magI);
    setMouseCallback("filtered magnitude", onMouse3, &magI);
    waitKey(0);
    destroyWindow("filtered magnitude");

    ///////////////////////////////
    // IFFT�� ���� ó�� ����

    //�� ��и��� ��ġ�� ������ (1~3��и� ��ȯ, 2~4��и� ��ȯ)
    rearrange(magI);
    rearrange(angI);

    //de-normalize
    magI = magI * (max_mag - min_mag) + min_mag;
    angI = angI * (max_ang - min_ang) + min_ang;

    //�α��� ����ȯ�� ������ȯ ����
    exp(magI, magI);
    magI -= 1;

    //������ǥ��� ��ȯ�ϰ�, Inverse FFT ����
    polarToCart(magI, angI, planes[0], planes[1], true);
    merge(planes, 2, complexI);

    //Inverse FFT
    dft(complexI, complexI, DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT);
    split(complexI, planes);
    normalize(planes[0], planes[0], 0, 1, NORM_MINMAX);
    mTmp = planes[0];
    cvtColor(mTmp,mTmp, COLOR_GRAY2BGR);
    imshow("src", mTmp);
    waitKey(0);
}