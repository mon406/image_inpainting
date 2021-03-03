#pragma once
/* �g�p�f�B���N�g���w��y�ђ�` */
#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
string win_src = "src";		// ���͉摜�E�B���h�E
string win_dst = "dst";		// �o�͉摜�E�B���h�E
string win_dst2 = "dst2";
string win_dst3 = "dst3";

/* ��v�摜 Mat�֐� */
Mat img_src;	// ���͉摜�i�J���[�j
Mat img_src2;	// ���͉摜�i�O���[�X�P�[���j
Mat img_mask;	// �}�X�N�摜�i�O���[�X�P�[���j
Mat img_SRC;	// ��C�O�摜�i�J���[�j
Mat img_SRC2;	// ��C�O�摜�i�O���[�X�P�[���j
Mat img_dst;	// ��Ǐ��p�b�`�@��o�͉摜�i�J���[�j
Mat img_dst2;	// MRF�K����o�͉摜�i�J���[�j
Mat img_dst3;	// Gamma����MRF�K����o�͉摜�i�J���[�j

/* �����p�����[�^��` */
int width;				// �摜�̕�
int height;				// �摜�̍���
int MAX_DATA;			// ���s�N�Z����
int MAX_INTENSE = 255;	// �ő�F�l
/* �����p�����[�^�w�� */
#define L 3						// �s���~�b�h���x��(0~L)
int Repetition = 5;				// ������
float MAX_COUNT_DEF = 23000;	// �q�X�g�O�����i�����Œ�̐ݒ�ő吔�j

/* ��Ǐ��p�b�`�@�p�����[�^�w�� */
double Converge = 1.0e-10;	// ��������l
int Repeat = 10000;			// �ő唽����
double Ratio = 50;			// �F�l�ƃe�N�X�`���̔䗦 Rambda
double r_max;				// ANN�̍ő�T���͈́i�T�������͈͂̈�Ӂj
double Ro = 0.5;			// ANN�̒T���͈͂̏k���W��
/* �p�b�`�T�C�Y */	
int PATCHSIZEint = 11;	// �p�b�`�̈�ӂ̃s�N�Z����
int PATCHstart = -5;		// �p�b�`�̒��S���猩�����[�܂��͏�[
int PATCHend = 6;			// �p�b�`�̒��S���猩���E�[�܂��͉��[

/* MRF�p�����[�^�w�� */
double Rambda = 1.0e-7;	// �f�[�^�����
double Alpha = 0.0001;	// �����������i�������p�����[�^�j
double Sigma = 16;		// �m�C�Y���x���i�W���΍��j
double Mean = 0;		// �m�C�Y���x���i���ρj
/* Gamma����MRF�p�����[�^�ǉ��w�� */
double Alpha2 = 0.0001;
double Gamma = 0.000001;	// Gammma�l


/* ���o�͂̊֐� */
void Read();	// �t�@�C���ǂݍ��� (l.108)
void Out();		// �t�@�C�������o��
/* ��C���x�]���̊֐� */
void Evaluation(Mat&, Mat&, double&);			// ��C���x�]�� (l.190)
void MSE(Mat&, Mat&, double&, double&);			// MSE&PSNR�Z�o
double SSIM(Mat&, Mat&);						// SSIM�Z�o
double KLD(vector<Mat>&, vector<Mat>&);			// KLD�Z�o
void drawHist_Color(Mat&, Mat&, Mat&, double&);	// �q�X�g�O�����쐬 & KLD�v�Z (l.310)
/* ��C�s�N�Z���F���̊֐� */
void find_Patch_pixcel(Mat&, Mat&, vector<Point2i>&);	// dilated occlusion (if the patch included occluded pixels, return the posision)
int unoccluded_checker(Point2i&, Mat&);					// dilated data (if the patch included occluded pixels, return the number of it)

/* �摜�s���~�b�h�iColorImage, TectureFeature, Occlusion�j */
class ImagePyramid {
public:
	vector<Mat> U;				// ColorImage pyramid
	vector<Mat> O;				// Occlusion pyramid
	vector<Mat> T;				// TectureFeature pyramid
	vector<Point2i> occ_pix;	// the posision of Occlusion pixcel
	vector<int> occ_pix_index;	// the number of  Occlusion pixcel each pyramid

	ImagePyramid();				// make pyramid
	void show();				// show pyramid information
	void copy(ImagePyramid&);	// copy pyramid (U,O,T�̂݃R�s�[)
	void copyTo(ImagePyramid&);	// copy pyramid
};
ImagePyramid::ImagePyramid() {
	/* Occlusion pyramid�̍쐬
		�P���k���i������̍폜�j*/
	Mat img_temp_before, img_temp_after;
	img_mask.copyTo(img_temp_before);	// O[0]
	O.push_back(img_temp_before);
	/* �I�N���[�W�����s�N�Z���̃J�E���g */
	int OccNUM = 0;
	Point2i pixcel_index;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img_temp_before.data[y * width + x] != 0) {
				OccNUM++;
				pixcel_index = Point2i(x, y);
				occ_pix.push_back(pixcel_index);
			}
		}
	}
	occ_pix_index.push_back(OccNUM);

	uchar gray = (uchar)0;
	Point2i downPoint, upPoint;
	for (int pyrLevel = 1; pyrLevel <= L; pyrLevel++) {
		OccNUM = 0;
		img_temp_after.create(img_temp_before.rows / 2, img_temp_before.cols / 2, CV_8UC1);
		img_temp_after.setTo(gray);
		for (int y = 0; y < img_temp_after.rows; y++) {
			for (int x = 0; x < img_temp_after.cols; x++) {
				downPoint = Point2i(x, y);
				upPoint = Point2i(x * 2, y * 2);	// �����s��̍폜�i�P�n�܂�j
				gray = img_temp_before.data[upPoint.y * img_temp_before.cols + upPoint.x];
				if (gray != 0) {
					gray = (uchar)255;
					img_temp_after.data[downPoint.y * img_temp_after.cols + downPoint.x] = gray;
					OccNUM++;
					occ_pix.push_back(downPoint);
				}
			}
		}
		O.push_back(img_temp_after);	// O[pyrLevel]:1~L
		img_temp_after.copyTo(img_temp_before);
		occ_pix_index.push_back(OccNUM);
	}

	/* Color Image pyramid�̍쐬
		�P���k���i������̍폜�j */
	img_SRC.copyTo(img_temp_before);	// U[0]
	U.push_back(img_temp_before);
	Vec3b color;
	for (int pyrLevel = 1; pyrLevel <= L; pyrLevel++) {
		img_temp_after.create(img_temp_before.rows / 2, img_temp_before.cols / 2, CV_8UC3);
		for (int y = 0; y < img_temp_after.rows; y++) {
			for (int x = 0; x < img_temp_after.cols; x++) {
				downPoint = Point2i(x, y);
				upPoint = Point2i(x * 2, y * 2);	// �����s��̍폜�i�P�n�܂�j
				color = img_temp_before.at<Vec3b>(upPoint.y, upPoint.x);
				img_temp_after.at<Vec3b>(downPoint.y, downPoint.x) = color;
			}
		}
		U.push_back(img_temp_after);	// U[pyrLevel]:1~L
		img_temp_after.copyTo(img_temp_before);
	}

	/* TectureFeature pyramid�̍쐬
		���͉摜�i�O���[�j����e�N�X�`�����o */
	Mat TextureFeature;
	TextureFeature = Mat(Size(width, height), CV_8UC2);
	int I_x, I_y;	// �e�N�X�`���������O���[�X�P�[���̍���
	int gray1, gray2;
	int texture_point, texture_point2;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			texture_point = y * width + x;
			if (img_mask.data[texture_point] != 0 || img_mask.data[texture_point + 1] != 0 || img_mask.data[texture_point + width] != 0) {
				I_x = 0; I_y = 0;	// �}�X�N�摜����=>�e�N�X�`��������"0"�Ƃ���
			}
			else {
				if (x < width - 1) {	// x ����
					gray1 = (int)img_SRC2.data[texture_point];
					gray2 = (int)img_SRC2.data[texture_point + 1];
					I_x = abs(gray2 - gray1);
				}
				else { I_x = 0; }
				if (y < height - 1) {	// y ����
					gray1 = (int)img_SRC2.data[texture_point];
					gray2 = (int)img_SRC2.data[texture_point + width];
					I_y = abs(gray2 - gray1);
				}
				else { I_y = 0; }
			}

			texture_point2 = y * width * 2 + x * 2;	// T(0)
			TextureFeature.data[texture_point2] = (uchar)I_x;
			TextureFeature.data[texture_point2 + 1] = (uchar)I_y;
		}
	}
	/* �e�w�ł̃e�N�X�`�����v�Z�i�_�E���T���v�����O�j*/
	TextureFeature.copyTo(img_temp_before);
	T.push_back(img_temp_before);
	for (int pryLevel = 1; pryLevel <= L; pryLevel++) {
		img_temp_after = Mat(Size(img_temp_before.cols / 2, img_temp_before.rows / 2), CV_8UC2);
		/* T^l(x, y) = T(2^l*x, 2^l*y) , l = 1~L �Ń_�E���T���v�����O
		��	=> T^l(x, y) = T(x, y) �Ń_�E���T���v�����O���Čv�Z���� 2^l �|���� */
		for (int y = 0; y < img_temp_after.rows; y++) {
			for (int x = 0; x < img_temp_after.cols; x++) {
				downPoint = Point2i(x, y);
				upPoint = Point2i(x * 2, y * 2);	// �����s��̍폜�i�P�n�܂�j
				texture_point = downPoint.y * img_temp_after.cols * 2 + downPoint.x * 2;
				texture_point2 = upPoint.y * img_temp_before.cols * 2 + upPoint.x * 2;
				I_x = (int)img_temp_before.data[texture_point2];
				I_y = (int)img_temp_before.data[texture_point2 + 1];
				img_temp_after.data[texture_point] = (uchar)I_x;
				img_temp_after.data[texture_point + 1] = (uchar)I_y;
			}
		}
		T.push_back(img_temp_after);	// T[pyrLevel]:1~L
		img_temp_after.copyTo(img_temp_before);
	}
}
void ImagePyramid::show() {
	for (int pyrLevel = 0; pyrLevel <= L; pyrLevel++) {
		cout << " L = " << pyrLevel << " : ( " << U[pyrLevel].cols << " : " << U[pyrLevel].rows << " )" << endl;
		cout << "  Occlusion number : " << occ_pix_index[pyrLevel] << endl;
	}
}
void ImagePyramid::copy(ImagePyramid& original_Pyramid) {
	for (int pyr_LEVEL = 0; pyr_LEVEL <= L; pyr_LEVEL++) {
		U[pyr_LEVEL] = original_Pyramid.U[pyr_LEVEL].clone();
		O[pyr_LEVEL] = original_Pyramid.O[pyr_LEVEL].clone();
		T[pyr_LEVEL] = original_Pyramid.U[pyr_LEVEL].clone();
	}
}
void ImagePyramid::copyTo(ImagePyramid& original_Pyramid) {
	U.clear();
	O.clear();
	T.clear();
	occ_pix_index.clear();
	occ_pix.clear();

	int pyr_occ_ind_before = 0, pyr_occ_ind_sum;
	for (int pyr_LEVEL = 0; pyr_LEVEL <= L; pyr_LEVEL++) {
		U.push_back(original_Pyramid.U[pyr_LEVEL].clone());
		O.push_back(original_Pyramid.O[pyr_LEVEL].clone());
		T.push_back(original_Pyramid.T[pyr_LEVEL].clone());
		occ_pix_index.push_back(original_Pyramid.occ_pix_index[pyr_LEVEL]);
		for (int pyr_occ_ind = 0; pyr_occ_ind < (occ_pix_index[pyr_LEVEL] - pyr_occ_ind_before); pyr_occ_ind++) {
			pyr_occ_ind_sum = pyr_occ_ind_before + pyr_occ_ind;
			occ_pix.push_back(original_Pyramid.occ_pix[pyr_occ_ind_sum]);
		}
		pyr_occ_ind_before += occ_pix_index[pyr_LEVEL];
	}
}

/* �摜�s���~�b�h�̔C�ӂ̑w */
class LClass {
public:
	int LEVEL;	// �s���~�b�h���x��
	int XSIZE;	// �摜�̕�
	int YSIZE;	// �摜�̍���
	Mat imgU;
	Mat imgT;
	Mat imgO;
	vector<Point2i> occ_p;	// the posision of Occlusion pixcel
	int occ_index;			// the number of  Occlusion pixcel

	LClass(ImagePyramid&, int);		// read pyramid from ImagePyramid
	void upsample(ImagePyramid&);	// upsample image
};
LClass::LClass(ImagePyramid& pyramid, int pyramidLEVEL) {
	LEVEL = pyramidLEVEL;
	XSIZE = pyramid.U[LEVEL].cols;
	YSIZE = pyramid.U[LEVEL].rows;
	imgU = pyramid.U[LEVEL];
	imgT = pyramid.T[LEVEL];
	imgO = pyramid.O[LEVEL];

	occ_index = pyramid.occ_pix_index[LEVEL];
	int before_LEVEL = 0;
	if (LEVEL != 0) {
		for (int i = 0; i <= (LEVEL - 1); i++) {
			before_LEVEL = before_LEVEL + (int)pyramid.occ_pix_index[i];
		}
	}
	occ_p.clear();
	for (int i = 0; i < occ_index; i++) {
		occ_p.push_back(pyramid.occ_pix[before_LEVEL + i]);
	}
}
void LClass::upsample(ImagePyramid& pyramid) {
	LEVEL = LEVEL - 1;
	XSIZE = pyramid.U[LEVEL].cols;
	YSIZE = pyramid.U[LEVEL].rows;
	imgU = pyramid.U[LEVEL];
	imgT = pyramid.T[LEVEL];
	imgO = pyramid.O[LEVEL];

	occ_index = pyramid.occ_pix_index[LEVEL];
	int before_LEVEL = 0;
	if (LEVEL != 0) {
		for (int i = 0; i <= (LEVEL - 1); i++) {
			before_LEVEL = before_LEVEL + (int)pyramid.occ_pix_index[i];
		}
	}
	occ_p.clear();
	for (int i = 0; i < occ_index; i++) {
		occ_p.push_back(pyramid.occ_pix[before_LEVEL + i]);
	}
}

/* �V�t�g�}�b�v�i���΃x�N�g���j��� */
class ShiftMap {
private:
	int point;
	Point2i p;
public:
	int shift_level;		// �s���~�b�h���x��
	int xsize;				// �摜�̕�
	int ysize;				// �摜�̍���
	vector<Point2i> shift;	// �V�t�g�}�b�v

	ShiftMap(LClass&);				// �����V�t�g�}�b�v (ramdom)
	Point2i nn(Point2i&);			// �V�t�g�}�b�v�Ăяo��
	int nnX(Point2i&);
	int nnY(Point2i&);
	void put(Point2i&, Point2i&);	// �V�t�g�}�b�v�ύX
	void upsample(Mat&);			// upsampling Shift map
	void zero(Mat&);				// undilated
};
ShiftMap::ShiftMap(LClass& lclass) {
	Mat dilate_occlusion;			// dilate occlusion H~
	vector<Point2i> dilate_occ;
	find_Patch_pixcel(lclass.imgO, dilate_occlusion, dilate_occ);

	shift_level = lclass.LEVEL;
	xsize = lclass.XSIZE;
	ysize = lclass.YSIZE;
	for (int i = 0; i < ysize; i++) {
		for (int j = 0; j < xsize; j++) {
			p = Point2i(j, i);
			while (dilate_occlusion.data[p.y * lclass.XSIZE + p.x] != 0) {
				p.x = rand() % lclass.XSIZE;	// ��C�̈�̏����V�t�g�}�b�v (ramdom)
				p.y = rand() % lclass.YSIZE;
			}
			p = Point2i(p.x - j, p.y - i);
			shift.push_back(p);
		}
	}
	p = Point2i(0, 0);
	for (int k = xsize * ysize; k < width * height; k++) {
		shift.push_back(p);
	}
}
Point2i ShiftMap::nn(Point2i& POINT) {
	point = POINT.y * xsize + POINT.x;
	return shift[point];
}
int ShiftMap::nnX(Point2i& POINT) {
	point = POINT.y * xsize + POINT.x;
	return shift[point].x;
}
int ShiftMap::nnY(Point2i& POINT) {
	point = POINT.y * xsize + POINT.x;
	return shift[point].y;
}
void ShiftMap::put(Point2i& POINT, Point2i& NNpoint) {
	point = POINT.y * xsize + POINT.x;
	shift[point] = NNpoint;
}
void ShiftMap::upsample(Mat& NEWoccl) {
	vector<Point2i> temp;
	Point2i tempP;
	for (int num = 0; num < xsize * ysize; num++) {
		tempP = Point2i(shift[num].x * 2, shift[num].y * 2);
		temp.push_back(tempP);
	}
	xsize = 2 * xsize;
	ysize = 2 * ysize;
	shift_level--;
	shift.clear();

	/* �V�t�g�}�b�v�S�_�̂�����C�̈�͈��p�� */
	int before, recheck_count = 0;
	Point2i check, now;
	vector<Point2i> recheck;
	for (int i = 0, countY = 0; i < ysize; i++) {
		for (int j = 0, countX = 0; j < xsize; j++) {
			p = Point2i(0, 0);
			now = Point2i(j, i);
			if (NEWoccl.data[i * NEWoccl.cols + j] != 0) {
				before = countY * (xsize / 2) + countX;
				check = Point2i(j + temp[before].x, i + temp[before].y);
				p = temp[before];
			}
			shift.push_back(p);
			if ((j + 1) % 2 == 0) { countX++; }
		}
		if ((i + 1) % 2 == 0) { countY++; }
	}
	temp.clear();
	/* �V�t�g�}�b�v���p���g�� */
	int before_check;
	for (int i = 0; i < recheck_count; i++) {
		/*�m�F�p*/
		cout << i << " ; " << recheck_count << " <- " << recheck[recheck_count] << endl;
		check = recheck[recheck_count];
		before = check.y * NEWoccl.cols + check.x;
		before_check = before - 1;
		if (shift[before_check] == Point2i(0, 0) || check.x - 1 < 0) {
			before_check = before + 1;
			if (shift[before_check] == Point2i(0, 0) || check.x + 1 >= NEWoccl.cols) {
				before_check = before - NEWoccl.cols;
				if (shift[before_check] == Point2i(0, 0) || check.y - 1 < 0) {
					before_check = before + NEWoccl.cols;
					if (shift[before_check] == Point2i(0, 0) || check.y + 1 >= NEWoccl.rows) {
						cout << "Shift Upsamplig ERROR!! :" << check << endl;
					}
					else {
						shift[before] = shift[before_check]; cout << "Shift Upsamplig DOWN" << check << endl;
					}
				}
				else { shift[before] = shift[before_check]; }
			}
			else {
				shift[before] = shift[before_check]; cout << "Shift Upsamplig RIGHT" << check << endl;
			}
		}
		else { shift[before] = shift[before_check]; }
	}
	recheck.clear();
}
void ShiftMap::zero(Mat& undilate_occl) {
	Point2i zero = Point2i(0, 0);
	for (int i = 0; i < undilate_occl.rows; i++) {
		for (int j = 0; j < undilate_occl.cols; j++) {
			point = i * undilate_occl.cols +i;
			if (undilate_occl.data[point] == 0) {
				shift[point] = zero;
			}
		}
	}
}

/* �e�p�b�`��� & �R�X�g(����x)�v�Z */
class Patch {
private:
	int texture_double;
	int start_value = PATCHstart;
	int end_value = PATCHSIZEint - start_value;
public:
	int texture_level;			// �e�N�X�`������ (�~2^l) �v�Z�p�s���~�b�h���x��
	int patchSizeX = PATCHSIZEint;
	int patchSizeY = PATCHSIZEint;
	int PatchSize = patchSizeX * patchSizeY;
	int hPatchSize;				// ���C�̈�p�b�`��
	int nElsTotal;				// �̈�O�p�b�`��
	vector<Point2i> NN;			// �V�t�g�}�b�v
	vector<Point2i> ANN;
	vector<Vec3b> ColorNN;		// �F�l(BGR)
	vector<Vec3b> ColorANN;
	vector<float> IxNN;			// �e�N�X�`������ (�v�Z���� 2^l �|����)
	vector<float> IxANN;
	vector<float> IyNN;
	vector<float> IyANN;
	vector<int> OcclusionChecker; // ��C�̈�m�F�l
	// -> (-1:�ގ��悪����C�̈�,0:���C�̈�,1:��C�s�N�Z������C�̈�,2:�̈�O)

	Patch();													// ������
	Patch(int, int, Mat&, Mat&, Mat&, ShiftMap&, int&);			// �C�ӂ̓_�ƃV�t�g�}�b�v��Ƃ̃R�X�g�֐�
	Patch(Point2i, Point2i, Mat&, Mat&, Mat&, ShiftMap&, int&);	// �Q�_��r���̃R�X�g�֐�
	double costfunction(int);	// �R�X�g�֐�(����x)�̌v�Z
	// -> (0:�I�N���[�W�����Ȃ�d, 1:�I�N���[�W�����Ȃ�d^2, 2:�I�N���[�W�����܂�d, 3:�I�N���[�W�����܂�d^2)
};
Patch::Patch() {
	hPatchSize = 0;
	nElsTotal = 0;
	NN.clear();
	ANN.clear();
	ColorNN.clear();
	ColorANN.clear();
	IxNN.clear();
	IxANN.clear();
	IyNN.clear();
	IyANN.clear();
	OcclusionChecker.clear();
}
Patch::Patch(int X_p, int Y_p, Mat& U_p, Mat& T_p, Mat& O_p, ShiftMap& sm_p, int& texture_LEVEL) {
	texture_level = texture_LEVEL;
	texture_double = pow(2, texture_level);

	Point2i XY_p = Point2i(X_p, Y_p);
	Point2i XY_p_temp;
	for (int ppY = Y_p + start_value; ppY < Y_p + end_value; ppY++) {
		for (int ppX = X_p + start_value; ppX < X_p + end_value; ppX++) {
			NN.push_back(Point2i(ppX, ppY));	// ��C�s�N�Z��
			XY_p_temp = Point2i(ppX + sm_p.nnX(XY_p), ppY + sm_p.nnY(XY_p));
			ANN.push_back(XY_p_temp);			// ��C�s�N�Z���̃V�t�g�}�b�v��
		}
	}
	int texture_index;
	hPatchSize = 0;	// ��C�̈���s�N�Z����
	nElsTotal = 0;	// �̈�O�s�N�Z����
	vector<Vec3b> tempColorNN;
	vector<Vec3b> tempColorANN;
	vector<float> tempIxNN;
	vector<float> tempIxANN;
	vector<float> tempIyNN;
	vector<float> tempIyANN;
	for (int pp = 0; pp < PatchSize; pp++) {
		if (NN[pp].x < 0 || NN[pp].x >= U_p.cols || NN[pp].y < 0 || NN[pp].y >= U_p.rows) {
			OcclusionChecker.push_back(2);
			nElsTotal++;
		}
		else if (ANN[pp].x < 0 || ANN[pp].x >= U_p.cols || ANN[pp].y < 0 || ANN[pp].y >= U_p.rows) {
			OcclusionChecker.push_back(2);
			nElsTotal++;
		}
		else {
			if (O_p.at<uchar>(NN[pp].y, NN[pp].x) != 0) {
				OcclusionChecker.push_back(-1);
				hPatchSize++;

				tempColorNN.push_back(U_p.at<Vec3b>(NN[pp].y, NN[pp].x));
				tempColorANN.push_back(U_p.at<Vec3b>(ANN[pp].y, ANN[pp].x));

				texture_index = NN[pp].y * T_p.cols * 2 + NN[pp].x * 2;
				tempIxNN.push_back((int)(T_p.data[texture_index]) * texture_double);
				tempIyNN.push_back((int)(T_p.data[texture_index + 1]) * texture_double);
				texture_index = ANN[pp].y * T_p.cols * 2 + ANN[pp].x * 2;
				tempIxANN.push_back((int)(T_p.data[texture_index]) * texture_double);
				tempIyANN.push_back((int)(T_p.data[texture_index + 1]) * texture_double);
			}
			else { 
				if (O_p.at<uchar>(ANN[pp].y, ANN[pp].x) != 0) {
					OcclusionChecker.push_back(1);
				}
				else { OcclusionChecker.push_back(0); }

				ColorNN.push_back(U_p.at<Vec3b>(NN[pp].y, NN[pp].x));
				ColorANN.push_back(U_p.at<Vec3b>(ANN[pp].y, ANN[pp].x));

				texture_index = NN[pp].y * T_p.cols * 2 + NN[pp].x * 2;
				IxNN.push_back((int)(T_p.data[texture_index]) * texture_double);
				IyNN.push_back((int)(T_p.data[texture_index + 1]) * texture_double);
				texture_index = ANN[pp].y * T_p.cols * 2 + ANN[pp].x * 2;
				IxANN.push_back((int)(T_p.data[texture_index]) * texture_double);
				IyANN.push_back((int)(T_p.data[texture_index + 1]) * texture_double);
			}
		}
	}
	if (hPatchSize > 0) {
		for (int hPatch_count = 0; hPatch_count < hPatchSize; hPatch_count++) {
			ColorNN.push_back(tempColorNN[hPatch_count]);
			ColorANN.push_back(tempColorANN[hPatch_count]);
			IxNN.push_back(tempIxNN[hPatch_count]);
			IyNN.push_back(tempIyNN[hPatch_count]);
			IxANN.push_back(tempIxANN[hPatch_count]);
			IyANN.push_back(tempIyANN[hPatch_count]);
		}
	}
	tempColorNN.clear();
	tempColorANN.clear();
	tempIxNN.clear();
	tempIxANN.clear();
	tempIyNN.clear();
	tempIyANN.clear();
}
Patch::Patch(Point2i P_1, Point2i P_2, Mat& U_p, Mat& T_p, Mat& O_p, ShiftMap& sm_p, int& texture_LEVEL) {
	texture_level = texture_LEVEL;
	texture_double = pow(2, texture_level);

	Point2i XY_p = P_1;
	for (int ppY = start_value; ppY < end_value; ppY++) {
		for (int ppX = start_value; ppX < end_value; ppX++) {
			NN.push_back(Point2i(P_1.x + ppX, P_1.y + ppY));	// ��C�p�b�`
			ANN.push_back(Point2i(P_2.x + ppX, P_2.y + ppY));	// �ގ��p�b�`
		}
	}
	int texture_index;
	hPatchSize = 0;
	nElsTotal = 0;
	vector<Vec3b> tempColorNN;
	vector<Vec3b> tempColorANN;
	vector<float> tempIxNN;
	vector<float> tempIxANN;
	vector<float> tempIyNN;
	vector<float> tempIyANN;
	for (int pp = 0; pp < PatchSize; pp++) {
		if (NN[pp].x < 0 || NN[pp].x >= U_p.cols || NN[pp].y < 0 || NN[pp].y >= U_p.rows) {
			OcclusionChecker.push_back(2);
			nElsTotal++;
		}
		else if (ANN[pp].x < 0 || ANN[pp].x >= U_p.cols || ANN[pp].y < 0 || ANN[pp].y >= U_p.rows) {
			OcclusionChecker.push_back(2);
			nElsTotal++;
		}
		else {
			if (O_p.at<uchar>(NN[pp].y, NN[pp].x) != 0) {
				OcclusionChecker.push_back(-1);
				hPatchSize++;

				tempColorNN.push_back(U_p.at<Vec3b>(NN[pp].y, NN[pp].x));
				tempColorANN.push_back(U_p.at<Vec3b>(ANN[pp].y, ANN[pp].x));

				texture_index = NN[pp].y * T_p.cols * 2 + NN[pp].x * 2;
				tempIxNN.push_back((int)(T_p.data[texture_index]) * texture_double);
				tempIyNN.push_back((int)(T_p.data[texture_index + 1]) * texture_double);
				texture_index = ANN[pp].y * T_p.cols * 2 + ANN[pp].x * 2;
				tempIxANN.push_back((int)(T_p.data[texture_index]) * texture_double);
				tempIyANN.push_back((int)(T_p.data[texture_index + 1]) * texture_double);
			}
			else {
				if (O_p.at<uchar>(ANN[pp].y, ANN[pp].x) != 0) {
					OcclusionChecker.push_back(1);
				}
				else { OcclusionChecker.push_back(0); }

				ColorNN.push_back(U_p.at<Vec3b>(NN[pp].y, NN[pp].x));
				ColorANN.push_back(U_p.at<Vec3b>(ANN[pp].y, ANN[pp].x));

				texture_index = NN[pp].y * T_p.cols * 2 + NN[pp].x * 2;
				IxNN.push_back((int)(T_p.data[texture_index]) * texture_double);
				IyNN.push_back((int)(T_p.data[texture_index + 1]) * texture_double);
				texture_index = ANN[pp].y * T_p.cols * 2 + ANN[pp].x * 2;
				IxANN.push_back((int)(T_p.data[texture_index]) * texture_double);
				IyANN.push_back((int)(T_p.data[texture_index + 1]) * texture_double);
			}
		}
	}
	if (hPatchSize > 0) {
		for (int hPatch_count = 0; hPatch_count < hPatchSize; hPatch_count++) {
			ColorNN.push_back(tempColorNN[hPatch_count]);
			ColorANN.push_back(tempColorANN[hPatch_count]);
			IxNN.push_back(tempIxNN[hPatch_count]);
			IyNN.push_back(tempIyNN[hPatch_count]);
			IxANN.push_back(tempIxANN[hPatch_count]);
			IyANN.push_back(tempIyANN[hPatch_count]);
		}
	}
	tempColorNN.clear();
	tempColorANN.clear();
	tempIxNN.clear();
	tempIxANN.clear();
	tempIyNN.clear();
	tempIyANN.clear();
}
double Patch::costfunction(int info) {
	int Total = PatchSize - nElsTotal;
	if (info == 0 || info == 1) { Total = Total - hPatchSize; }

	int diff, sumU = 0;
	double diff_x, diff_y;
	double sumTx = 0, sumTy = 0;
	double answer;
	if (info == 0 || info == 1) {
		for (int pp = 0; pp < Total; pp++) {
			if (OcclusionChecker[pp] == 0) {
				// Cost sum of U
				for (int channel = 0; channel < 3; channel++) {
					diff = (int)ColorNN[pp][channel] - (int)ColorANN[pp][channel];
					diff = diff * diff;
					sumU = sumU + diff;
				}
				// Cost sum of T
				sumTx = sumTx + (double)abs(IxNN[pp] - IxANN[pp]);
				sumTy = sumTy + (double)abs(IyNN[pp] - IyANN[pp]);
			}
		}
	}
	else if (info == 2 || info == 3) {
		for (int pp = 0; pp < Total; pp++) {
			if (OcclusionChecker[pp] <= 0) {
				// Cost sum of U
				for (int channel = 0; channel < 3; channel++) {
					diff = (int)ColorNN[pp][channel] - (int)ColorANN[pp][channel];
					diff = diff * diff;
					sumU = sumU + diff;
				}
				// Cost sum of T
				sumTx = sumTx + (double)abs(IxNN[pp] - IxANN[pp]);
				sumTy = sumTy + (double)abs(IyNN[pp] - IyANN[pp]);
			}
		}
	}
	else {
		cout << "ERROR: fail calculate cost function." << endl;
	}
	sumTx = sqrt((double)(sumTx + sumTy));
	answer = (double)sumU + (double)(Ratio * sumTx);

	if (answer == 0 || Total == 0) { answer = -1; }		// ��r�s�N�Z�����Ȃ��ꍇ-1��Ԃ�
	//if (answer == 0 || Total <= 10) { answer = -1; }
	else {
		answer = (double)answer / (double)Total;
		if (info == 0 || info == 2) { answer = sqrt(answer); }
	}
	return answer;
}


/* dilated occlusion H~ (if the patch included occluded pixels, return the posision) 
	������:�g���O�I�N���[�W�����A������:�g����I�N���[�W�����A��O����:�g�����C�s�N�Z��*/
void find_Patch_pixcel(Mat& L_img, Mat& dilateOCC_img, vector<Point2i>& dilateOCC_pix) {
	int conclude_occ;
	Point2i now_p;
	uchar gray = (uchar)MAX_INTENSE;
	dilateOCC_pix.clear();

	L_img.copyTo(dilateOCC_img);	// �g���O�I�N���[�W����
	for (int y = 0; y < L_img.rows; y++) {
		for (int x = 0; x < L_img.cols; x++) {
			now_p = Point2i(x, y);
			conclude_occ = unoccluded_checker(now_p, L_img);
			if (conclude_occ != 0) {	// �g����I�N���[�W����
				dilateOCC_img.data[y * L_img.cols + x] = gray;
				dilateOCC_pix.push_back(Point2i(x, y));
			}
		}
	}
}
/* dilated data D~ (if the patch included occluded pixels, return the number of it) 
	������:�p�b�`�̒��S�s�N�Z���A������:�I�N���[�W����*/
int unoccluded_checker(Point2i& center, Mat& occluded) {
	int number_occ = 0;
	for (int y_p = center.y + PATCHstart; y_p < center.y + PATCHend; y_p++) {
		for (int x_p = center.x + PATCHstart; x_p < center.x + PATCHend; x_p++) {
			if (x_p >= 0 && x_p < occluded.cols && y_p >= 0 && y_p < occluded.rows) {
				if (occluded.data[y_p * occluded.cols + x_p] != 0) {
					number_occ++;
				}
			}
		}
	}
	return number_occ;	// �ߖTN_p�Ɋ܂܂���C�s�N�Z����
}


/* ��Ǐ��p�b�`�@�֘A�̊֐� */
void non_local_patch_matching(ImagePyramid&, Mat&);			// ��Ǐ��p�b�`�@ (l.410)
void InpaintingInitialisation(LClass&, ShiftMap&);			// Inpainting initialisation (l.526)
void PatchMatching_ANN(LClass&, ShiftMap&, Mat&, vector<Point2i>&, int&);		// Patch Match Algorithm (ANN) (l.598)
void PatchMatching_ANN_2D(LClass&, ShiftMap&, Mat&, vector<Point2i>&, int&);
void PatchMatching_All(LClass&, ShiftMap&, Mat&, vector<Point2i>&, int&);		// Patch Match Algorithm (�S�T��)
void Reconstruction(LClass&, ShiftMap&, int&);									// Reconstruction of U&T (l.789)
void Reconstruction_first(LClass&, ShiftMap&, vector<Point2i>&, Mat&, int&);	// Reconstruction of U&T at first

double SHIGMA(vector<double>&);								// sort & return 75th percentile (l.1093)
void firstReconstruction_SM(LClass&, ShiftMap&);			// Reconstruction of U at first
void firstReconstruction_COLOR(Mat, LClass&);				// Reconstruction at first
void annealingReconstruction(LClass&, ShiftMap&);			// Annealing
void update_pyramid(ImagePyramid&, LClass& newLClass);		// �s���~�b�h�X�V (l.1240)


/* MRF�֘A�̊֐� */
void OCC_MRF_GaussSeidel(Mat&, ImagePyramid&, double&);				// MRF (l.1247)
void OCC_GMRF_GaussSeidel(Mat&, ImagePyramid&, double&, double&);	// MRF with Gamma (l.1585)