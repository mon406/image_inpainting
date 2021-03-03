#pragma once
/* 使用ディレクトリ指定及び定義 */
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
string win_src = "src";		// 入力画像ウィンドウ
string win_dst = "dst";		// 出力画像ウィンドウ
string win_dst2 = "dst2";
string win_dst3 = "dst3";

/* 主要画像 Mat関数 */
Mat img_src;	// 入力画像（カラー）
Mat img_src2;	// 入力画像（グレースケール）
Mat img_mask;	// マスク画像（グレースケール）
Mat img_SRC;	// 補修前画像（カラー）
Mat img_SRC2;	// 補修前画像（グレースケール）
Mat img_dst;	// 非局所パッチ法後出力画像（カラー）
Mat img_dst2;	// MRF適応後出力画像（カラー）
Mat img_dst3;	// GammaありMRF適応後出力画像（カラー）

/* 実験パラメータ定義 */
int width;				// 画像の幅
int height;				// 画像の高さ
int MAX_DATA;			// 総ピクセル数
int MAX_INTENSE = 255;	// 最大色値
/* 実験パラメータ指定 */
#define L 3						// ピラミッドレベル(0~L)
int Repetition = 5;				// 実験回数
float MAX_COUNT_DEF = 23000;	// ヒストグラム（高さ固定の設定最大数）

/* 非局所パッチ法パラメータ指定 */
double Converge = 1.0e-10;	// 収束判定値
int Repeat = 10000;			// 最大反復回数
double Ratio = 50;			// 色値とテクスチャの比率 Rambda
double r_max;				// ANNの最大探索範囲（探索正方範囲の一辺）
double Ro = 0.5;			// ANNの探索範囲の縮小係数
/* パッチサイズ */	
int PATCHSIZEint = 11;	// パッチの一辺のピクセル数
int PATCHstart = -5;		// パッチの中心から見た左端または上端
int PATCHend = 6;			// パッチの中心から見た右端または下端

/* MRFパラメータ指定 */
double Rambda = 1.0e-7;	// データ項より
double Alpha = 0.0001;	// 平滑化項より（平滑化パラメータ）
double Sigma = 16;		// ノイズレベル（標準偏差）
double Mean = 0;		// ノイズレベル（平均）
/* GammaありMRFパラメータ追加指定 */
double Alpha2 = 0.0001;
double Gamma = 0.000001;	// Gammma値


/* 入出力の関数 */
void Read();	// ファイル読み込み (l.108)
void Out();		// ファイル書き出し
/* 補修精度評価の関数 */
void Evaluation(Mat&, Mat&, double&);			// 補修精度評価 (l.190)
void MSE(Mat&, Mat&, double&, double&);			// MSE&PSNR算出
double SSIM(Mat&, Mat&);						// SSIM算出
double KLD(vector<Mat>&, vector<Mat>&);			// KLD算出
void drawHist_Color(Mat&, Mat&, Mat&, double&);	// ヒストグラム作成 & KLD計算 (l.310)
/* 補修ピクセル認識の関数 */
void find_Patch_pixcel(Mat&, Mat&, vector<Point2i>&);	// dilated occlusion (if the patch included occluded pixels, return the posision)
int unoccluded_checker(Point2i&, Mat&);					// dilated data (if the patch included occluded pixels, return the number of it)

/* 画像ピラミッド（ColorImage, TectureFeature, Occlusion） */
class ImagePyramid {
public:
	vector<Mat> U;				// ColorImage pyramid
	vector<Mat> O;				// Occlusion pyramid
	vector<Mat> T;				// TectureFeature pyramid
	vector<Point2i> occ_pix;	// the posision of Occlusion pixcel
	vector<int> occ_pix_index;	// the number of  Occlusion pixcel each pyramid

	ImagePyramid();				// make pyramid
	void show();				// show pyramid information
	void copy(ImagePyramid&);	// copy pyramid (U,O,Tのみコピー)
	void copyTo(ImagePyramid&);	// copy pyramid
};
ImagePyramid::ImagePyramid() {
	/* Occlusion pyramidの作成
		単純縮小（偶数列の削除）*/
	Mat img_temp_before, img_temp_after;
	img_mask.copyTo(img_temp_before);	// O[0]
	O.push_back(img_temp_before);
	/* オクルージョンピクセルのカウント */
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
				upPoint = Point2i(x * 2, y * 2);	// 偶数行列の削除（１始まり）
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

	/* Color Image pyramidの作成
		単純縮小（偶数列の削除） */
	img_SRC.copyTo(img_temp_before);	// U[0]
	U.push_back(img_temp_before);
	Vec3b color;
	for (int pyrLevel = 1; pyrLevel <= L; pyrLevel++) {
		img_temp_after.create(img_temp_before.rows / 2, img_temp_before.cols / 2, CV_8UC3);
		for (int y = 0; y < img_temp_after.rows; y++) {
			for (int x = 0; x < img_temp_after.cols; x++) {
				downPoint = Point2i(x, y);
				upPoint = Point2i(x * 2, y * 2);	// 偶数行列の削除（１始まり）
				color = img_temp_before.at<Vec3b>(upPoint.y, upPoint.x);
				img_temp_after.at<Vec3b>(downPoint.y, downPoint.x) = color;
			}
		}
		U.push_back(img_temp_after);	// U[pyrLevel]:1~L
		img_temp_after.copyTo(img_temp_before);
	}

	/* TectureFeature pyramidの作成
		入力画像（グレー）からテクスチャ抽出 */
	Mat TextureFeature;
	TextureFeature = Mat(Size(width, height), CV_8UC2);
	int I_x, I_y;	// テクスチャ特徴＝グレースケールの差分
	int gray1, gray2;
	int texture_point, texture_point2;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			texture_point = y * width + x;
			if (img_mask.data[texture_point] != 0 || img_mask.data[texture_point + 1] != 0 || img_mask.data[texture_point + width] != 0) {
				I_x = 0; I_y = 0;	// マスク画像が白=>テクスチャ特徴を"0"とする
			}
			else {
				if (x < width - 1) {	// x 次元
					gray1 = (int)img_SRC2.data[texture_point];
					gray2 = (int)img_SRC2.data[texture_point + 1];
					I_x = abs(gray2 - gray1);
				}
				else { I_x = 0; }
				if (y < height - 1) {	// y 次元
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
	/* 各層でのテクスチャを計算（ダウンサンプリング）*/
	TextureFeature.copyTo(img_temp_before);
	T.push_back(img_temp_before);
	for (int pryLevel = 1; pryLevel <= L; pryLevel++) {
		img_temp_after = Mat(Size(img_temp_before.cols / 2, img_temp_before.rows / 2), CV_8UC2);
		/* T^l(x, y) = T(2^l*x, 2^l*y) , l = 1~L でダウンサンプリング
		※	=> T^l(x, y) = T(x, y) でダウンサンプリングして計算時に 2^l 掛ける */
		for (int y = 0; y < img_temp_after.rows; y++) {
			for (int x = 0; x < img_temp_after.cols; x++) {
				downPoint = Point2i(x, y);
				upPoint = Point2i(x * 2, y * 2);	// 偶数行列の削除（１始まり）
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

/* 画像ピラミッドの任意の層 */
class LClass {
public:
	int LEVEL;	// ピラミッドレベル
	int XSIZE;	// 画像の幅
	int YSIZE;	// 画像の高さ
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

/* シフトマップ（相対ベクトル）情報 */
class ShiftMap {
private:
	int point;
	Point2i p;
public:
	int shift_level;		// ピラミッドレベル
	int xsize;				// 画像の幅
	int ysize;				// 画像の高さ
	vector<Point2i> shift;	// シフトマップ

	ShiftMap(LClass&);				// 初期シフトマップ (ramdom)
	Point2i nn(Point2i&);			// シフトマップ呼び出し
	int nnX(Point2i&);
	int nnY(Point2i&);
	void put(Point2i&, Point2i&);	// シフトマップ変更
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
				p.x = rand() % lclass.XSIZE;	// 補修領域の初期シフトマップ (ramdom)
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

	/* シフトマップ４点のうち補修領域は引継ぎ */
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
	/* シフトマップ引継ぎ拡張 */
	int before_check;
	for (int i = 0; i < recheck_count; i++) {
		/*確認用*/
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

/* 各パッチ情報 & コスト(相違度)計算 */
class Patch {
private:
	int texture_double;
	int start_value = PATCHstart;
	int end_value = PATCHSIZEint - start_value;
public:
	int texture_level;			// テクスチャ特徴 (×2^l) 計算用ピラミッドレベル
	int patchSizeX = PATCHSIZEint;
	int patchSizeY = PATCHSIZEint;
	int PatchSize = patchSizeX * patchSizeY;
	int hPatchSize;				// 非補修領域パッチ数
	int nElsTotal;				// 領域外パッチ数
	vector<Point2i> NN;			// シフトマップ
	vector<Point2i> ANN;
	vector<Vec3b> ColorNN;		// 色値(BGR)
	vector<Vec3b> ColorANN;
	vector<float> IxNN;			// テクスチャ特徴 (計算時に 2^l 掛ける)
	vector<float> IxANN;
	vector<float> IyNN;
	vector<float> IyANN;
	vector<int> OcclusionChecker; // 補修領域確認値
	// -> (-1:類似先がが補修領域,0:非補修領域,1:補修ピクセルが補修領域,2:領域外)

	Patch();													// 初期化
	Patch(int, int, Mat&, Mat&, Mat&, ShiftMap&, int&);			// 任意の点とシフトマップ先とのコスト関数
	Patch(Point2i, Point2i, Mat&, Mat&, Mat&, ShiftMap&, int&);	// ２点比較時のコスト関数
	double costfunction(int);	// コスト関数(相違度)の計算
	// -> (0:オクルージョンなしd, 1:オクルージョンなしd^2, 2:オクルージョン含むd, 3:オクルージョン含むd^2)
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
			NN.push_back(Point2i(ppX, ppY));	// 補修ピクセル
			XY_p_temp = Point2i(ppX + sm_p.nnX(XY_p), ppY + sm_p.nnY(XY_p));
			ANN.push_back(XY_p_temp);			// 補修ピクセルのシフトマップ先
		}
	}
	int texture_index;
	hPatchSize = 0;	// 補修領域内ピクセル数
	nElsTotal = 0;	// 領域外ピクセル数
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
			NN.push_back(Point2i(P_1.x + ppX, P_1.y + ppY));	// 補修パッチ
			ANN.push_back(Point2i(P_2.x + ppX, P_2.y + ppY));	// 類似パッチ
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

	if (answer == 0 || Total == 0) { answer = -1; }		// 比較ピクセルがない場合-1を返す
	//if (answer == 0 || Total <= 10) { answer = -1; }
	else {
		answer = (double)answer / (double)Total;
		if (info == 0 || info == 2) { answer = sqrt(answer); }
	}
	return answer;
}


/* dilated occlusion H~ (if the patch included occluded pixels, return the posision) 
	第一引数:拡張前オクルージョン、第二引数:拡張後オクルージョン、第三引数:拡張後補修ピクセル*/
void find_Patch_pixcel(Mat& L_img, Mat& dilateOCC_img, vector<Point2i>& dilateOCC_pix) {
	int conclude_occ;
	Point2i now_p;
	uchar gray = (uchar)MAX_INTENSE;
	dilateOCC_pix.clear();

	L_img.copyTo(dilateOCC_img);	// 拡張前オクルージョン
	for (int y = 0; y < L_img.rows; y++) {
		for (int x = 0; x < L_img.cols; x++) {
			now_p = Point2i(x, y);
			conclude_occ = unoccluded_checker(now_p, L_img);
			if (conclude_occ != 0) {	// 拡張後オクルージョン
				dilateOCC_img.data[y * L_img.cols + x] = gray;
				dilateOCC_pix.push_back(Point2i(x, y));
			}
		}
	}
}
/* dilated data D~ (if the patch included occluded pixels, return the number of it) 
	第一引数:パッチの中心ピクセル、第二引数:オクルージョン*/
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
	return number_occ;	// 近傍N_pに含まれる補修ピクセル数
}


/* 非局所パッチ法関連の関数 */
void non_local_patch_matching(ImagePyramid&, Mat&);			// 非局所パッチ法 (l.410)
void InpaintingInitialisation(LClass&, ShiftMap&);			// Inpainting initialisation (l.526)
void PatchMatching_ANN(LClass&, ShiftMap&, Mat&, vector<Point2i>&, int&);		// Patch Match Algorithm (ANN) (l.598)
void PatchMatching_ANN_2D(LClass&, ShiftMap&, Mat&, vector<Point2i>&, int&);
void PatchMatching_All(LClass&, ShiftMap&, Mat&, vector<Point2i>&, int&);		// Patch Match Algorithm (全探索)
void Reconstruction(LClass&, ShiftMap&, int&);									// Reconstruction of U&T (l.789)
void Reconstruction_first(LClass&, ShiftMap&, vector<Point2i>&, Mat&, int&);	// Reconstruction of U&T at first

double SHIGMA(vector<double>&);								// sort & return 75th percentile (l.1093)
void firstReconstruction_SM(LClass&, ShiftMap&);			// Reconstruction of U at first
void firstReconstruction_COLOR(Mat, LClass&);				// Reconstruction at first
void annealingReconstruction(LClass&, ShiftMap&);			// Annealing
void update_pyramid(ImagePyramid&, LClass& newLClass);		// ピラミッド更新 (l.1240)


/* MRF関連の関数 */
void OCC_MRF_GaussSeidel(Mat&, ImagePyramid&, double&);				// MRF (l.1247)
void OCC_GMRF_GaussSeidel(Mat&, ImagePyramid&, double&, double&);	// MRF with Gamma (l.1585)