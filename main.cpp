#include "main.h"
#include <time.h>

int main()
{
	Read();
	clock_t start, end;	// 処理時間表示用

	/* 画像ピラミッド作成 */
	ImagePyramid image_pyramid_original = ImagePyramid();	// calculate O,U,T pyramid
	cout << "--- 画像情報 -------------------------------------------------" << endl;
	image_pyramid_original.show();
	cout << "--------------------------------------------------------------" << endl;
	/* パッチサイズ */
	cout << "パッチ : " << PATCHSIZEint << "×" << PATCHSIZEint << endl;
	cout << endl;

	/* 非局所パッチ法 */
	ImagePyramid image_pyramid_best;
	image_pyramid_best.copyTo(image_pyramid_original);
	Mat img_now;
	double MSE_best = -1.0, MSE_ave = 0.0, MSE_now = 0.0;
	for (int Exp = 0; Exp < Repetition; Exp++) {
		cout << "=== 非局所パッチ法 (" << Exp << ") ===============================================" << endl;
		start = clock();
		ImagePyramid image_pyramid = ImagePyramid();
		non_local_patch_matching(image_pyramid, img_now);	// 非局所パッチ法
		end = clock();
		const double time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
		cout << " patch_match   : time " << time << "[ms]" << endl;
		Evaluation(img_src, img_now, MSE_now);
		MSE_ave += MSE_now;

		/* 最適ピラミッド更新 */
		if (MSE_best == -1) {
			MSE_best = MSE_now;
			img_now.copyTo(img_dst);
			image_pyramid_best.copyTo(image_pyramid);
			cout << " UPDATE NEW BEST_PYRAMID..." << endl;
		}
		else {
			if (MSE_now < MSE_best) {
				MSE_best = MSE_now;
				img_now.copyTo(img_dst);
				image_pyramid_best.copyTo(image_pyramid);
				cout << " UPDATE NEW BEST_PYRAMID..." << endl;
			}
		}
		cout << endl;
	}
	MSE_ave = (double)MSE_ave / (double)Repetition;		// 平均MSE
	/* 実行結果確認 */
	cout << "非局所パッチ法 : 最適MSE = " << MSE_best << endl;
	cout << "　　　　　　　 : 平均MSE = " << MSE_ave << endl;
	cout << "======================================================================" << endl;
	cout << endl;
	img_dst.copyTo(img_dst2);
	img_dst.copyTo(img_dst3);

	/* MRF */
	double Alpha_best = Alpha, Alpha_now;
	MSE_best = -1.0;
	cout << "=== MRF 確率的画像処理 ===============================================" << endl;
	for (double Alpha_plus = 0; Alpha_plus < 0.0002; Alpha_plus = Alpha_plus + 0.00001) {
		Alpha_now = (double)(Alpha + Alpha_plus);	// 平滑化項より（平滑化パラメータ）
		img_dst.copyTo(img_now);

		start = clock();
		OCC_MRF_GaussSeidel(img_now, image_pyramid_best, Alpha_now);		// MRF
		end = clock();
		const double time2 = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
		cout << "Func()        : time " << time2 << "[ms]" << endl;
		Evaluation(img_src, img_now, MSE_now);
		cout << "Alpha = " << Alpha_now << " , MSE = " << MSE_now << endl;
		cout << endl;

		/* 最適補修後画像更新 */
		if (MSE_best == -1) {
			MSE_best = MSE_now;
			Alpha_best = Alpha_now;
			img_now.copyTo(img_dst2);
		}
		else {
			if (MSE_now < MSE_best) {
				MSE_best = MSE_now;
				Alpha_best = Alpha_now;
				img_now.copyTo(img_dst2);
			}
		}
	}
	cout << "======================================================================" << endl;
	cout << endl;

	/* GammaありMRF */
	double Alpha2_best = Alpha2, Alpha2_now;
	double Alpha2_max = 0.0006;
	double Gamma_best = Gamma, Gamma_now;
	MSE_best = -1.0;
	if (Alpha_best >= Alpha + 0.0002) { 
		Alpha2 = Alpha_best - 0.0002;
		Alpha2_max = Alpha2_max - 0.0002;
	}
	cout << "=== MRF 確率的画像処理 (with Gamma) ===================================" << endl;
	for (double Alpha2_plus = 0; Alpha2_plus < Alpha2_max; Alpha2_plus = Alpha2_plus + 0.00001) {
		Alpha2_now = (double)(Alpha2 + Alpha2_plus);	// 平滑化項より（平滑化パラメータ）
		for (double Gamma_plus = Gamma; Gamma_plus <= 0.000008; Gamma_plus = Gamma_plus + 0.0000005) {
			Gamma_now = Gamma_plus;								// Gammma値
			img_dst.copyTo(img_now);

			start = clock();
			OCC_GMRF_GaussSeidel(img_now, image_pyramid_best, Alpha2_now, Gamma_now);	// MRF with Gamma
			end = clock();
			const double time3 = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
			cout << "Func()        : time " << time3 << "[ms]" << endl;
			Evaluation(img_src, img_now, MSE_now);
			cout << "Alpha2 = " << Alpha2_now << " , Gamma = " << Gamma_now << " , MSE = " << MSE_now << endl;
			cout << endl;

			/* 最適補修後画像更新 */
			if (MSE_best == -1) {
				MSE_best = MSE_now;
				Alpha2_best = Alpha2_now;
				Gamma_best = Gamma_now;
				img_now.copyTo(img_dst3);
			}
			else {
				if (MSE_now < MSE_best) {
					MSE_best = MSE_now;
					Alpha2_best = Alpha2_now;
					Gamma_best = Gamma_now;
					img_now.copyTo(img_dst3);
				}
			}
		}
	}
	cout << "======================================================================" << endl;
	cout << endl;

	/* パラメータ表示 */
	cout << "--- パラメータ -----------------------------------------------" << endl;
	cout << "  Ratio      = " << (double)Ratio << "  （色値テクスチャ比率）" << endl;
	cout << "  Repetition = " << (int)Repetition << "  （パッチ法実験回数）" << endl;
	cout << "  Rambda = " << (double)Rambda << "  （データ項）" << endl;
	cout << "  Alpha  = " << (double)Alpha_best << "  （平滑化項）" << endl;
	cout << "  Sigma  = " << (double)Sigma << "  （ノイズレベル：標準偏差）" << endl;
	cout << "  Mean   = " << (double)Mean << "   （ノイズレベル：平均）" << endl;
	cout << "  Alpha2 = " << (double)Alpha2_best << "  （平滑化項 withGamma）" << endl;
	cout << "  Gamma  = " << (double)Gamma_best << "  （MRF:パラメータ）" << endl;
	cout << "--------------------------------------------------------------" << endl;
	cout << endl;

	/* 補修精度評価の関数 */
	double mse_num;
	cout << "=== 補修結果最終比較 =================================================" << endl;
	cout << "  PM   : ";
	Evaluation(img_src, img_dst, mse_num);
	cout << "  MRF  : ";
	Evaluation(img_src, img_dst2, mse_num);
	cout << "  GMRF : ";
	Evaluation(img_src, img_dst3, mse_num);
	cout << "======================================================================" << endl;
	cout << endl;

	Out();
	return 0;
}


// ファイル読み込み
void Read() {
	string file_src = "C:\\Users\\mon25\\Desktop\\inpainting_program\\src.jpg";			// 入力画像のファイル名
	string file_src2 = "C:\\Users\\mon25\\Desktop\\inpainting_program\\occlusion.jpg";	// 補修領域画像のファイル名
	img_src = imread(file_src, 1);		// 入力画像（カラー）の読み込み
	img_src2 = imread(file_src, 0);		// 入力画像（グレースケール）の読み込み
	img_mask = imread(file_src2, 0);	// マスク画像（グレースケール）の読み込み

	/* パラメータ定義 */
	width = img_src.cols;
	height = img_src.rows;
	MAX_DATA = width * height;
	img_dst = Mat(Size(width, height), CV_8UC3);	// 出力画像（カラー）の初期化設定
	img_dst2 = Mat(Size(width, height), CV_8UC3);
	img_dst3 = Mat(Size(width, height), CV_8UC3);

	/* 補修領域の情報を削除 */
	Mat img_mask_not;
	threshold(img_mask, img_mask, 100, 255, THRESH_BINARY);	// マスク画像の2値変換
	bitwise_not(img_mask, img_mask_not);					// マスク画像（白黒反転）
	img_src.copyTo(img_SRC, img_mask_not);		// 補修する画像作成（カラー）
	img_src2.copyTo(img_SRC2, img_mask_not);	// 補修する画像作成（グレースケール）

	/* Occlusionを指定した色に変更 (Red) */
	//Vec3b color;
	//color[0] = 0, color[1] = 0, color[2] = 255;	// BGR
	//for (int PL = 0; PL <= L; PL++) {
	//	for (int indY = 0; indY < height; indY++) {
	//		for (int indX = 0; indX < width; indX++) {
	//			if (img_mask.data[indY * width + indX] != 0) {
	//				img_SRC.at<Vec3b>(indY, indX) = color;
	//			}
	//		}
	//	}
	//}
}
// ファイル書き出し
void Out() {
	string file_dst = "C:\\Users\\mon25\\Desktop\\inpainting_program\\dst.jpg";		// 出力画像のファイル名(PatchMatch)
	string file_dst2 = "C:\\Users\\mon25\\Desktop\\inpainting_program\\dst2.jpg";	// 出力画像のファイル名(MRF)
	string file_dst3 = "C:\\Users\\mon25\\Desktop\\inpainting_program\\dst3.jpg";	// 出力画像のファイル名(GammaありMRF)
	string file_dst4 = "C:\\Users\\mon25\\Desktop\\inpainting_program\\dst_hist.jpg";		// 出力画像のファイル名(PatchMatchのヒストグラム)
	string file_dst5 = "C:\\Users\\mon25\\Desktop\\inpainting_program\\dst_hist2.jpg";		// 出力画像のファイル名(MRFのヒストグラム)
	string file_dst6 = "C:\\Users\\mon25\\Desktop\\inpainting_program\\dst_hist3.jpg";		// 出力画像のファイル名(GammaありMRFのヒストグラム)
	string file_dst7 = "C:\\Users\\mon25\\Desktop\\inpainting_program\\dst_hist_src.jpg";	// 出力画像のファイル名(入力画像のヒストグラム)

	/* ウィンドウ生成 */
	namedWindow(win_src, WINDOW_AUTOSIZE);
	namedWindow(win_dst, WINDOW_AUTOSIZE);
	namedWindow(win_dst2, WINDOW_AUTOSIZE);
	namedWindow(win_dst3, WINDOW_AUTOSIZE);

	/* ヒストグラム作成 & KLD計算 */
	Mat hist1, hist2, hist3, hist_src;	// ヒストグラム画像
	double kld1, kld2, kld3, kld4;
	drawHist_Color(img_src, img_dst, hist1, kld1);
	drawHist_Color(img_src, img_dst2, hist2, kld2);
	drawHist_Color(img_src, img_dst3, hist3, kld3);
	drawHist_Color(img_src, img_src, hist_src, kld4);
	cout << "=== 補修結果最終比較(KLD) ============================================" << endl;
	cout << "  PM   = " << kld1 << endl;
	cout << "  MRF  = " << kld2 << endl;
	cout << "  GMRF = " << kld3 << endl;
	cout << "======================================================================" << endl;

	/* 画像の表示 & 保存 */
	imshow(win_src, img_SRC);		// 入力画像を表示
	imshow(win_dst, img_dst);		// 出力画像を表示
	imshow(win_dst2, img_dst2);
	imshow(win_dst3, img_dst3);
	imwrite(file_dst, img_dst);		// 処理結果の保存(補修画像)
	imwrite(file_dst2, img_dst2);
	imwrite(file_dst3, img_dst3);
	imwrite(file_dst4, hist1);		// 処理結果の保存(ヒストグラム)
	imwrite(file_dst5, hist2);
	imwrite(file_dst6, hist3);
	imwrite(file_dst7, hist_src);

	waitKey(0); // キー入力待ち
}


// 補修精度評価（カラー画像のみ比較）
void Evaluation(Mat& original, Mat& inpaint, double& dstMSE) {
	if (original.channels() != 3 || inpaint.channels() != 3) {
		cout << "ERROR! MSE()  :  Can't calcurate MSE because of its channel." << endl;
	}
	else if (original.rows != inpaint.rows || original.cols != inpaint.cols) {
		cout << "ERROR! MSE()  :  Image size is wrong." << endl;
	}
	else {
		double inpaint_MSE = 0.0, inpaint_PSNR = 0.0, inpaint_SSIM;
		MSE(original, inpaint, inpaint_MSE, inpaint_PSNR);
		inpaint_SSIM = SSIM(original, inpaint);
		cout << " MSE = " << inpaint_MSE << " , PSNR = " << inpaint_PSNR << " , SSIM = " << inpaint_SSIM << endl;
		dstMSE = inpaint_MSE;
	}
}
// MSE&PSNR算出
void MSE(Mat& Original, Mat& Inpaint, double& MSE, double& PSNR) {
	double MSE_sum = 0.0;	// MSE値
	double image_cost;		// 画素値の差分
	int compare_size, color_ind;
	int occ_pix_count = 0;

	/* MSE計算(RGB) */
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			image_cost = 0.0;
			color_ind = i * width * 3 + j * 3;
			for (int k = 0; k < 3; k++) {
				image_cost = image_cost + pow((int)Inpaint.data[color_ind] - (int)Original.data[color_ind], 2.0);
				color_ind++;
			}
			MSE_sum = MSE_sum + (double)image_cost;
			occ_pix_count++;
		}
	}
	compare_size = occ_pix_count * 3;
	MSE_sum = (double)MSE_sum / (double)compare_size;

	/* PSNR計算 */
	double PSNR_sum;
	PSNR_sum = 20 * (double)log10(MAX_INTENSE) - 10 * (double)log10(MSE_sum);

	MSE = MSE_sum;
	PSNR = PSNR_sum;
}
// SSIM算出
double SSIM(Mat& image_1, Mat& image_2) {
	const double C1 = pow(0.01 * 255, 2), C2 = pow(0.03 * 255, 2);

	Mat I1, I2;
	image_1.convertTo(I1, CV_32F);	// cannot calculate on one byte large values
	image_2.convertTo(I2, CV_32F);
	Mat I2_2 = I2.mul(I2);			// I2^2
	Mat I1_2 = I1.mul(I1);			// I1^2
	Mat I1_I2 = I1.mul(I2);			// I1 * I2

	Mat mu1, mu2;   // PRELIMINARY COMPUTING
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);
	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);

	Mat sigma1_2, sigma2_2, sigma12;
	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;
	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;
	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;

	Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

	Mat ssim_map;
	divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;
	Scalar mssim = mean(ssim_map); // mssim = average of ssim map

	/* SSIM平均(RGB) */
	double SSIM;
	SSIM = (double)mssim[0] + (double)mssim[1] + (double)mssim[2];
	SSIM = (double)SSIM / 3.0;
	return SSIM;
}
// KLD算出
double KLD(vector<Mat>& img1_RGB, vector<Mat>& img2_RGB) {
	int PIX_SIZE;
	double Px, Qx, Log_func;
	double KLD = 0.0;

	if (img1_RGB.size() != img2_RGB.size() || img1_RGB[0].cols != img2_RGB[0].cols || img1_RGB[0].rows != img2_RGB[0].rows) {
		cout << "ERROR! KLD()  :  Image size isn't the same." << endl;
		cout << img1_RGB.size() << " <-> " << img2_RGB.size() << endl;
	}
	else if (img1_RGB.size() != 3) {
		cout << "ERROR! KLD()  :  Image isn't color." << endl;
		cout << img1_RGB.size() << endl;
	}
	else {
		PIX_SIZE = img1_RGB[0].cols * img1_RGB[0].rows;
		for (int KLD_c = 0; KLD_c < img1_RGB.size(); KLD_c++) {
			for (int KLD_i = 0; KLD_i < PIX_SIZE; KLD_i++) {
				Px = (double)img1_RGB[KLD_c].data[KLD_i] / (double)PIX_SIZE;
				Qx = (double)img2_RGB[KLD_c].data[KLD_i] / (double)PIX_SIZE;
				if (Px > 0 && Qx > 0) { Log_func = log((double)(Px / Qx)); }
				else { Log_func = 0.0; }
				KLD += (double)(Px * Log_func);
			}
		}
		KLD = (double)KLD / 3.0;
	}

	return KLD;
}
// ヒストグラム計算&描画（第一引数:image_src、第二引数:image_color、第三引数:hist、第四引数:KLD）
void drawHist_Color(Mat& targetImg, Mat& Color, Mat& image_hist, double& KLD_return) {
	if (Color.channels() != 3 || targetImg.channels() != 3) {
		cout << "ERROR! drawHist_Color()  :  Can't draw Histgram because of its channel." << endl;
	}
	else if (Color.cols != targetImg.cols || Color.rows != targetImg.rows) {
		cout << "ERROR! drawHist_Color()  :  Can't draw Histgram because of image size." << endl;
	}
	else {
		vector<Mat> image_SRC;		// KLD算出用RGB
		vector<Mat> image_INPAINT;

		Mat channels_src[3];
		Mat channels[3];
		int cha_index, ha_index;
		uchar Gray;
		for (int channel = 0; channel < 3; channel++) {
			channels_src[channel] = Mat(Size(targetImg.cols, targetImg.rows), CV_8UC1);
			channels[channel] = Mat(Size(Color.cols, Color.rows), CV_8UC1);
			for (int i = 0; i < Color.rows; i++) {
				for (int j = 0; j < Color.cols; j++) {
					cha_index = i * Color.cols * 3 + j * 3 + channel;
					ha_index = i * Color.cols + j;
					Gray = (uchar)targetImg.data[cha_index];
					channels_src[channel].data[ha_index] = Gray;
					Gray = (uchar)Color.data[cha_index];
					channels[channel].data[ha_index] = Gray;
				}
			}
		}

		/* 変数宣言 */
		Mat R, G, B;
		Mat Rs, Gs, Bs;
		int hist_size = 256;
		float range[] = { 0, 256 };
		const float* hist_range = range;

		/* 画素数を数える */
		calcHist(&channels[0], 1, 0, Mat(), B, 1, &hist_size, &hist_range);
		calcHist(&channels[1], 1, 0, Mat(), G, 1, &hist_size, &hist_range);
		calcHist(&channels[2], 1, 0, Mat(), R, 1, &hist_size, &hist_range);
		image_INPAINT.push_back(B);
		image_INPAINT.push_back(G);
		image_INPAINT.push_back(R);
		calcHist(&channels_src[0], 1, 0, Mat(), Bs, 1, &hist_size, &hist_range);
		calcHist(&channels_src[1], 1, 0, Mat(), Gs, 1, &hist_size, &hist_range);
		calcHist(&channels_src[2], 1, 0, Mat(), Rs, 1, &hist_size, &hist_range);
		image_SRC.push_back(B);
		image_SRC.push_back(G);
		image_SRC.push_back(R);

		/* 確認（ヒストグラム高さ固定のため）*/
		double Min_count[3], Max_count[3];
		for (int ch = 0; ch < 3; ch++) {
			if (ch == 0) { minMaxLoc(B, &Min_count[ch], &Max_count[ch]); }
			else if (ch == 1) { minMaxLoc(G, &Min_count[ch], &Max_count[ch]); }
			else if (ch == 2) { minMaxLoc(R, &Min_count[ch], &Max_count[ch]); }
			if (Max_count[ch] > MAX_COUNT_DEF) {
				cout << "NOTE! 設定最大数不適：MAX = " << (int)Max_count[ch] << " , channel = " << ch << endl;
			}
		}

		/* ヒストグラム生成用の画像を作成 */
		image_hist = Mat(Size(276, 320), CV_8UC3, Scalar(255, 255, 255));

		/* 背景を描画（見やすくするためにヒストグラム部分の背景をグレーにする） */
		for (int i = 0; i < 3; i++) {
			rectangle(image_hist, Point(10, 10 + 100 * i), Point(265, 100 + 100 * i), Scalar(230, 230, 230), -1);
		}

		for (int i = 0; i < 256; i++) {
			line(image_hist, Point(10 + i, 100), Point(10 + i, 100 - (int)((float)(R.at<float>(i) / MAX_COUNT_DEF) * 80)), Scalar(0, 0, 255), 1, 8, 0);
			line(image_hist, Point(10 + i, 200), Point(10 + i, 200 - (int)((float)(G.at<float>(i) / MAX_COUNT_DEF) * 80)), Scalar(0, 255, 0), 1, 8, 0);
			line(image_hist, Point(10 + i, 300), Point(10 + i, 300 - (int)((float)(B.at<float>(i) / MAX_COUNT_DEF) * 80)), Scalar(255, 0, 0), 1, 8, 0);

			if (i % 10 == 0) {		// 横軸10ずつラインを引く
				line(image_hist, Point(10 + i, 100), Point(10 + i, 10),
					Scalar(170, 170, 170), 1, 8, 0);
				line(image_hist, Point(10 + i, 200), Point(10 + i, 110),
					Scalar(170, 170, 170), 1, 8, 0);
				line(image_hist, Point(10 + i, 300), Point(10 + i, 210),
					Scalar(170, 170, 170), 1, 8, 0);

				if (i % 50 == 0) {	// 横軸50ずつ濃いラインを引く
					line(image_hist, Point(10 + i, 100), Point(10 + i, 10),
						Scalar(50, 50, 50), 1, 8, 0);
					line(image_hist, Point(10 + i, 200), Point(10 + i, 110),
						Scalar(50, 50, 50), 1, 8, 0);
					line(image_hist, Point(10 + i, 300), Point(10 + i, 210),
						Scalar(50, 50, 50), 1, 8, 0);
				}
			}
		}

		KLD_return = KLD(image_SRC, image_INPAINT);
	}
}


/* 非局所パッチ法 */
void non_local_patch_matching(ImagePyramid& img_pyr, Mat& dstIMG) {
	LClass Image = LClass(img_pyr, L);		// LEVEL:(0~)L
	ShiftMap SM = ShiftMap(Image);			// shift map decide randomly

	/* Initialisation */
	cout << " Initialisation..." << endl;	// 実行結果確認
	InpaintingInitialisation(Image, SM);
	cout << endl;
	//SM.zero(Image.imgO);
	Image.imgU.copyTo(img_dst3);			// 確認用

	/* iteration */
	Mat UP_img;
	Mat Dilate_Occlusion;			// dilate occlusion H~
	vector<Point2i> PM_occ;
	int c_num;
	Point2i occ_point;
	for (int pLEVEL = L; pLEVEL >= 0; pLEVEL--) {
		int K = 0;			// 反復回数
		double e = 1.0;		// 閾値
		while (e > 0.1 && K < 20) {
			// UpSample
			if (K == 0) {
				if (pLEVEL < L) {
					Image.imgU.copyTo(UP_img);
					Image.upsample(img_pyr);
					SM.upsample(Image.imgO);

					// Reconstruction at first
					cout << " Upsampling & first reconstruction at: " << pLEVEL << endl;
					firstReconstruction_SM(Image, SM);			// shiftmap upsampling
					//firstReconstruction_COLOR(UP_img, Image);	// color upsampling
				}
				find_Patch_pixcel(Image.imgO, Dilate_Occlusion, PM_occ);
				/*Dilate_Occlusion = Image.imgO;
				PM_occ = Image.occ_p;*/
				c_num = Image.occ_index;
			}

			if (pLEVEL != -1) {
				vector<Vec3b> Before;
				for (int i = 0; i < c_num; i++) {
					occ_point = Image.occ_p[i];
					Before.push_back(Image.imgU.at<Vec3b>(occ_point.y, occ_point.x));
				}

				//cout << " PatchMatching..." << endl;	// 実行結果確認
				int costNUM = 3;
				//PatchMatching_All(Image, SM, Dilate_Occlusion, PM_occ, costNUM);		// Patch Match(全探索)
				PatchMatching_ANN(Image, SM, Dilate_Occlusion, PM_occ, costNUM); 		// Patch Match(ANN)
				//PatchMatching_ANN_2D(Image, SM, Dilate_Occlusion, PM_occ, costNUM);
				//if (K == 0 && pLEVEL == L) { PatchMatching_All(Image, SM, Dilate_Occlusion, PM_occ, costNUM); }
				//else { PatchMatching_ANN(Image, SM, Dilate_Occlusion, PM_occ, costNUM); }
				//else { PatchMatching_ANN_2D(Image, SM, Dilate_Occlusion, PM_occ, costNUM); }

				//cout << " Reconstruction U&T..." << endl;	// 実行結果確認
				Reconstruction(Image, SM, costNUM);		// Reconstruction U&T

				vector<Vec3b> After;
				for (int i = 0; i < c_num; i++) {
					occ_point = Image.occ_p[i];
					After.push_back(Image.imgU.at<Vec3b>(occ_point.y, occ_point.x));
				}

				double Unorm = norm(Before, After, NORM_L2);
				int avg_num = 3 * c_num;
				e = (double)(Unorm) / (double)(avg_num);
				//if (pLEVEL == 2 && K == 0) { K = 20; }	// K(0~)回目で中断
				K++;
				Before.clear();
				After.clear();
				cout << " 成功: " << pLEVEL << ":" << K << ":" << e << endl;	// 実行結果確認
			}
			else {
				cout << " First: " << pLEVEL << endl;
				e = 0.1;
			}
		}
		/* Annealing */
		if (pLEVEL == 0) {
			cout << "Annealing実行" << endl;
			annealingReconstruction(Image, SM);
		}
		/*cout << " Annealing実行" << endl;
		annealingReconstruction(Image, SM);*/

		/* image pyramid 更新 */
		update_pyramid(img_pyr, Image);
	}
	cout << endl;

	/* 出力 */
	Image.imgU.copyTo(dstIMG);
}

/* Inpainting initialisation (「オニオンピール」手法による初期値設定) */
void InpaintingInitialisation(LClass& imageL, ShiftMap& smL) {
	Mat current_occlusion;					// current occlusion H'
	imageL.imgO.copyTo(current_occlusion);
	int OccPixelNum = imageL.occ_index;		// the number of current occlusion H'

	/* 拡張補修領域の非補修領域ピクセルにおけるシフトマップの決定 */
	Mat Dilate_Occlusion;					// current dilated occlusion H'~
	vector<Point2i> PM_occ;					// the number of current dilated occlusion H'~
	int OCC_int = 1;
	find_Patch_pixcel(imageL.imgO, Dilate_Occlusion, PM_occ);
	PatchMatching_All(imageL, smL, Dilate_Occlusion, PM_occ, OCC_int);
	//Dilate_Occlusion = current_occlusion.clone();
	//PM_occ = imageL.occ_p;
	/* 確認用 */
	cout << "   Dilated OccPixel: " << PM_occ.size() << endl;
	cout << "   OccPixel: " << OccPixelNum << endl;

	/* 補修領域の初期化 */
	while (OccPixelNum != 0) {		// aH' <- H'
		vector<Point2i> OccPixel;	// the position of current layer to inpaint aH'(B)
		Point2i pix;
		int indexP;
		for (int y = 0; y < imageL.YSIZE; y++) {
			for (int x = 0; x < imageL.XSIZE; x++) {
				pix = Point2i(x, y);
				indexP = y * current_occlusion.cols + x;
				if (current_occlusion.data[indexP] != 0) {
					if (current_occlusion.data[indexP - 1] == 0 && (x - 1) >= 0) {
						OccPixel.push_back(pix);	// left
					}
					else if (current_occlusion.data[indexP + 1] == 0 && (x + 1) < current_occlusion.cols) {
						OccPixel.push_back(pix);	// right
					}
					else if (current_occlusion.data[indexP - current_occlusion.cols] == 0 && (y - 1) >= 0) {
						OccPixel.push_back(pix);	// up
					}
					else if (current_occlusion.data[indexP + current_occlusion.cols] == 0 && (y + 1) < current_occlusion.rows) {
						OccPixel.push_back(pix);	// down
					}
					/*else { cout << "occlusion pixel in occlusion: " << pix << endl; }*/
				}
			}
		}

		// Patch Matching
		int costNUM = 1;
		//PatchMatching_All(imageL, smL, Dilate_Occlusion, OccPixel, costNUM);		// Patch Match(全探索)
		PatchMatching_ANN(imageL, smL, Dilate_Occlusion, OccPixel, costNUM);		// Patch Match(ANN探索)
		//PatchMatching_ANN_2D(imageL, smL, Dilate_Occlusion, OccPixel, costNUM);

		// Reconstruction U&T
		Reconstruction_first(imageL, smL, OccPixel, current_occlusion, costNUM);

		// Erosion(H',B)
		Point2i Point;
		int zero = 0;
		uchar gray = (uchar)zero;
		for (int occl_num = 0; occl_num < OccPixel.size(); occl_num++) {
			Point = OccPixel[occl_num];
			current_occlusion.data[Point.y * current_occlusion.cols + Point.x] = gray;
		}
		OccPixel.clear();

		OccPixelNum = 0;		// the number of current occlusion H'
		for (int num = 0; num < current_occlusion.rows * current_occlusion.cols; num++) {
			if (current_occlusion.data[num] != 0) {
				OccPixelNum++;
			}
		}
		/* 確認用 */
		cout << "   OccPixel: " << OccPixelNum << endl;
	}
}

/* Patch Match Algorithm (ANN探索) */
void PatchMatching_ANN(LClass& image, ShiftMap& sm, Mat& dilate_occ, vector<Point2i>& search_p, int& cost_num) {
	/* 最大探索範囲指定（画像幅の縦横で長い辺）*/
	if (image.XSIZE > image.YSIZE) { r_max = image.XSIZE; }
	else { r_max = image.YSIZE; }

	int Occ_number = search_p.size();		// the number of occlusion H'

	Point2i P, A, B, P2;
	double p, q;				// 比較コスト
	double Round;				// シフトマップ周辺探索領域
	int MaxRoundx, MaxRoundy, MinRoundx, MinRoundy;
	int occ_checker;
	Point2i Point, point_now, minCost_Point;
	vector<Point2i> Point_abc;	// 比較ピクセル
	Patch PATCH;
	if (cost_num != 1 && cost_num != 3) { cout << "ERROR! cost_num in PatchMatching_ANN()" << endl; }

	/* iteration */
	for (int kk = 0; kk < 10; kk++) {
		/* 左、上のシフトマップ */
		for (int pp = 0; pp < Occ_number; pp++) {
			P = search_p[pp];			// Point_abc = a, b, c
			Point_abc.push_back(P);
			A = Point2i(P.x - 1, P.y);
			Point_abc.push_back(A);
			B = Point2i(P.x, P.y - 1);
			Point_abc.push_back(B);
			/* 最小コスト探索 in a,b,c */
			p = DBL_MAX;				// 最大値を代入
			minCost_Point.x = P.x + sm.nnX(P);
			minCost_Point.y = P.y + sm.nnY(P);
			for (int Point_Index = 0; Point_Index < 3; Point_Index++) {
				point_now = Point_abc[Point_Index];
				if (point_now.x >= 0 && point_now.x < image.XSIZE && point_now.y >= 0 && point_now.y < image.YSIZE) {
					Point.x = P.x + sm.nnX(point_now);
					Point.y = P.y + sm.nnY(point_now);
					if (Point.x >= 0 && Point.x < image.XSIZE && Point.y >= 0 && Point.y < image.YSIZE) {
						occ_checker = unoccluded_checker(Point, dilate_occ);
						if (occ_checker == 0) {
							PATCH = Patch(P, Point, image.imgU, image.imgT, dilate_occ, sm, image.LEVEL);
							q = PATCH.costfunction(cost_num);
							PATCH = Patch();
						}

						if (p > q && q > 0) {
							p = q;
							minCost_Point = Point;
						}
					}
				}
			}
			if (minCost_Point != P) {
				minCost_Point.x = minCost_Point.x - P.x;
				minCost_Point.y = minCost_Point.y - P.y;
				sm.put(P, minCost_Point);
				Point_abc.clear();
			}
		}
		/* 右、下のシフトマップ */
		for (int pp = Occ_number - 1; pp >= 0; pp--) {
			P = search_p[pp];			// Point_abc = a, b, c
			Point_abc.push_back(P);
			A = Point2i(P.x + 1, P.y);
			Point_abc.push_back(A);
			B = Point2i(P.x, P.y + 1);
			Point_abc.push_back(B);
			/* 最小コスト探索 in a,b,c */
			p = DBL_MAX;				// 最大値を代入
			minCost_Point.x = P.x + sm.nnX(P);
			minCost_Point.y = P.y + sm.nnY(P);
			for (int Point_Index = 0; Point_Index < 3; Point_Index++) {
				point_now = Point_abc[Point_Index];
				if (point_now.x >= 0 && point_now.x < image.XSIZE && point_now.y >= 0 && point_now.y < image.YSIZE) {
					Point.x = P.x + sm.nnX(point_now);
					Point.y = P.y + sm.nnY(point_now);
					if (Point.x >= 0 && Point.x < image.XSIZE && Point.y >= 0 && Point.y < image.YSIZE) {
						occ_checker = unoccluded_checker(Point, dilate_occ);
						if (occ_checker == 0) {
							PATCH = Patch(P, Point, image.imgU, image.imgT, dilate_occ, sm, image.LEVEL);
							q = PATCH.costfunction(cost_num);
							PATCH = Patch();
						}

						if (p > q && q > 0) {
							p = q;
							minCost_Point = Point;
						}
					}
				}
			}
			if (minCost_Point != P) {
				minCost_Point.x = minCost_Point.x - P.x;
				minCost_Point.y = minCost_Point.y - P.y;
				sm.put(P, minCost_Point);
				Point_abc.clear();
			}
		}
		/* シフトマップ先の周辺 */
		for (int pp = 0; pp < Occ_number; pp++) {
			P = search_p[pp];
			P2.x = P.x + sm.nnX(P);
			P2.y = P.y + sm.nnY(P);
			Round = (int)((r_max * pow(Ro, kk)) / 2);

			if (Round > 0) {
				/* 検索対象正方範囲を定義 */
				MaxRoundx = P2.x + Round;
				if (MaxRoundx >= image.XSIZE) { MaxRoundx = image.XSIZE; }
				MaxRoundy = P2.y + Round;
				if (MaxRoundy >= image.YSIZE) { MaxRoundy = image.YSIZE; }
				MinRoundx = P2.x - Round;
				if (MinRoundx < 0) { MinRoundx = 0; }
				MinRoundy = P2.y - Round;
				if (MinRoundy < 0) { MinRoundy = 0; }

				/* 検索対象ピクセルを指定範囲からランダムに選出 */
				int rand_check = 1, kkk = 0;
				while (rand_check == 1 && kkk < 1000) {
					Point.x = (int)(rand() % MaxRoundx + MinRoundx);	// 比較先x(ramdom)
					Point.y = (int)(rand() % MaxRoundy + MinRoundy);	// 比較先y(ramdom)
					occ_checker = unoccluded_checker(Point, dilate_occ);
					if (occ_checker == 0) {
						rand_check = 0;
					}
					kkk++;
				}
				if (rand_check == 1 && kkk == 1000) { break; }	// ループ対策

				/* 最小コストと比較 */
				PATCH = Patch(P, P2, image.imgU, image.imgT, dilate_occ, sm, image.LEVEL);
				p = PATCH.costfunction(cost_num);	// 現在の最適値を代入
				if (p < 0) { p = DBL_MAX; }
				minCost_Point = P2;
				PATCH = Patch(P, Point, image.imgU, image.imgT, image.imgO, sm, image.LEVEL);
				q = PATCH.costfunction(cost_num);
				PATCH = Patch();
				if (p > q && q > 0) {
					p = q;
					minCost_Point = Point;

					minCost_Point = minCost_Point - P;
					sm.put(P, minCost_Point);
				}
			}
		}
	}
}
void PatchMatching_ANN_2D(LClass& image, ShiftMap& sm, Mat& dilate_occ, vector<Point2i>& search_p, int& cost_num) {
	/* 最大探索範囲指定（画像幅の縦横で長い辺）*/
	if (image.XSIZE > image.YSIZE) { r_max = image.XSIZE; }
	else { r_max = image.YSIZE; }

	int Occ_number = search_p.size();		// the number of occlusion H'

	Point2i P, A, B, P2;
	double p, q;				// 比較コスト
	double Round;				// シフトマップ周辺探索領域
	int z_max = 1;				// 周辺探索ピクセル数
	int MaxRoundx, MaxRoundy, MinRoundx, MinRoundy;
	int occ_checker;
	Point2i Point, point_now, minCost_Point;
	vector<Point2i> Point_abc;	// 比較ピクセル
	Patch PATCH;
	if (cost_num != 1 && cost_num != 3) { cout << "ERROR! cost_num in PatchMatching_ANN()" << endl; }

	/* iteration */
	for (int kk = 0; kk < 10; kk++) {
		for (int pp = 0; pp < Occ_number; pp++) {
			/* kが偶数(kkが奇数)の時⇒左、上のシフトマップ */
			if (kk % 2 != 0) {
				P = search_p[pp];			// Point_abc = a, b, c
				Point_abc.push_back(P);
				A = Point2i(P.x - 1, P.y);
				Point_abc.push_back(A);
				B = Point2i(P.x, P.y - 1);
				Point_abc.push_back(B);
				/* 最小コスト探索 in a,b,c */
				p = DBL_MAX;				// 最大値を代入
				minCost_Point.x = P.x + sm.nnX(P);
				minCost_Point.y = P.y + sm.nnY(P);
				for (int Point_Index = 0; Point_Index < 3; Point_Index++) {
					point_now = Point_abc[Point_Index];
					if (point_now.x >= 0 && point_now.x < image.XSIZE && point_now.y >= 0 && point_now.y < image.YSIZE) {
						Point.x = P.x + sm.nnX(point_now);
						Point.y = P.y + sm.nnY(point_now);
						if (Point.x >= 0 && Point.x < image.XSIZE && Point.y >= 0 && Point.y < image.YSIZE) {
							occ_checker = unoccluded_checker(Point, dilate_occ);
							if (occ_checker == 0) {
								PATCH = Patch(P, Point, image.imgU, image.imgT, dilate_occ, sm, image.LEVEL);
								q = PATCH.costfunction(cost_num);
								PATCH = Patch();
							}

							if (p > q && q > 0) {
								p = q;
								minCost_Point = Point;
							}
						}
					}
				}
				if (minCost_Point != P) {
					minCost_Point.x = minCost_Point.x - P.x;
					minCost_Point.y = minCost_Point.y - P.y;
					sm.put(P, minCost_Point);
					Point_abc.clear();
				}
			}
			/* kが奇数の時⇒右、下のシフトマップ */
			else {
				P = search_p[pp];			// Point_abc = a, b, c
				Point_abc.push_back(P);
				A = Point2i(P.x + 1, P.y);
				Point_abc.push_back(A);
				B = Point2i(P.x, P.y + 1);
				Point_abc.push_back(B);
				/* 最小コスト探索 in a,b,c */
				p = DBL_MAX;				// 最大値を代入
				minCost_Point.x = P.x + sm.nnX(P);
				minCost_Point.y = P.y + sm.nnY(P);
				for (int Point_Index = 0; Point_Index < 3; Point_Index++) {
					point_now = Point_abc[Point_Index];
					if (point_now.x >= 0 && point_now.x < image.XSIZE && point_now.y >= 0 && point_now.y < image.YSIZE) {
						Point.x = P.x + sm.nnX(point_now);
						Point.y = P.y + sm.nnY(point_now);
						if (Point.x >= 0 && Point.x < image.XSIZE && Point.y >= 0 && Point.y < image.YSIZE) {
							occ_checker = unoccluded_checker(Point, dilate_occ);
							if (occ_checker == 0) {
								PATCH = Patch(P, Point, image.imgU, image.imgT, dilate_occ, sm, image.LEVEL);
								q = PATCH.costfunction(cost_num);
								PATCH = Patch();
							}

							if (p > q && q > 0) {
								p = q;
								minCost_Point = Point;
							}
						}
					}
				}
				if (minCost_Point != P) {
					minCost_Point.x = minCost_Point.x - P.x;
					minCost_Point.y = minCost_Point.y - P.y;
					sm.put(P, minCost_Point);
					Point_abc.clear();
				}
			}

			z_max = (int)floor((double)-log(r_max) / (double)log(Ro));
			//cout << "z_max = " << z_max << endl;	// 確認用
			/* シフトマップ先の周辺 */
			if (z_max > 0) {
				for (int zz = 0; zz < z_max; zz++) {
					P = search_p[pp];
					P2.x = P.x + sm.nnX(P);
					P2.y = P.y + sm.nnY(P);
					Round = (int)((r_max * pow(Ro, zz)) / 2);
					//cout << zz << " : Round = " << Round << endl;	// 確認用
					if (Round > 0) {
						/* 検索対象正方範囲を定義 */
						MaxRoundx = P2.x + Round;
						if (MaxRoundx >= image.XSIZE) { MaxRoundx = image.XSIZE; }
						MaxRoundy = P2.y + Round;
						if (MaxRoundy >= image.YSIZE) { MaxRoundy = image.YSIZE; }
						MinRoundx = P2.x - Round;
						if (MinRoundx < 0) { MinRoundx = 0; }
						MinRoundy = P2.y - Round;
						if (MinRoundy < 0) { MinRoundy = 0; }

						/* 検索対象ピクセルを指定範囲からランダムに選出 */
						int rand_check = 1, kkk = 0;
						while (rand_check == 1 && kkk < 1000) {
							Point.x = (int)(rand() % MaxRoundx + MinRoundx);	// 比較先x(ramdom)
							Point.y = (int)(rand() % MaxRoundy + MinRoundy);	// 比較先y(ramdom)
							occ_checker = unoccluded_checker(Point, dilate_occ);
							if (occ_checker == 0) {
								rand_check = 0;
							}
							kkk++;
						}
						if (rand_check == 1 && kkk == 1000) { break; }	// ループ対策

						/* 最小コストと比較 */
						PATCH = Patch(P, P2, image.imgU, image.imgT, dilate_occ, sm, image.LEVEL);
						p = PATCH.costfunction(cost_num);	// 現在の最適値を代入
						if (p < 0) { p = DBL_MAX; }
						minCost_Point = P2;
						PATCH = Patch(P, Point, image.imgU, image.imgT, image.imgO, sm, image.LEVEL);
						q = PATCH.costfunction(cost_num);
						PATCH = Patch();
						if (p > q && q > 0) {
							p = q;
							minCost_Point = Point;

							minCost_Point = minCost_Point - P;
							sm.put(P, minCost_Point);
						}
					}
					else { break; }
				}
			}
		}
	}
}
/* Patch Match Algorithm (全探索) */
void PatchMatching_All(LClass& image, ShiftMap& sm, Mat& dilate_occ, vector<Point2i>& search_p, int& cost_num) {
	int Occ_number = search_p.size();

	double p, q;		// 比較コスト
	int p_ind, occ_checker;
	Point2i Point, point_now, minCost_Point;
	Patch PATCH;
	int Patch_start = -PATCHstart;
	int Patch_end = image.YSIZE + PATCHstart;
	if (cost_num != 1 && cost_num != 3) { cout << "ERROR! cost_num in PatchMatching_All()" << endl; }

	for (int pp = 0; pp < Occ_number; pp++) {
		Point = search_p[pp];
		p = DBL_MAX;	// 最大値を代入
		minCost_Point.x = Point.x + sm.nnX(Point);
		minCost_Point.y = Point.y + sm.nnY(Point);
		for (int i = Patch_start; i < Patch_end; i++) {
			for (int j = Patch_start; j < Patch_end; j++) {
				point_now = Point2i(j, i);
				p_ind = i * image.XSIZE + j;
				if (point_now.x >= 0 && point_now.x < image.XSIZE && point_now.y >= 0 && point_now.y < image.YSIZE) {
					if (image.imgO.data[p_ind] == 0) {
						occ_checker = unoccluded_checker(point_now, dilate_occ);
						if (occ_checker == 0) {
							PATCH = Patch(Point, point_now, image.imgU, image.imgT, dilate_occ, sm, image.LEVEL);
							q = PATCH.costfunction(cost_num);
							PATCH = Patch();

							if (p > q && q > 0) {
								p = q;
								minCost_Point = point_now;
							}
						}
					}
				}
			}
		}
		minCost_Point = minCost_Point - Point;
		sm.put(Point, minCost_Point);
	}
}

/* Reconstruction of U&T */
void Reconstruction(LClass& Img, ShiftMap& Sm, int& cost_number) {
	int c_num = Img.occ_index;

	Patch PATCH;
	int PatchSize2 = PATCHSIZEint * PATCHSIZEint;
	if (cost_number != 1 && cost_number != 3) { cout << "ERROR! cost_num in Reconstruction()" << endl; }

	double shigma;
	vector<double> Cost;
	vector<double> sqrtCost;
	double cost_num, sqrtcost_num;
	Point2i A, B, C, D;
	double costSUM, costSUMsqrt;
	vector<double> colorR, colorG, colorB;
	double Spq, avgColorR, avgColorG, avgColorB;
	vector<double> S_pq;

	Vec3b color;
	uchar r, g, b;
	vector<float> TX, TY;
	float tx, ty;
	int t_index;
	float t_level = pow(2, Img.LEVEL);
	double avgX, avgY;
	int non;
	for (int i = 0; i < c_num; i++) {
		A = Img.occ_p[i];
		if (Img.imgO.data[A.y * Img.XSIZE + A.x] != 0) {
			non = 0;
			for (int y = A.y + PATCHstart; y < A.y + PATCHend; y++) {
				for (int x = A.x + PATCHstart; x < A.x + PATCHend; x++) {
					if (x >= 0 && x < Img.XSIZE && y >= 0 && y < Img.YSIZE) {
						C = Point2i(x, y);
						B.x = C.x + Sm.nnX(C);
						B.y = C.y + Sm.nnY(C);
						PATCH = Patch(C.x, C.y, Img.imgU, Img.imgT, Img.imgO, Sm, Img.LEVEL);
						if (PATCH.costfunction(cost_number) < 0) {
							cost_num = 0, sqrtcost_num = 0;
						}
						else {
							cost_num = PATCH.costfunction(cost_number);
							sqrtcost_num = PATCH.costfunction(cost_number - 1);
						}

						D.x = A.x + Sm.nnX(C);
						D.y = A.y + Sm.nnY(C);
						if (D.x >= 0 && D.x < Img.XSIZE && D.y >= 0 && D.y < Img.YSIZE) {
							color = Img.imgU.at<Vec3b>(D.y, D.x);	// ピクセル値（カラー）を取得
							r = color[2];	// R,G,B値に分解
							g = color[1];
							b = color[0];
							colorR.push_back((double)r);
							colorG.push_back((double)g);
							colorB.push_back((double)b);
							t_index = D.y * Img.XSIZE * 2 + D.x * 2;	// テクスチャ特徴を取得
							tx = (float)Img.imgT.data[t_index] * t_level;
							ty = (float)Img.imgT.data[t_index + 1] * t_level;
							TX.push_back(tx);
							TY.push_back(ty);

							Cost.push_back(cost_num);
							sqrtCost.push_back(sqrtcost_num);
						}
						else { non++; }
						PATCH = Patch();
					}
					else { non++; }
				}
			}

			// sort Cost and return 75 persentile
			shigma = SHIGMA(sqrtCost);
			Spq = 0.0;
			if (shigma > 0) {
				double num_temp;
				for (int ii = 0; ii < (PatchSize2 - non); ii++) {
					num_temp = (double)Cost[ii] / (double)(2 * shigma * shigma);
					num_temp = (double)exp(-num_temp);
					S_pq.push_back(num_temp);
					Spq = Spq + num_temp;
				}
			}
			
			if (Spq > 0) {
				// reconstruct u(p)
				avgColorR = 0.0, avgColorG = 0.0, avgColorB = 0.0;
				for (int ii = 0; ii < (PatchSize2 - non); ii++) {
					avgColorR = avgColorR + S_pq[ii] * colorR[ii];
					avgColorG = avgColorG + S_pq[ii] * colorG[ii];
					avgColorB = avgColorB + S_pq[ii] * colorB[ii];
				}
				r = (uchar)(avgColorR / Spq);	// R,G,B値を処理
				g = (uchar)(avgColorG / Spq);
				b = (uchar)(avgColorB / Spq);
				color = Vec3b(b, g, r);				// R,G,B値からピクセル値（カラー）を生成
				Img.imgU.at<Vec3b>(A.y, A.x) = color;	// ピクセル値（カラー）を設定

				// reconstruct t(p)
				avgX = 0.0, avgY = 0.0;
				for (int ii = 0; ii < (PatchSize2 - non); ii++) {
					avgX = avgX + S_pq[ii] * TX[ii];
					avgY = avgY + S_pq[ii] * TY[ii];
				}
				tx = (float)(avgX / Spq);
				ty = (float)(avgY / Spq);
				tx = (float)(tx / t_level);
				ty = (float)(ty / t_level);
				t_index = A.y * Img.XSIZE * 2 + A.x * 2;
				Img.imgT.data[t_index] = (uchar)tx;
				Img.imgT.data[t_index + 1] = (uchar)ty;
			}
			else if (Spq == 0) {
				B.x = A.x + Sm.nnX(A);
				B.y = A.y + Sm.nnY(A);
				if (B.x >= 0 && B.x < Img.XSIZE && B.y >= 0 && B.y < Img.YSIZE) {
					color = Img.imgU.at<Vec3b>(B.y, B.x);		// ピクセル値（カラー）を取得
					t_index = B.y * Img.XSIZE * 2 + B.x * 2;	// テクスチャ特徴を取得
					tx = (float)Img.imgT.data[t_index];
					ty = (float)Img.imgT.data[t_index + 1];

					Img.imgU.at<Vec3b>(A.y, A.x) = color;
					t_index = A.y * Img.XSIZE * 2 + A.x * 2;
					Img.imgT.data[t_index] = (uchar)tx;
					Img.imgT.data[t_index + 1] = (uchar)ty;
				}
				else { cout << "    Spq=0  ; " << A << " + " << Sm.nn(A) << " = " << B << endl; }
				cout << "ERROR! : Reconstruction : " << A << " , Spq = 0" << endl; 
			}
			else { cout << "ERROR! : Reconstruction : " << A << " , Spq = " << Spq << endl; }

			Cost.clear();
			sqrtCost.clear();
			colorR.clear();
			colorG.clear();
			colorB.clear();
			TX.clear();
			TY.clear();
			S_pq.clear();
		}
	}
}
/* Reconstruction of U&T at first */
void Reconstruction_first(LClass& Img, ShiftMap& Sm, vector<Point2i>& reconst_p, Mat& currentOCC, int& cost_number) {
	int PSIZEint = PATCHSIZEint;
	int Pstart = PATCHstart;
	int Pend = PATCHend;

	int c_num = reconst_p.size();
	vector<int> occNUMx;
	vector<int> occNUMy;
	for (int i = 0; i < c_num; i++) {
		occNUMx.push_back(reconst_p[i].x);
		occNUMy.push_back(reconst_p[i].y);
	}

	Patch PATCH;
	int PatchSize2 = PSIZEint * PSIZEint;
	if (cost_number != 1 && cost_number != 3) { cout << "ERROR! Patch size in Reconstruction_first()" << endl; }

	double shigma;
	vector<double> Cost;
	vector<double> sqrtCost;
	double cost_num, sqrtcost_num;
	Point2i A, B, C, D;
	double costSUM, costSUMsqrt;
	vector<double> colorR, colorG, colorB;
	double Spq, avgColorR, avgColorG, avgColorB;
	vector<double> S_pq;

	Vec3b color;
	uchar r, g, b;
	vector<float> TX, TY;
	float tx, ty;
	int t_index;
	float t_level = pow(2, Img.LEVEL);
	double avgX, avgY;
	int non;
	for (int i = 0; i < c_num; i++) {
		A.x = occNUMx[i];
		A.y = occNUMy[i];
		if (Img.imgO.data[A.y * Img.XSIZE + A.x] != 0) {
			non = 0;
			for (int y = A.y + Pstart; y < A.y + Pend; y++) {
				for (int x = A.x + Pstart; x < A.x + Pend; x++) {
					if (x >= 0 && x < Img.XSIZE && y >= 0 && y < Img.YSIZE) {
						C = Point2i(x, y);
						B.x = C.x + Sm.nnX(C);
						B.y = C.y + Sm.nnY(C);
						PATCH = Patch(C.x, C.y, Img.imgU, Img.imgT, currentOCC, Sm, Img.LEVEL);
						if (PATCH.costfunction(cost_number) < 0) {
							cost_num = 0.0, sqrtcost_num = 0.0;
						}
						else {
							cost_num = PATCH.costfunction(cost_number);
							sqrtcost_num = PATCH.costfunction(cost_number - 1);
						}


						D.x = A.x + Sm.nnX(C);
						D.y = A.y + Sm.nnY(C);
						if (D.x >= 0 && D.x < Img.XSIZE && D.y >= 0 && D.y < Img.YSIZE && cost_num >= 0) {
							if (currentOCC.data[D.y * Img.XSIZE + D.x] == 0) {
								color = Img.imgU.at<Vec3b>(D.y, D.x);	// ピクセル値（カラー）を取得
								r = color[2];	// R,G,B値に分解
								g = color[1];
								b = color[0];
								colorR.push_back((double)r);
								colorG.push_back((double)g);
								colorB.push_back((double)b);
								t_index = D.y * Img.XSIZE * 2 + D.x * 2;	// テクスチャ特徴を取得
								tx = (float)Img.imgT.data[t_index] * t_level;
								ty = (float)Img.imgT.data[t_index + 1] * t_level;
								TX.push_back(tx);
								TY.push_back(ty);

								Cost.push_back(cost_num);
								sqrtCost.push_back(sqrtcost_num);
							}
							else { non++; }
						}
						else { non++; }
						PATCH = Patch();
					}
					else { non++; }
				}
			}

			// sort Cost and return 75 persentile
			shigma = SHIGMA(sqrtCost);
			Spq = 0.0;
			if (shigma > 0) {
				double num_temp;
				for (int ii = 0; ii < (PatchSize2 - non); ii++) {
					num_temp = (double)Cost[ii] / (double)(2 * shigma * shigma);
					num_temp = (double)exp(-num_temp);
					S_pq.push_back(num_temp);
					Spq = Spq + num_temp;
				}
			}
			
			if (Spq > 0) {
				// reconstruct u(p)
				avgColorR = 0.0, avgColorG = 0.0, avgColorB = 0.0;
				for (int ii = 0; ii < (PatchSize2 - non); ii++) {
					avgColorR = avgColorR + S_pq[ii] * colorR[ii];
					avgColorG = avgColorG + S_pq[ii] * colorG[ii];
					avgColorB = avgColorB + S_pq[ii] * colorB[ii];
				}
				r = (uchar)(avgColorR / Spq);	// R,G,B値を処理
				g = (uchar)(avgColorG / Spq);
				b = (uchar)(avgColorB / Spq);
				color = Vec3b(b, g, r);				// R,G,B値からピクセル値（カラー）を生成
				Img.imgU.at<Vec3b>(A.y, A.x) = color;	// ピクセル値（カラー）を設定

				// reconstruct t(p)
				avgX = 0.0, avgY = 0.0;
				for (int ii = 0; ii < (PatchSize2 - non); ii++) {
					avgX = avgX + S_pq[ii] * TX[ii];
					avgY = avgY + S_pq[ii] * TY[ii];
				}
				tx = (float)(avgX / Spq);
				ty = (float)(avgY / Spq);
				tx = (float)(tx / t_level);
				ty = (float)(ty / t_level);
				t_index = A.y * Img.XSIZE * 2 + A.x * 2;
				Img.imgT.data[t_index] = (uchar)tx;
				Img.imgT.data[t_index + 1] = (uchar)ty;
			}
			else if (Spq == 0) {
				B.x = A.x + Sm.nnX(A);
				B.y = A.y + Sm.nnY(A);
				if (B.x >= 0 && B.x < Img.XSIZE && B.y >= 0 && B.y < Img.YSIZE) {
					color = Img.imgU.at<Vec3b>(B.y, B.x);	// ピクセル値（カラー）を取得
					t_index = B.y * Img.XSIZE * 2 + B.x * 2;	// テクスチャ特徴を取得
					tx = (float)Img.imgT.data[t_index];
					ty = (float)Img.imgT.data[t_index + 1];

					Img.imgU.at<Vec3b>(A.y, A.x) = color;
					t_index = A.y * Img.XSIZE * 2 + A.x * 2;
					Img.imgT.data[t_index] = (uchar)tx;
					Img.imgT.data[t_index + 1] = (uchar)ty;
				}
				cout << "ERROR! : Reconstruction_first : " << A << " , Spq = 0" << endl;
			}
			else {
				cout << "ERROR! Initialisation : Reconstruction_first : " << A << " , Spq = " << Spq << endl;
				cout << "                               -> Shift map is " << B << endl;
			}

			Cost.clear();
			sqrtCost.clear();
			colorR.clear();
			colorG.clear();
			colorB.clear();
			TX.clear();
			TY.clear();
			S_pq.clear();
		}
	}
	occNUMx.clear();
	occNUMy.clear();
}

/* sort & return 75th percentile */
double SHIGMA(vector<double>& array) {
	/* sort Cost */
	int c_num = 0;
	vector<double> CostUP(array.size());	// 0含む
	copy(array.begin(), array.end(), CostUP.begin());
	c_num = array.size();
	sort(CostUP.begin(), CostUP.end());

	/* search 75th percentile */
	double Q3 = 100000.0;
	int Quartile;
	for (int i = 0; i < c_num; i++) {
		if (c_num % 2 == 0) {
			int half = c_num / 2;
			if (half % 2 == 0) {
				Quartile = (half - 1) + half / 2;
				Q3 = (double)(CostUP[Quartile] + CostUP[Quartile + 1]) / 2.0;
			}
			else {
				Quartile = (half - 1) + (half + 1) / 2;
				Q3 = (double)CostUP[Quartile];
			}
		}
		else {
			int half = (c_num - 1) / 2;
			if (half % 2 == 0) {
				Quartile = (half - 1) + half / 2;
				Q3 = (double)(CostUP[Quartile] + CostUP[Quartile + 1]) / 2.0;
			}
			else {
				Quartile = (half - 1) + (half + 1) / 2;
				Q3 = (double)CostUP[Quartile];
			}
		}
	}

	CostUP.clear();
	return Q3;
}

/* Reconstruction of U at first by ShiftMap */
void firstReconstruction_SM(LClass& Img, ShiftMap& Sm) {
	Vec3b color;
	int TX, TY;
	int t_index, TXcode, TYcode;
	Point2i currentPoint, ANNpoint;
	for (int i = 0; i < Img.occ_index; i++) {
		currentPoint = Img.occ_p[i];
		ANNpoint = Point2i(currentPoint.x + Sm.nnX(currentPoint), currentPoint.y + Sm.nnY(currentPoint));
		while (ANNpoint.x < 0 || ANNpoint.x >= Img.XSIZE || ANNpoint.y < 0 || ANNpoint.y >= Img.YSIZE) {
			if (ANNpoint.x < 0) {
				ANNpoint.x++;
			}
			if (ANNpoint.x >= Img.XSIZE) {
				ANNpoint.x--;
			}
			if (ANNpoint.y < 0) {
				ANNpoint.y++;
			}
			if (ANNpoint.y >= Img.YSIZE) {
				ANNpoint.y--;
			}
		}
		color = Img.imgU.at<Vec3b>(ANNpoint.y, ANNpoint.x);			// 色値
		Img.imgU.at<Vec3b>(currentPoint.y, currentPoint.x) = color;
		t_index = ANNpoint.y * Img.XSIZE * 2 + ANNpoint.x * 2;		// テクスチャ値
		TX = (int)Img.imgT.data[t_index];
		TY = (int)Img.imgT.data[t_index + 1];
		t_index = currentPoint.y * Img.XSIZE * 2 + currentPoint.x * 2;
		Img.imgT.data[t_index] = (uchar)TX;
		Img.imgT.data[t_index + 1] = (uchar)TY;
	}
}
/* Reconstruction at first by color */
void firstReconstruction_COLOR(Mat UP_img, LClass& Img)
{
	Mat UPimg;
	pyrUp(UP_img, UPimg);

	Vec3b color;
	int TX, TY;
	int t_index, TXcode, TYcode;
	Point2i currentPoint;
	for (int i = 0; i < Img.occ_index; i++) {
		currentPoint = Img.occ_p[i];
		color = UPimg.at<Vec3b>(currentPoint.y, currentPoint.x);		// 色値
		Img.imgU.at<Vec3b>(currentPoint.y, currentPoint.x) = color;
		t_index = currentPoint.y * Img.XSIZE * 2 + currentPoint.x * 2;	// テクスチャ値
		TX = (int)UPimg.data[t_index];
		TY = (int)UPimg.data[t_index + 1];
		Img.imgT.data[t_index] = (uchar)TX;
		Img.imgT.data[t_index + 1] = (uchar)TY;
	}
}

/* アニーリング（焼きなまし） */
void annealingReconstruction(LClass& Img, ShiftMap& Sm) {
	int c_num = Img.occ_index;

	Patch PATCH;
	double COST, COST2, minCOST;
	double Cost, Cost2, minCost;
	Point2i A, B, C, Pix, minPix;
	Vec3b color;
	int TX, TY;
	int t_index, TXcode, TYcode;
	for (int i = 0; i < c_num; i++) {
		A = Img.occ_p[i];
		minCOST = DBL_MAX; // 最大値を代入
		C.x = A.x + Sm.nnX(A);
		C.y = A.y + Sm.nnY(A);
		minPix = C;
		for (int y = A.y + PATCHstart; y < A.y + PATCHend; y++) {
			for (int x = A.x + PATCHstart; x < A.x + PATCHend; x++) {
				if (x >= 0 && x < Img.XSIZE && y >= 0 && y < Img.YSIZE) {
					B = Point2i(x, y);
					B = B + Sm.nn(B);
					PATCH = Patch(C, B, Img.imgU, Img.imgT, Img.imgO, Sm, Img.LEVEL);
					COST = PATCH.costfunction(3);
					COST2 = PATCH.costfunction(2);
					if (minCOST > COST && COST > 0) {
						Pix = Point2i(A.x + Sm.nnX(B), A.y + Sm.nnY(B));
						if (Pix.x >= 0 && Pix.x < Img.XSIZE && Pix.y >= 0 && Pix.y < Img.YSIZE) {
							minCOST = COST;
							minPix = Pix;
							color = Img.imgU.at<Vec3b>(Pix.y, Pix.x);		// ピクセル値（カラー）を取得
							t_index = Pix.y * Img.XSIZE * 2 + Pix.x * 2;	// テクスチャ特徴を取得
							TX = (int)Img.imgT.data[t_index];
							TY = (int)Img.imgT.data[t_index + 1];
						}
						/*else { cout << " ERROR! Annealing in " << Pix << endl; }*/
					}
				}
				/*else { cout << "B does not include " << Point2i(x, y) << endl; }*/
			}
		}
		
		if (minPix != C) {
			Img.imgU.at<Vec3b>(A.y, A.x) = color;		// ピクセル値（カラー）を設定
			t_index = A.y * Img.XSIZE * 2 + A.x * 2;	// テクスチャを設定
			Img.imgT.data[t_index] = (uchar)TX;
			Img.imgT.data[t_index + 1] = (uchar)TY;
		}
	}
}

/* 補修後ピラミッド更新 */
void update_pyramid(ImagePyramid& nowImagePyramid, LClass& newLClass) {
	nowImagePyramid.U[newLClass.LEVEL] = newLClass.imgU.clone();
	nowImagePyramid.T[newLClass.LEVEL] = newLClass.imgT.clone();
}


/* 補修領域におけるマルコフ確率場による確率的画像処理 */
void OCC_MRF_GaussSeidel(Mat& Image_DST, ImagePyramid& ImgPyr, double& alpha_num) {
	/* パラメータ設定 */
	vector<double> ALPHA;
	vector<double> SIGMA;
	for (int i_pyr = 0; i_pyr <= L; i_pyr++) {
		ALPHA.push_back(alpha_num);
		SIGMA.push_back(Sigma);
	}
	//cout << " ALPHA = " << Alpha << ", SIGMA  = " << Sigma << endl;

	double errorConvergence;
	double number[3];
	double denom, ave[3];
	double M_number[3];	// オクルージョン境界部
	double D_number[3];	// オクルージョン内部
	double Yi[3];

	Vec3b color;
	uchar r, g, b;
	vector<Mat> RandomMap_R, RandomMap_G, RandomMap_B;
	vector<Mat> RandomMap_temp;
	Mat Map_R, Map_G, Map_B;
	for (int i_pyr = 0; i_pyr <= L; i_pyr++) {
		Map_R = Mat(ImgPyr.U[i_pyr].rows, ImgPyr.U[i_pyr].cols, CV_64FC3, Scalar::all(0.5));
		Map_G = Mat(ImgPyr.U[i_pyr].rows, ImgPyr.U[i_pyr].cols, CV_64FC3, Scalar::all(0.5));
		Map_B = Mat(ImgPyr.U[i_pyr].rows, ImgPyr.U[i_pyr].cols, CV_64FC3, Scalar::all(0.5));
		RandomMap_R.push_back(Map_B.clone());
		RandomMap_G.push_back(Map_G.clone());
		RandomMap_B.push_back(Map_R.clone());
	}
	Mat Image_dst;
	Image_DST.copyTo(Image_dst);

	// オクルージョンピクセルの抽出
	int occ_number, occ_number_before = 0;
	vector<int> up;
	vector<int> down;
	vector<int> left;
	vector<int> right;
	vector<int> pyr_up;
	vector<int> pyr_down;
	vector<int> pyr_down2;
	vector<int> pyr_down3;
	vector<int> pyr_down4;
	int index, X_index, Y_index;
	//int index_sum = 0;
	int h_x, h_y, h_index, h_Level;
	for (int int_pyr = 0; int_pyr <= L; int_pyr++) {
		occ_number = ImgPyr.occ_pix_index[int_pyr];

		for (int i = 0; i < occ_number; i++) {
			int index_now = occ_number_before + i;
			X_index = ImgPyr.occ_pix[index_now].x;
			Y_index = ImgPyr.occ_pix[index_now].y;
			index = Y_index * ImgPyr.O[int_pyr].cols + X_index;
			//cout << int_pyr << " " << index_now << " : " << X_index << " " << Y_index << " " << index << endl;	// 確認用
			// 隣接データは「データ外 =0」か「境界部 M=1」か「内部 D=2」に含まれる
			if (X_index < 1) { left.push_back(0); }				// left
			else if (ImgPyr.O[int_pyr].data[index - 1] == 0) { left.push_back(1); }
			else { left.push_back(2); }
			if (Y_index < 1) { up.push_back(0); }					// up
			else if (ImgPyr.O[int_pyr].data[index - ImgPyr.U[int_pyr].cols] == 0) { up.push_back(1); }
			else { up.push_back(2); }
			if (X_index >= ImgPyr.U[int_pyr].cols - 1) { right.push_back(0); }	// right
			else if (ImgPyr.O[int_pyr].data[index + 1] == 0) { right.push_back(1); }
			else { right.push_back(2); }
			if (Y_index >= ImgPyr.U[int_pyr].rows - 1) { down.push_back(0); }	//down
			else if (ImgPyr.O[int_pyr].data[index + ImgPyr.U[int_pyr].cols] == 0) { down.push_back(1); }
			else { down.push_back(2); }

			h_Level = int_pyr + 1;			//pyramid up
			if (h_Level > L) { pyr_up.push_back(0); }
			else {
				if (X_index % 2 == 0) { h_x = X_index / 2; }
				else { h_x = (X_index - 1) / 2; }
				if (Y_index % 2 == 0) { h_y = Y_index / 2; }
				else { h_y = (Y_index - 1) / 2; }
				h_index = h_y * ImgPyr.O[h_Level].cols + h_x;
				if (h_x >= 0 && h_x < ImgPyr.O[h_Level].cols && h_y >= 0 && h_y < ImgPyr.O[h_Level].rows) {
					if (ImgPyr.O[h_Level].data[h_index] != 0) { pyr_up.push_back(2); }
					else { pyr_up.push_back(1); }
				}
				else{ pyr_up.push_back(0); }
			}
			h_Level = int_pyr - 1;			//pyramid down
			if (h_Level < 0) {
				pyr_down.push_back(0);
				pyr_down2.push_back(0);
				pyr_down3.push_back(0);
				pyr_down4.push_back(0);
			}
			else {
				h_x = X_index * 2;	//対応する４マスに対応付け
				h_y = Y_index * 2;
				if (h_x >= 0 && h_x < ImgPyr.O[h_Level].cols && h_y >= 0 && h_y < ImgPyr.O[h_Level].rows) {
					h_index = h_y * ImgPyr.O[h_Level].cols + h_x;
					if((h_x - 1) < 0 || (h_x - 1) >= ImgPyr.O[h_Level].cols || (h_y - 1) < 0 || (h_y - 1) >= ImgPyr.O[h_Level].rows){ pyr_down.push_back(0); }
					else if (ImgPyr.O[h_Level].data[h_index - 1 - ImgPyr.O[h_Level].cols] != 0) { pyr_down.push_back(2); }
					else { pyr_down.push_back(1); }
					if (h_x < 0 || h_x >= ImgPyr.O[h_Level].cols || (h_y - 1) < 0 || (h_y - 1) >= ImgPyr.O[h_Level].rows) { pyr_down2.push_back(0); }
					else if (ImgPyr.O[h_Level].data[h_index - ImgPyr.O[h_Level].cols] != 0) { pyr_down2.push_back(2); }
					else { pyr_down2.push_back(1); }
					if ((h_x - 1) < 0 || (h_x - 1) >= ImgPyr.O[h_Level].cols || h_y < 0 || h_y >= ImgPyr.O[h_Level].rows) { pyr_down3.push_back(0); }
					else if (ImgPyr.O[h_Level].data[h_index - 1] != 0) { pyr_down3.push_back(2); }
					else { pyr_down3.push_back(1); }
					if (h_x < 0 || h_x >= ImgPyr.O[h_Level].cols || h_y < 0 || h_y >= ImgPyr.O[h_Level].rows) { pyr_down4.push_back(0); }
					else if (ImgPyr.O[h_Level].data[h_index] != 0) { pyr_down4.push_back(2); }
					else { pyr_down4.push_back(1); }
				}
				else { 
					pyr_down.push_back(0);
					pyr_down2.push_back(0);
					pyr_down3.push_back(0);
					pyr_down4.push_back(0);
				}
			}
		}

		occ_number_before += ImgPyr.occ_pix_index[int_pyr];
	}

	// RGB値からxを決める
	int pix_X, pix_Y;

	// ノイズ除去
	double Sigma_index, Alpha_index;
	int occ_index;
	for (int index_R = 0; index_R < Repeat; index_R++) {
		errorConvergence = 0;
		occ_index = 0;
		occ_number = 0;
		for (int int_pyr = 0; int_pyr <= L; int_pyr++) {
			Alpha_index = ALPHA[int_pyr];
			Sigma_index = (double)SIGMA[int_pyr] * SIGMA[int_pyr];
			
			RandomMap_temp.push_back(RandomMap_B[int_pyr]);
			RandomMap_temp.push_back(RandomMap_G[int_pyr]);
			RandomMap_temp.push_back(RandomMap_R[int_pyr]);
			occ_number += ImgPyr.occ_pix_index[int_pyr];

			for (int Y_index = 0; Y_index < ImgPyr.U[int_pyr].rows; Y_index++) {
				for (int X_index = 0; X_index < ImgPyr.U[int_pyr].cols; X_index++) {
					pix_X = X_index;
					pix_Y = Y_index;

					color = ImgPyr.U[int_pyr].at<Vec3b>(pix_Y, pix_X);	// ピクセル値（カラー）を取得
					r = color[2];	// R,G,B値に分解
					g = color[1];
					b = color[0];
					Yi[2] = (double)r / (double)MAX_INTENSE;
					Yi[1] = (double)g / (double)MAX_INTENSE;
					Yi[0] = (double)b / (double)MAX_INTENSE;

					index = pix_Y * ImgPyr.U[int_pyr].cols + pix_X;

					for (int color_index = 0; color_index < 3; color_index++) {
						number[color_index] = (double)Yi[color_index] / Sigma_index;
					}
					denom = Rambda + (double)(1.0 / Sigma_index);

					if (ImgPyr.O[int_pyr].data[index] != 0) {
						for (int color_index = 0; color_index < 3; color_index++) {
							M_number[color_index] = 0;
							D_number[color_index] = 0;
						}
						if (left[occ_index] != 0) {		// left
							if (left[occ_index] == 1) {
								M_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X - 1);
								M_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X - 1);
								M_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X - 1);
							}
							else {
								D_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X - 1);
								D_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X - 1);
								D_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X - 1);
							}
							denom += Alpha_index;
						}
						if (right[occ_index] != 0) {		// right
							if (right[occ_index] == 1) {
								M_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X + 1);
								M_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X + 1);
								M_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X + 1);
							}
							else {
								D_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X + 1);
								D_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X + 1);
								D_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X + 1);
							}
							denom += Alpha_index;
						}
						if (up[occ_index] != 0) {		// up
							if (up[occ_index] == 1) {
								M_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y - 1, pix_X);
								M_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y - 1, pix_X);
								M_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y - 1, pix_X);
							}
							else {
								D_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y - 1, pix_X);
								D_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y - 1, pix_X);
								D_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y - 1, pix_X);
							}
							denom += Alpha_index;
						}
						if (down[occ_index] != 0) {		// down
							if (down[occ_index] == 1) {
								M_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y + 1, pix_X);
								M_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y + 1, pix_X);
								M_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y + 1, pix_X);
							}
							else {
								D_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y + 1, pix_X);
								D_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y + 1, pix_X);
								D_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y + 1, pix_X);
							}
							denom += Alpha_index;
						}

						if (pyr_up[occ_index] != 0) {		//pyramid up
							h_Level = int_pyr + 1;
							if (pix_X % 2 == 0) { h_x = pix_X / 2; }
							else { h_x = (pix_X - 1) / 2; }
							if (pix_Y % 2 == 0) { h_y = pix_Y / 2; }
							else { h_y = (pix_Y - 1) / 2; }
							
							if (pyr_up[occ_index] == 1) {
								M_number[2] += (double)RandomMap_R[h_Level].at<double>(h_y, h_x);
								M_number[1] += (double)RandomMap_G[h_Level].at<double>(h_y, h_x);
								M_number[0] += (double)RandomMap_B[h_Level].at<double>(h_y, h_x);
							}
							else {
								D_number[2] += (double)RandomMap_R[h_Level].at<double>(h_y, h_x);
								D_number[1] += (double)RandomMap_G[h_Level].at<double>(h_y, h_x);
								D_number[0] += (double)RandomMap_B[h_Level].at<double>(h_y, h_x);
							}
							denom += Alpha_index;
						}
						if (pyr_down[occ_index] != 0) {		//pyramid down
							h_Level = int_pyr - 1;
							h_x = pix_X * 2;
							h_y = pix_Y * 2;

							if (pyr_down[occ_index] == 1) {
								M_number[2] += (double)RandomMap_R[h_Level].at<double>(h_y - 1, h_x - 1);
								M_number[1] += (double)RandomMap_G[h_Level].at<double>(h_y - 1, h_x - 1);
								M_number[0] += (double)RandomMap_B[h_Level].at<double>(h_y - 1, h_x - 1);
							}
							else if (pyr_down[occ_index] == 2) {
								D_number[2] += (double)RandomMap_R[h_Level].at<double>(h_y - 1, h_x - 1);
								D_number[1] += (double)RandomMap_G[h_Level].at<double>(h_y - 1, h_x - 1);
								D_number[0] += (double)RandomMap_B[h_Level].at<double>(h_y - 1, h_x - 1);
							}
							if (pyr_down2[occ_index] == 1) {
								M_number[2] += (double)RandomMap_R[h_Level].at<double>(h_y - 1, h_x);
								M_number[1] += (double)RandomMap_G[h_Level].at<double>(h_y - 1, h_x);
								M_number[0] += (double)RandomMap_B[h_Level].at<double>(h_y - 1, h_x);
							}
							else if (pyr_down2[occ_index] == 2) {
								D_number[2] += (double)RandomMap_R[h_Level].at<double>(h_y - 1, h_x);
								D_number[1] += (double)RandomMap_G[h_Level].at<double>(h_y - 1, h_x);
								D_number[0] += (double)RandomMap_B[h_Level].at<double>(h_y - 1, h_x);
							}
							if (pyr_down3[occ_index] == 1) {
								M_number[2] += (double)RandomMap_R[h_Level].at<double>(h_y, h_x - 1);
								M_number[1] += (double)RandomMap_G[h_Level].at<double>(h_y, h_x - 1);
								M_number[0] += (double)RandomMap_B[h_Level].at<double>(h_y, h_x - 1);
							}
							else if (pyr_down3[occ_index] == 2) {
								D_number[2] += (double)RandomMap_R[h_Level].at<double>(h_y, h_x - 1);
								D_number[1] += (double)RandomMap_G[h_Level].at<double>(h_y, h_x - 1);
								D_number[0] += (double)RandomMap_B[h_Level].at<double>(h_y, h_x - 1);
							}
							if (pyr_down4[occ_index] == 1) {
								M_number[2] += (double)RandomMap_R[h_Level].at<double>(h_y, h_x);
								M_number[1] += (double)RandomMap_G[h_Level].at<double>(h_y, h_x);
								M_number[0] += (double)RandomMap_B[h_Level].at<double>(h_y, h_x);
							}
							else if (pyr_down4[occ_index] == 2) {
								D_number[2] += (double)RandomMap_R[h_Level].at<double>(h_y, h_x);
								D_number[1] += (double)RandomMap_G[h_Level].at<double>(h_y, h_x);
								D_number[0] += (double)RandomMap_B[h_Level].at<double>(h_y, h_x);
							}
							denom += (double)(Alpha_index * 4);
						}

						for (int color_index = 0; color_index < 3; color_index++) {
							number[color_index] += Alpha_index * (double)(M_number[color_index] + D_number[color_index]);
						}

						occ_index++;
						if (occ_index > occ_number) { cout << "WARNING! : occ_index > occ_number in " << int_pyr << endl; }
					}

					for (int color_index = 0; color_index < 3; color_index++) {
						ave[color_index] = (double)number[color_index] / (double)denom;
						if (color_index == 0) {
							errorConvergence += fabs(RandomMap_B[int_pyr].at<double>(pix_Y, pix_X) - ave[color_index]);
						}
						else if (color_index == 1) {
							errorConvergence += fabs(RandomMap_G[int_pyr].at<double>(pix_Y, pix_X) - ave[color_index]);
						}
						else if (color_index == 2) {
							errorConvergence += fabs(RandomMap_R[int_pyr].at<double>(pix_Y, pix_X) - ave[color_index]);
						}

						int R_temp_index = int_pyr * 3 + color_index;
						RandomMap_temp[R_temp_index].at<double>(pix_Y, pix_X) = (double)ave[color_index];
					}
				}
			}
			errorConvergence = (double)(errorConvergence / 3);
		}

		int temp_index = 0;
		for (int int_pyr = 0; int_pyr /*< 1*/ <= L; int_pyr++) {
			RandomMap_temp[temp_index].copyTo(RandomMap_B[int_pyr].clone());
			RandomMap_temp[temp_index + 1].copyTo(RandomMap_G[int_pyr].clone());
			RandomMap_temp[temp_index + 2].copyTo(RandomMap_R[int_pyr].clone());
			temp_index = temp_index + 3;
		}
		RandomMap_temp.clear();

		if (errorConvergence / MAX_DATA < Converge) {
			cout << "収束成功: errorConvergence = " << errorConvergence << " , Iteration " << index_R + 1 << endl;
			break;
		}
		/*else {
			cout << "収束失敗!: errorConvergence = " << errorConvergence << " , Iteration " << index_R << endl;
		}*/
	}

	/* 画像補修 */
	int occ_X, occ_Y;
	occ_number = ImgPyr.occ_pix_index[0];
	for (int i = 0; i < occ_number; i++) {
		occ_X = ImgPyr.occ_pix[i].x;
		occ_Y = ImgPyr.occ_pix[i].y;

		ave[0] = (int)((double)RandomMap_B[0].at<double>(occ_Y, occ_X) * (double)MAX_INTENSE);
		ave[1] = (int)((double)RandomMap_G[0].at<double>(occ_Y, occ_X) * (double)MAX_INTENSE);
		ave[2] = (int)((double)RandomMap_R[0].at<double>(occ_Y, occ_X) * (double)MAX_INTENSE);
		for (int color_index = 0; color_index < 3; color_index++) {
			if (ave[color_index] < 0) {
				ave[color_index] = 0;
				cout << "WARNING! : under0" << Point2i(occ_X, occ_Y) << endl;
			}
			else if (ave[color_index] > 255) {
				ave[color_index] = 255;
				cout << "WARNING! : over255" << Point2i(occ_X, occ_Y) << endl;
			}
		}

		color[2] = (uchar)ave[2];	// R,G,B値に分解
		color[1] = (uchar)ave[1];
		color[0] = (uchar)ave[0];
		Image_dst.at<Vec3b>(occ_Y, occ_X) = color;	// ピクセル値（カラー）
	}
	Image_dst.copyTo(Image_DST);

	RandomMap_R.clear();
	RandomMap_G.clear();
	RandomMap_B.clear();
	up.clear();
	down.clear();
	left.clear();
	right.clear();
	pyr_up.clear();
	pyr_down.clear();
	pyr_down2.clear();
	pyr_down3.clear();
	pyr_down4.clear();

	ALPHA.clear();
	SIGMA.clear();
}

/* 補修領域におけるマルコフ確率場による確率的画像処理(GammaありMRF) */
void OCC_GMRF_GaussSeidel(Mat& Image_DST, ImagePyramid& ImgPyr, double& alpha2_num, double& gamma_num) {
	/* パラメータ設定 */
	vector<double> ALPHA;
	vector<double> SIGMA;
	vector<double> GAMMA;
	for (int i_pyr = 0; i_pyr <= L; i_pyr++) {
		ALPHA.push_back(alpha2_num);
		SIGMA.push_back(Sigma);
		GAMMA.push_back(gamma_num);
	}
	//cout << " ALPHA = " << Alpha << ", SIGMA  = " << Sigma << ", GAMMA  = " << Gamma << endl;

	double errorConvergence;
	double number[3];
	double denom, ave[3];
	double M_number[3];	// オクルージョン境界部
	double D_number[3];	// オクルージョン内部
	double number2[3];
	double denom2, ave2[3];
	double M_number2[3];	// オクルージョン境界部
	double D_number2[3];
	double Yi[3];

	Vec3b color;
	uchar r, g, b;
	vector<Mat> RandomMap_R, RandomMap_G, RandomMap_B;
	vector<Mat> RandomMap_temp;
	vector<Mat> RandomMap_R2, RandomMap_G2, RandomMap_B2;
	vector<Mat> RandomMap_temp2;
	Mat Map_R, Map_G, Map_B;
	for (int i_pyr = 0; i_pyr <= L; i_pyr++) {
		Map_R = Mat(ImgPyr.U[i_pyr].rows, ImgPyr.U[i_pyr].cols, CV_64FC3, Scalar::all(0));
		Map_G = Mat(ImgPyr.U[i_pyr].rows, ImgPyr.U[i_pyr].cols, CV_64FC3, Scalar::all(0));
		Map_B = Mat(ImgPyr.U[i_pyr].rows, ImgPyr.U[i_pyr].cols, CV_64FC3, Scalar::all(0));
		RandomMap_R.push_back(Map_B.clone());
		RandomMap_G.push_back(Map_G.clone());
		RandomMap_B.push_back(Map_R.clone());
		RandomMap_R2.push_back(Map_B.clone());
		RandomMap_G2.push_back(Map_G.clone());
		RandomMap_B2.push_back(Map_R.clone());
	}
	Mat Image_dst;
	Image_DST.copyTo(Image_dst);

	// オクルージョンピクセルの抽出
	int occ_number, occ_number_before = 0;
	vector<int> up;
	vector<int> down;
	vector<int> left;
	vector<int> right;
	vector<int> pyr_up;
	vector<int> pyr_down;
	vector<int> pyr_down2;
	vector<int> pyr_down3;
	vector<int> pyr_down4;
	int index, X_index, Y_index;
	int index_sum = 0;
	int h_x, h_y, h_index, h_Level;
	for (int int_pyr = 0; int_pyr <= L; int_pyr++) {
		occ_number = ImgPyr.occ_pix_index[int_pyr];
		index_sum += occ_number;

		for (int i = 0; i < occ_number; i++) {
			X_index = ImgPyr.occ_pix[occ_number_before + i].x;
			Y_index = ImgPyr.occ_pix[occ_number_before + i].y;
			index = Y_index * ImgPyr.O[int_pyr].cols + X_index;

			// 隣接データは「データ外 =0」か「境界部 M=1」か「内部 D=2」に含まれる
			if (X_index < 1) { left.push_back(0); }				// left
			else if (ImgPyr.O[int_pyr].data[index - 1] == 0) { left.push_back(1); }	//※修正必要？
			else { left.push_back(2); }
			if (Y_index < 1) { up.push_back(0); }					// up
			else if (ImgPyr.O[int_pyr].data[index - ImgPyr.U[int_pyr].cols] == 0) { up.push_back(1); }
			else { up.push_back(2); }
			if (X_index >= ImgPyr.U[int_pyr].cols - 1) { right.push_back(0); }	// right
			else if (ImgPyr.O[int_pyr].data[index + 1] == 0) { right.push_back(1); }
			else { right.push_back(2); }
			if (Y_index >= ImgPyr.U[int_pyr].rows - 1) { down.push_back(0); }	//down
			else if (ImgPyr.O[int_pyr].data[index + ImgPyr.U[int_pyr].cols] == 0) { down.push_back(1); }
			else { down.push_back(2); }

			h_Level = int_pyr + 1;			//pyramid up
			if (h_Level > L) { pyr_up.push_back(0); }
			else {
				if (X_index % 2 == 0) { h_x = X_index / 2; }
				else { h_x = (X_index - 1) / 2; }
				if (Y_index % 2 == 0) { h_y = Y_index / 2; }
				else { h_y = (Y_index - 1) / 2; }
				h_index = h_y * ImgPyr.O[h_Level].cols + h_x;
				if (h_x >= 0 && h_x < ImgPyr.O[h_Level].cols && h_y >= 0 && h_y < ImgPyr.O[h_Level].rows) {
					if (ImgPyr.O[h_Level].data[h_index] != 0) { pyr_up.push_back(2); }
					else { pyr_up.push_back(1); }
				}
				else { pyr_up.push_back(0); }
			}
			h_Level = int_pyr - 1;			//pyramid down
			if (h_Level < 0) {
				pyr_down.push_back(0);
				pyr_down2.push_back(0);
				pyr_down3.push_back(0);
				pyr_down4.push_back(0);
			}
			else {
				h_x = X_index * 2;	//対応する４マスに対応付け
				h_y = Y_index * 2;
				if (h_x >= 0 && h_x < ImgPyr.O[h_Level].cols && h_y >= 0 && h_y < ImgPyr.O[h_Level].rows) {
					h_index = h_y * ImgPyr.O[h_Level].cols + h_x;
					if ((h_x - 1) < 0 || (h_x - 1) >= ImgPyr.O[h_Level].cols || (h_y - 1) < 0 || (h_y - 1) >= ImgPyr.O[h_Level].rows) { pyr_down.push_back(0); }
					else if (ImgPyr.O[h_Level].data[h_index - 1 - ImgPyr.O[h_Level].cols] != 0) { pyr_down.push_back(2); }
					else { pyr_down.push_back(1); }
					if (h_x < 0 || h_x >= ImgPyr.O[h_Level].cols || (h_y - 1) < 0 || (h_y - 1) >= ImgPyr.O[h_Level].rows) { pyr_down2.push_back(0); }
					else if (ImgPyr.O[h_Level].data[h_index - ImgPyr.O[h_Level].cols] != 0) { pyr_down2.push_back(2); }
					else { pyr_down2.push_back(1); }
					if ((h_x - 1) < 0 || (h_x - 1) >= ImgPyr.O[h_Level].cols || h_y < 0 || h_y >= ImgPyr.O[h_Level].rows) { pyr_down3.push_back(0); }
					else if (ImgPyr.O[h_Level].data[h_index - 1] != 0) { pyr_down3.push_back(2); }
					else { pyr_down3.push_back(1); }
					if (h_x < 0 || h_x >= ImgPyr.O[h_Level].cols || h_y < 0 || h_y >= ImgPyr.O[h_Level].rows) { pyr_down4.push_back(0); }
					else if (ImgPyr.O[h_Level].data[h_index] != 0) { pyr_down4.push_back(2); }
					else { pyr_down4.push_back(1); }
				}
				else {
					pyr_down.push_back(0);
					pyr_down2.push_back(0);
					pyr_down3.push_back(0);
					pyr_down4.push_back(0);
				}
			}
		}

		occ_number_before += ImgPyr.occ_pix_index[int_pyr];
	}

	// RGB値からxを決める
	int pix_X, pix_Y;

	// ノイズ除去
	double Sigma_index, Alpha_index, Gamma_index;
	int occ_index;
	double link_num;
	for (int index_R = 0; index_R < Repeat; index_R++) {
		errorConvergence = 0;
		occ_index = 0;
		occ_number = 0;
		for (int int_pyr = 0; int_pyr /*< 1*/ <= L; int_pyr++) {
			Alpha_index = ALPHA[int_pyr];
			Sigma_index = (double)SIGMA[int_pyr] * SIGMA[int_pyr];
			Gamma_index = GAMMA[int_pyr];
			
			RandomMap_temp.push_back(RandomMap_B[int_pyr].clone());
			RandomMap_temp.push_back(RandomMap_G[int_pyr].clone());
			RandomMap_temp.push_back(RandomMap_R[int_pyr].clone());
			RandomMap_temp2.push_back(RandomMap_B2[int_pyr].clone());
			RandomMap_temp2.push_back(RandomMap_G2[int_pyr].clone());
			RandomMap_temp2.push_back(RandomMap_R2[int_pyr].clone());
			occ_number += ImgPyr.occ_pix_index[int_pyr];

			for (int Y_index = 0; Y_index < ImgPyr.U[int_pyr].rows; Y_index++) {
				for (int X_index = 0; X_index < ImgPyr.U[int_pyr].cols; X_index++) {
					pix_X = X_index;
					pix_Y = Y_index;

					color = ImgPyr.U[int_pyr].at<Vec3b>(pix_Y, pix_X);	// ピクセル値（カラー）を取得
					r = color[2];	// R,G,B値に分解
					g = color[1];
					b = color[0];
					Yi[2] = ((double)(r * 2.0) / (double)MAX_INTENSE) - 1.0;	// (-1)~1 正規化
					Yi[1] = ((double)(g * 2.0) / (double)MAX_INTENSE) - 1.0;
					Yi[0] = ((double)(b * 2.0) / (double)MAX_INTENSE) - 1.0;

					for (int color_index = 0; color_index < 3; color_index++) {
						number[color_index] = (double)Yi[color_index] / (double)Sigma_index;
						number2[color_index] = 0;
					}
					denom = Rambda + (1 / Sigma_index);
					denom2 = Rambda + Gamma_index;
					link_num = 0;

					index = pix_Y * ImgPyr.U[int_pyr].cols + pix_X;

					if (ImgPyr.O[int_pyr].data[index] != 0) {
						for (int color_index = 0; color_index < 3; color_index++) {
							M_number[color_index] = 0;
							D_number[color_index] = 0;
							M_number2[color_index] = 0;
							D_number2[color_index] = 0;
						}
						if (left[occ_index] != 0) {		// left
							if (left[occ_index] == 1) {
								M_number[2] += Alpha_index * (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X - 1);
								M_number[1] += Alpha_index * (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X - 1);
								M_number[0] += Alpha_index * (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X - 1);
								M_number2[2] += Alpha_index * ((double)RandomMap_R2[int_pyr].at<double>(pix_Y, pix_X - 1) - (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X - 1));
								M_number2[1] += Alpha_index * ((double)RandomMap_G2[int_pyr].at<double>(pix_Y, pix_X - 1) - (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X - 1));
								M_number2[0] += Alpha_index * ((double)RandomMap_B2[int_pyr].at<double>(pix_Y, pix_X - 1) - (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X - 1));
							}
							else {
								D_number[2] += Alpha_index * (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X - 1);
								D_number[1] += Alpha_index * (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X - 1);
								D_number[0] += Alpha_index * (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X - 1);
								D_number2[2] += Alpha_index * ((double)RandomMap_R2[int_pyr].at<double>(pix_Y, pix_X - 1) - (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X - 1));
								D_number2[1] += Alpha_index * ((double)RandomMap_G2[int_pyr].at<double>(pix_Y, pix_X - 1) - (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X - 1));
								D_number2[0] += Alpha_index * ((double)RandomMap_B2[int_pyr].at<double>(pix_Y, pix_X - 1) - (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X - 1));
							}
							denom += Alpha_index;
							denom2 += Alpha_index;
							link_num++;
						}
						if (right[occ_index] != 0) {		// right
							if (right[occ_index] == 1) {
								M_number[2] += Alpha_index * (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X + 1);
								M_number[1] += Alpha_index * (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X + 1);
								M_number[0] += Alpha_index * (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X + 1);
								M_number2[2] += Alpha_index * ((double)RandomMap_R2[int_pyr].at<double>(pix_Y, pix_X + 1) - (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X + 1));
								M_number2[1] += Alpha_index * ((double)RandomMap_G2[int_pyr].at<double>(pix_Y, pix_X + 1) - (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X + 1));
								M_number2[0] += Alpha_index * ((double)RandomMap_B2[int_pyr].at<double>(pix_Y, pix_X + 1) - (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X + 1));
							}
							else {
								D_number[2] += Alpha_index * (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X + 1);
								D_number[1] += Alpha_index * (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X + 1);
								D_number[0] += Alpha_index * (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X + 1);
								D_number2[2] += Alpha_index * ((double)RandomMap_R2[int_pyr].at<double>(pix_Y, pix_X + 1) - (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X + 1));
								D_number2[1] += Alpha_index * ((double)RandomMap_G2[int_pyr].at<double>(pix_Y, pix_X + 1) - (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X + 1));
								D_number2[0] += Alpha_index * ((double)RandomMap_B2[int_pyr].at<double>(pix_Y, pix_X + 1) - (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X + 1));
							}
							denom += Alpha_index;
							denom2 += Alpha_index;
							link_num++;
						}
						if (up[occ_index] != 0) {		// up
							if (up[occ_index] == 1) {
								M_number[2] += Alpha_index * (double)RandomMap_R[int_pyr].at<double>(pix_Y - 1, pix_X);
								M_number[1] += Alpha_index * (double)RandomMap_G[int_pyr].at<double>(pix_Y - 1, pix_X);
								M_number[0] += Alpha_index * (double)RandomMap_B[int_pyr].at<double>(pix_Y - 1, pix_X);
								M_number2[2] += Alpha_index * ((double)RandomMap_R2[int_pyr].at<double>(pix_Y - 1, pix_X) - (double)RandomMap_R[int_pyr].at<double>(pix_Y - 1, pix_X));
								M_number2[1] += Alpha_index * ((double)RandomMap_G2[int_pyr].at<double>(pix_Y - 1, pix_X) - (double)RandomMap_G[int_pyr].at<double>(pix_Y - 1, pix_X));
								M_number2[0] += Alpha_index * ((double)RandomMap_B2[int_pyr].at<double>(pix_Y - 1, pix_X) - (double)RandomMap_B[int_pyr].at<double>(pix_Y - 1, pix_X));
							}
							else {
								D_number[2] += Alpha_index * (double)RandomMap_R[int_pyr].at<double>(pix_Y - 1, pix_X);
								D_number[1] += Alpha_index * (double)RandomMap_G[int_pyr].at<double>(pix_Y - 1, pix_X);
								D_number[0] += Alpha_index * (double)RandomMap_B[int_pyr].at<double>(pix_Y - 1, pix_X);
								D_number2[2] += Alpha_index * ((double)RandomMap_R2[int_pyr].at<double>(pix_Y - 1, pix_X) - (double)RandomMap_R[int_pyr].at<double>(pix_Y - 1, pix_X));
								D_number2[1] += Alpha_index * ((double)RandomMap_G2[int_pyr].at<double>(pix_Y - 1, pix_X) - (double)RandomMap_G[int_pyr].at<double>(pix_Y - 1, pix_X));
								D_number2[0] += Alpha_index * ((double)RandomMap_B2[int_pyr].at<double>(pix_Y - 1, pix_X) - (double)RandomMap_B[int_pyr].at<double>(pix_Y - 1, pix_X));
							}
							denom += Alpha_index;
							denom2 += Alpha_index;
							link_num++;
						}
						if (down[occ_index] != 0) {		// down
							if (down[occ_index] == 1) {
								M_number[2] += Alpha_index * (double)RandomMap_R[int_pyr].at<double>(pix_Y + 1, pix_X);
								M_number[1] += Alpha_index * (double)RandomMap_G[int_pyr].at<double>(pix_Y + 1, pix_X);
								M_number[0] += Alpha_index * (double)RandomMap_B[int_pyr].at<double>(pix_Y + 1, pix_X);
								M_number2[2] += Alpha_index * ((double)RandomMap_R2[int_pyr].at<double>(pix_Y + 1, pix_X) - (double)RandomMap_R[int_pyr].at<double>(pix_Y + 1, pix_X));
								M_number2[1] += Alpha_index * ((double)RandomMap_G2[int_pyr].at<double>(pix_Y + 1, pix_X) - (double)RandomMap_G[int_pyr].at<double>(pix_Y + 1, pix_X));
								M_number2[0] += Alpha_index * ((double)RandomMap_B2[int_pyr].at<double>(pix_Y + 1, pix_X) - (double)RandomMap_B[int_pyr].at<double>(pix_Y + 1, pix_X));
							}
							else {
								D_number[2] += Alpha_index * (double)RandomMap_R[int_pyr].at<double>(pix_Y + 1, pix_X);
								D_number[1] += Alpha_index * (double)RandomMap_G[int_pyr].at<double>(pix_Y + 1, pix_X);
								D_number[0] += Alpha_index * (double)RandomMap_B[int_pyr].at<double>(pix_Y + 1, pix_X);
								D_number2[2] += Alpha_index * ((double)RandomMap_R2[int_pyr].at<double>(pix_Y + 1, pix_X) - (double)RandomMap_R[int_pyr].at<double>(pix_Y + 1, pix_X));
								D_number2[1] += Alpha_index * ((double)RandomMap_G2[int_pyr].at<double>(pix_Y + 1, pix_X) - (double)RandomMap_G[int_pyr].at<double>(pix_Y + 1, pix_X));
								D_number2[0] += Alpha_index * ((double)RandomMap_B2[int_pyr].at<double>(pix_Y + 1, pix_X) - (double)RandomMap_B[int_pyr].at<double>(pix_Y + 1, pix_X));
							}
							denom += Alpha_index;
							denom2 += Alpha_index;
							link_num++;
						}

						if (pyr_up[occ_index] != 0) {		//pyramid up
							h_Level = int_pyr + 1;
							if (pix_X % 2 == 0) { h_x = pix_X / 2; }
							else { h_x = (pix_X - 1) / 2; }
							if (pix_Y % 2 == 0) { h_y = pix_Y / 2; }
							else { h_y = (pix_Y - 1) / 2; }

							if (pyr_up[occ_index] == 1) {
								M_number[2] += Alpha_index * (double)RandomMap_R[h_Level].at<double>(h_y, h_x);
								M_number[1] += Alpha_index * (double)RandomMap_G[h_Level].at<double>(h_y, h_x);
								M_number[0] += Alpha_index * (double)RandomMap_B[h_Level].at<double>(h_y, h_x);
								M_number2[2] += Alpha_index * ((double)RandomMap_R2[h_Level].at<double>(h_y, h_x) - (double)RandomMap_R[h_Level].at<double>(h_y, h_x));
								M_number2[1] += Alpha_index * ((double)RandomMap_G2[h_Level].at<double>(h_y, h_x) - (double)RandomMap_G[h_Level].at<double>(h_y, h_x));
								M_number2[0] += Alpha_index * ((double)RandomMap_B2[h_Level].at<double>(h_y, h_x) - (double)RandomMap_B[h_Level].at<double>(h_y, h_x));
							}
							else {
								D_number[2] += Alpha_index * (double)RandomMap_R[h_Level].at<double>(h_y, h_x);
								D_number[1] += Alpha_index * (double)RandomMap_G[h_Level].at<double>(h_y, h_x);
								D_number[0] += Alpha_index * (double)RandomMap_B[h_Level].at<double>(h_y, h_x);
								D_number2[2] += Alpha_index * ((double)RandomMap_R2[h_Level].at<double>(h_y, h_x) - (double)RandomMap_R[h_Level].at<double>(h_y, h_x));
								D_number2[1] += Alpha_index * ((double)RandomMap_G2[h_Level].at<double>(h_y, h_x) - (double)RandomMap_G[h_Level].at<double>(h_y, h_x));
								D_number2[0] += Alpha_index * ((double)RandomMap_B2[h_Level].at<double>(h_y, h_x) - (double)RandomMap_B[h_Level].at<double>(h_y, h_x));
							}
							denom += Alpha_index;
							denom2 += Alpha_index;
							link_num++;
						}
						if (pyr_down[occ_index] != 0) {		//pyramid down
							h_Level = int_pyr - 1;
							h_x = pix_X * 2;
							h_y = pix_Y * 2;

							if (pyr_down[occ_index] == 1) {
								M_number[2] += Alpha_index * (double)RandomMap_R[h_Level].at<double>(h_y - 1, h_x - 1);
								M_number[1] += Alpha_index * (double)RandomMap_G[h_Level].at<double>(h_y - 1, h_x - 1);
								M_number[0] += Alpha_index * (double)RandomMap_B[h_Level].at<double>(h_y - 1, h_x - 1);
								M_number2[2] += Alpha_index * ((double)RandomMap_R2[h_Level].at<double>(h_y - 1, h_x - 1) - (double)RandomMap_R[h_Level].at<double>(h_y - 1, h_x - 1));
								M_number2[1] += Alpha_index * ((double)RandomMap_G2[h_Level].at<double>(h_y - 1, h_x - 1) - (double)RandomMap_G[h_Level].at<double>(h_y - 1, h_x - 1));
								M_number2[0] += Alpha_index * ((double)RandomMap_B2[h_Level].at<double>(h_y - 1, h_x - 1) - (double)RandomMap_B[h_Level].at<double>(h_y - 1, h_x - 1));
							}
							else if (pyr_down[occ_index] == 2) {
								D_number[2] += Alpha_index * (double)RandomMap_R[h_Level].at<double>(h_y - 1, h_x - 1);
								D_number[1] += Alpha_index * (double)RandomMap_G[h_Level].at<double>(h_y - 1, h_x - 1);
								D_number[0] += Alpha_index * (double)RandomMap_B[h_Level].at<double>(h_y - 1, h_x - 1);
								D_number2[2] += Alpha_index * ((double)RandomMap_R2[h_Level].at<double>(h_y - 1, h_x - 1) - (double)RandomMap_R[h_Level].at<double>(h_y - 1, h_x - 1));
								D_number2[1] += Alpha_index * ((double)RandomMap_G2[h_Level].at<double>(h_y - 1, h_x - 1) - (double)RandomMap_G[h_Level].at<double>(h_y - 1, h_x - 1));
								D_number2[0] += Alpha_index * ((double)RandomMap_B2[h_Level].at<double>(h_y - 1, h_x - 1) - (double)RandomMap_B[h_Level].at<double>(h_y - 1, h_x - 1));
							}
							if (pyr_down2[occ_index] == 1) {
								M_number[2] += Alpha_index * (double)RandomMap_R[h_Level].at<double>(h_y - 1, h_x);
								M_number[1] += Alpha_index * (double)RandomMap_G[h_Level].at<double>(h_y - 1, h_x);
								M_number[0] += Alpha_index * (double)RandomMap_B[h_Level].at<double>(h_y - 1, h_x);
								M_number2[2] += Alpha_index * ((double)RandomMap_R2[h_Level].at<double>(h_y - 1, h_x) - (double)RandomMap_R[h_Level].at<double>(h_y - 1, h_x));
								M_number2[1] += Alpha_index * ((double)RandomMap_G2[h_Level].at<double>(h_y - 1, h_x) - (double)RandomMap_G[h_Level].at<double>(h_y - 1, h_x));
								M_number2[0] += Alpha_index * ((double)RandomMap_B2[h_Level].at<double>(h_y - 1, h_x) - (double)RandomMap_B[h_Level].at<double>(h_y - 1, h_x));
							}
							else if (pyr_down2[occ_index] == 2) {
								D_number[2] += Alpha_index * (double)RandomMap_R[h_Level].at<double>(h_y - 1, h_x);
								D_number[1] += Alpha_index * (double)RandomMap_G[h_Level].at<double>(h_y - 1, h_x);
								D_number[0] += Alpha_index * (double)RandomMap_B[h_Level].at<double>(h_y - 1, h_x);
								D_number2[2] += Alpha_index * ((double)RandomMap_R2[h_Level].at<double>(h_y - 1, h_x) - (double)RandomMap_R[h_Level].at<double>(h_y - 1, h_x));
								D_number2[1] += Alpha_index * ((double)RandomMap_G2[h_Level].at<double>(h_y - 1, h_x) - (double)RandomMap_G[h_Level].at<double>(h_y - 1, h_x));
								D_number2[0] += Alpha_index * ((double)RandomMap_B2[h_Level].at<double>(h_y - 1, h_x) - (double)RandomMap_B[h_Level].at<double>(h_y - 1, h_x));
							}
							if (pyr_down3[occ_index] == 1) {
								M_number[2] += Alpha_index * (double)RandomMap_R[h_Level].at<double>(h_y, h_x - 1);
								M_number[1] += Alpha_index * (double)RandomMap_G[h_Level].at<double>(h_y, h_x - 1);
								M_number[0] += Alpha_index * (double)RandomMap_B[h_Level].at<double>(h_y, h_x - 1);
								M_number2[2] += Alpha_index * ((double)RandomMap_R2[h_Level].at<double>(h_y, h_x - 1) - (double)RandomMap_R[h_Level].at<double>(h_y, h_x - 1));
								M_number2[1] += Alpha_index * ((double)RandomMap_G2[h_Level].at<double>(h_y, h_x - 1) - (double)RandomMap_G[h_Level].at<double>(h_y, h_x - 1));
								M_number2[0] += Alpha_index * ((double)RandomMap_B2[h_Level].at<double>(h_y, h_x - 1) - (double)RandomMap_B[h_Level].at<double>(h_y, h_x - 1));
							}
							else if (pyr_down3[occ_index] == 2) {
								D_number[2] += Alpha_index * (double)RandomMap_R[h_Level].at<double>(h_y, h_x - 1);
								D_number[1] += Alpha_index * (double)RandomMap_G[h_Level].at<double>(h_y, h_x - 1);
								D_number[0] += Alpha_index * (double)RandomMap_B[h_Level].at<double>(h_y, h_x - 1);
								D_number2[2] += Alpha_index * ((double)RandomMap_R2[h_Level].at<double>(h_y, h_x - 1) - (double)RandomMap_R[h_Level].at<double>(h_y, h_x - 1));
								D_number2[1] += Alpha_index * ((double)RandomMap_G2[h_Level].at<double>(h_y, h_x - 1) - (double)RandomMap_G[h_Level].at<double>(h_y, h_x - 1));
								D_number2[0] += Alpha_index * ((double)RandomMap_B2[h_Level].at<double>(h_y, h_x - 1) - (double)RandomMap_B[h_Level].at<double>(h_y, h_x - 1));
							}
							if (pyr_down4[occ_index] == 1) {
								M_number[2] += Alpha_index * (double)RandomMap_R[h_Level].at<double>(h_y, h_x);
								M_number[1] += Alpha_index * (double)RandomMap_G[h_Level].at<double>(h_y, h_x);
								M_number[0] += Alpha_index * (double)RandomMap_B[h_Level].at<double>(h_y, h_x);
								M_number2[2] += Alpha_index * ((double)RandomMap_R2[h_Level].at<double>(h_y, h_x) - (double)RandomMap_R[h_Level].at<double>(h_y, h_x));
								M_number2[1] += Alpha_index * ((double)RandomMap_G2[h_Level].at<double>(h_y, h_x) - (double)RandomMap_G[h_Level].at<double>(h_y, h_x));
								M_number2[0] += Alpha_index * ((double)RandomMap_B2[h_Level].at<double>(h_y, h_x) - (double)RandomMap_B[h_Level].at<double>(h_y, h_x));
							}
							else if (pyr_down4[occ_index] == 2) {
								D_number[2] += Alpha_index * (double)RandomMap_R[h_Level].at<double>(h_y, h_x);
								D_number[1] += Alpha_index * (double)RandomMap_G[h_Level].at<double>(h_y, h_x);
								D_number[0] += Alpha_index * (double)RandomMap_B[h_Level].at<double>(h_y, h_x);
								D_number2[2] += Alpha_index * ((double)RandomMap_R2[h_Level].at<double>(h_y, h_x) - (double)RandomMap_R[h_Level].at<double>(h_y, h_x));
								D_number2[1] += Alpha_index * ((double)RandomMap_G2[h_Level].at<double>(h_y, h_x) - (double)RandomMap_G[h_Level].at<double>(h_y, h_x));
								D_number2[0] += Alpha_index * ((double)RandomMap_B2[h_Level].at<double>(h_y, h_x) - (double)RandomMap_B[h_Level].at<double>(h_y, h_x));
							}
							denom += (double)(Alpha_index * 4);
							denom2 += (double)(Alpha_index * 4);
							link_num++;
						}
						number2[2] += Alpha * link_num * (double)RandomMap_R[int_pyr].at<double>(Y_index, X_index);
						number2[1] += Alpha * link_num * (double)RandomMap_G[int_pyr].at<double>(Y_index, X_index);
						number2[0] += Alpha * link_num * (double)RandomMap_B[int_pyr].at<double>(Y_index, X_index);


						for (int color_index = 0; color_index < 3; color_index++) {
							number[color_index] += (double)(M_number[color_index] + D_number[color_index]);
							number2[color_index] += (double)(M_number2[color_index] + D_number2[color_index]);
						}

						occ_index++;
						if (occ_index > occ_number) { cout << "WARNING! : occ_index > occ_number in " << int_pyr << endl; }
					}

					for (int color_index = 0; color_index < 3; color_index++) {
						ave2[color_index] = (double)number2[color_index] / (double)denom2;
						number[color_index] += (double)Gamma_index * ave2[color_index];
						ave[color_index] = (double)number[color_index] / (double)denom;

						if (color_index == 0) {
							errorConvergence += fabs(RandomMap_B[int_pyr].at<double>(pix_Y, pix_X) - ave[color_index]);
						}
						else if (color_index == 1) {
							errorConvergence += fabs(RandomMap_G[int_pyr].at<double>(pix_Y, pix_X) - ave[color_index]);
						}
						else if (color_index == 2) {
							errorConvergence += fabs(RandomMap_R[int_pyr].at<double>(pix_Y, pix_X) - ave[color_index]);
						}

						int R_temp_index = int_pyr * 3 + color_index;
						RandomMap_temp[R_temp_index].at<double>(pix_Y, pix_X) = (double)ave[color_index];
						RandomMap_temp2[R_temp_index].at<double>(pix_Y, pix_X) = (double)ave2[color_index];
					}
				}
			}
			errorConvergence = (double)(errorConvergence / 3);
		}

		int temp_index = 0;
		for (int int_pyr = 0; int_pyr <= L; int_pyr++) {
			RandomMap_temp[temp_index].copyTo(RandomMap_B[int_pyr]);
			RandomMap_temp[temp_index + 1].copyTo(RandomMap_G[int_pyr]);
			RandomMap_temp[temp_index + 2].copyTo(RandomMap_R[int_pyr]);
			RandomMap_temp2[temp_index].copyTo(RandomMap_B2[int_pyr]);
			RandomMap_temp2[temp_index + 1].copyTo(RandomMap_G2[int_pyr]);
			RandomMap_temp2[temp_index + 2].copyTo(RandomMap_R2[int_pyr]);
			temp_index = temp_index + 3;
		}
		RandomMap_temp.clear();
		RandomMap_temp2.clear();

		if (errorConvergence / MAX_DATA < Converge) {
			cout << "収束成功: errorConvergence = " << errorConvergence << " , Iteration " << index_R + 1 << endl;
			break;
		}
		/*else {
			cout << "収束失敗!: errorConvergence = " << errorConvergence << " , Iteration " << index_R << endl;
		}*/
	}

	/* 画像補修 */
	int occ_X, occ_Y;
	occ_number = ImgPyr.occ_pix_index[0];
	for (int occ_index = 0; occ_index < occ_number; occ_index++) {
		occ_X = ImgPyr.occ_pix[occ_index].x;
		occ_Y = ImgPyr.occ_pix[occ_index].y;

		ave[0] = (int)(((double)((double)RandomMap_B[0].at<double>(occ_Y, occ_X) + 1.0) / (double)2.0) * (double)MAX_INTENSE);
		ave[1] = (int)(((double)((double)RandomMap_G[0].at<double>(occ_Y, occ_X) + 1.0) / (double)2.0) * (double)MAX_INTENSE);
		ave[2] = (int)(((double)((double)RandomMap_R[0].at<double>(occ_Y, occ_X) + 1.0) / (double)2.0) * (double)MAX_INTENSE);
		for (int color_index = 0; color_index < 3; color_index++) {
			if (ave[color_index] < 0) {
				ave[color_index] = 0;
				cout << "WARNING! : under0" << Point2i(occ_X, occ_Y) << endl;
			}
			else if (ave[color_index] > 255) {
				ave[color_index] = 255;
				cout << "WARNING! : over255" << Point2i(occ_X, occ_Y) << endl;
			}
		}

		color[2] = (uchar)ave[2];	// R,G,B値に分解
		color[1] = (uchar)ave[1];
		color[0] = (uchar)ave[0];
		Image_dst.at<Vec3b>(occ_Y, occ_X) = color;	// ピクセル値（カラー）
	}

	Image_dst.copyTo(Image_DST);

	RandomMap_R.clear();
	RandomMap_G.clear();
	RandomMap_B.clear();
	RandomMap_R2.clear();
	RandomMap_G2.clear();
	RandomMap_B2.clear();
	up.clear();
	down.clear();
	left.clear();
	right.clear();
	pyr_up.clear();
	pyr_down.clear();
	pyr_down2.clear();
	pyr_down3.clear();
	pyr_down4.clear();

	ALPHA.clear();
	SIGMA.clear();
	GAMMA.clear();
}
