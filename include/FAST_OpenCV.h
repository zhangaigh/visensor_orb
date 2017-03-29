//from opencv src 

#ifndef FAST_OpenCV_H
#define FAST_OpenCV_H
using namespace cv;
using namespace std;

//#define CV_SSE2 1

#define VERIFY_CORNERS 0

inline void makeOffsets(int pixel[25], int rowStride, int patternSize)
{
	static const int offsets16[][2] =
	{
		{ 0,  3 },{ 1,  3 },{ 2,  2 },{ 3,  1 },{ 3, 0 },{ 3, -1 },{ 2, -2 },{ 1, -3 },
		{ 0, -3 },{ -1, -3 },{ -2, -2 },{ -3, -1 },{ -3, 0 },{ -3,  1 },{ -2,  2 },{ -1,  3 }
	};

	static const int offsets12[][2] =
	{
		{ 0,  2 },{ 1,  2 },{ 2,  1 },{ 2, 0 },{ 2, -1 },{ 1, -2 },
		{ 0, -2 },{ -1, -2 },{ -2, -1 },{ -2, 0 },{ -2,  1 },{ -1,  2 }
	};

	static const int offsets8[][2] =
	{
		{ 0,  1 },{ 1,  1 },{ 1, 0 },{ 1, -1 },
		{ 0, -1 },{ -1, -1 },{ -1, 0 },{ -1,  1 }
	};

	const int(*offsets)[2] = patternSize == 16 ? offsets16 :
		patternSize == 12 ? offsets12 :
		patternSize == 8 ? offsets8 : 0;

	int k = 0;
	for (; k < patternSize; k++)
		pixel[k] = offsets[k][0] + offsets[k][1] * rowStride;
	for (; k < 25; k++)
		pixel[k] = pixel[k - patternSize];
}

#if VERIFY_CORNERS
static void testCorner(const uchar* ptr, const int pixel[], int K, int N, int threshold) {
	// check that with the computed "threshold" the pixel is still a corner
	// and that with the increased-by-1 "threshold" the pixel is not a corner anymore
	for (int delta = 0; delta <= 1; delta++)
	{
		int v0 = std::min(ptr[0] + threshold + delta, 255);
		int v1 = std::max(ptr[0] - threshold - delta, 0);
		int c0 = 0, c1 = 0;

		for (int k = 0; k < N; k++)
		{
			int x = ptr[pixel[k]];
			if (x > v0)
			{
				if (++c0 > K)
					break;
				c1 = 0;
			}
			else if (x < v1)
			{
				if (++c1 > K)
					break;
				c0 = 0;
			}
			else
			{
				c0 = c1 = 0;
			}
		}
		CV_Assert((delta == 0 && std::max(c0, c1) > K) ||
			(delta == 1 && std::max(c0, c1) <= K));
	}
}
#endif

inline int cornerScore16(const unsigned char* ptr, const int pixel[], int threshold)
{
	const int K = 8, N = K * 3 + 1;
	int k, v = ptr[0];
	short d[N];
	for (k = 0; k < N; k++)
		d[k] = (short)(v - ptr[pixel[k]]);

#if CV_SSE2
	__m128i q0 = _mm_set1_epi16(-1000), q1 = _mm_set1_epi16(1000);
	for (k = 0; k < 16; k += 8)
	{
		__m128i v0 = _mm_loadu_si128((__m128i*)(d + k + 1));
		__m128i v1 = _mm_loadu_si128((__m128i*)(d + k + 2));
		__m128i a = _mm_min_epi16(v0, v1);
		__m128i b = _mm_max_epi16(v0, v1);
		v0 = _mm_loadu_si128((__m128i*)(d + k + 3));
		a = _mm_min_epi16(a, v0);
		b = _mm_max_epi16(b, v0);
		v0 = _mm_loadu_si128((__m128i*)(d + k + 4));
		a = _mm_min_epi16(a, v0);
		b = _mm_max_epi16(b, v0);
		v0 = _mm_loadu_si128((__m128i*)(d + k + 5));
		a = _mm_min_epi16(a, v0);
		b = _mm_max_epi16(b, v0);
		v0 = _mm_loadu_si128((__m128i*)(d + k + 6));
		a = _mm_min_epi16(a, v0);
		b = _mm_max_epi16(b, v0);
		v0 = _mm_loadu_si128((__m128i*)(d + k + 7));
		a = _mm_min_epi16(a, v0);
		b = _mm_max_epi16(b, v0);
		v0 = _mm_loadu_si128((__m128i*)(d + k + 8));
		a = _mm_min_epi16(a, v0);
		b = _mm_max_epi16(b, v0);
		v0 = _mm_loadu_si128((__m128i*)(d + k));
		q0 = _mm_max_epi16(q0, _mm_min_epi16(a, v0));
		q1 = _mm_min_epi16(q1, _mm_max_epi16(b, v0));
		v0 = _mm_loadu_si128((__m128i*)(d + k + 9));
		q0 = _mm_max_epi16(q0, _mm_min_epi16(a, v0));
		q1 = _mm_min_epi16(q1, _mm_max_epi16(b, v0));
	}
	q0 = _mm_max_epi16(q0, _mm_sub_epi16(_mm_setzero_si128(), q1));
	q0 = _mm_max_epi16(q0, _mm_unpackhi_epi64(q0, q0));
	q0 = _mm_max_epi16(q0, _mm_srli_si128(q0, 4));
	q0 = _mm_max_epi16(q0, _mm_srli_si128(q0, 2));
	threshold = (short)_mm_cvtsi128_si32(q0) - 1;
#else
	int a0 = threshold;
	for (k = 0; k < 16; k += 2)
	{
		int a = std::min((int)d[k + 1], (int)d[k + 2]);
		a = std::min(a, (int)d[k + 3]);
		if (a <= a0)
			continue;
		a = std::min(a, (int)d[k + 4]);
		a = std::min(a, (int)d[k + 5]);
		a = std::min(a, (int)d[k + 6]);
		a = std::min(a, (int)d[k + 7]);
		a = std::min(a, (int)d[k + 8]);
		a0 = std::max(a0, std::min(a, (int)d[k]));
		a0 = std::max(a0, std::min(a, (int)d[k + 9]));
	}

	int b0 = -a0;
	for (k = 0; k < 16; k += 2)
	{
		int b = std::max((int)d[k + 1], (int)d[k + 2]);
		b = std::max(b, (int)d[k + 3]);
		b = std::max(b, (int)d[k + 4]);
		b = std::max(b, (int)d[k + 5]);
		if (b >= b0)
			continue;
		b = std::max(b, (int)d[k + 6]);
		b = std::max(b, (int)d[k + 7]);
		b = std::max(b, (int)d[k + 8]);

		b0 = std::min(b0, std::max(b, (int)d[k]));
		b0 = std::min(b0, std::max(b, (int)d[k + 9]));
	}

	threshold = -b0 - 1;
#endif

#if VERIFY_CORNERS
	testCorner(ptr, pixel, K, N, threshold);
#endif
	return threshold;
}

inline void wl_FAST(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression)
{
	int patternSize = 16;
	Mat img = _img.getMat();
	const int K = patternSize / 2, N = patternSize + K + 1;

#if CV_SSE2
	const int quarterPatternSize = patternSize / 4;
	(void)quarterPatternSize;
#endif

	int i, j, k, pixel[25];
	makeOffsets(pixel, (int)img.step, patternSize);

	keypoints.clear();

	threshold = std::min(std::max(threshold, 0), 255);

#if CV_SSE2
	__m128i delta = _mm_set1_epi8(-128), t = _mm_set1_epi8((char)threshold), K16 = _mm_set1_epi8((char)K);
	(void)K16;
	(void)delta;
	(void)t;
#endif

	uchar threshold_tab[512];
	for (i = -255; i <= 255; i++)
		threshold_tab[i + 255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);

	AutoBuffer<uchar> _buf((img.cols + 16) * 3 * (sizeof(int) + sizeof(uchar)) + 128);
	uchar* buf[3];
	buf[0] = _buf; buf[1] = buf[0] + img.cols; buf[2] = buf[1] + img.cols;
	int* cpbuf[3];
	cpbuf[0] = (int*)alignPtr(buf[2] + img.cols, sizeof(int)) + 1;
	cpbuf[1] = cpbuf[0] + img.cols + 1;
	cpbuf[2] = cpbuf[1] + img.cols + 1;
	memset(buf[0], 0, img.cols * 3);

	for (i = 3; i < img.rows - 2; i++)
	{
		const uchar* ptr = img.ptr<uchar>(i) + 3;
		uchar* curr = buf[(i - 3) % 3];
		int* cornerpos = cpbuf[(i - 3) % 3];
		memset(curr, 0, img.cols);
		int ncorners = 0;

		if (i < img.rows - 3)
		{
			j = 3;
#if CV_SSE2
			if (patternSize == 16)
			{
				for (; j < img.cols - 16 - 3; j += 16, ptr += 16)
				{
					__m128i m0, m1;
					__m128i v0 = _mm_loadu_si128((const __m128i*)ptr);
					__m128i v1 = _mm_xor_si128(_mm_subs_epu8(v0, t), delta);
					v0 = _mm_xor_si128(_mm_adds_epu8(v0, t), delta);

					__m128i x0 = _mm_sub_epi8(_mm_loadu_si128((const __m128i*)(ptr + pixel[0])), delta);
					__m128i x1 = _mm_sub_epi8(_mm_loadu_si128((const __m128i*)(ptr + pixel[quarterPatternSize])), delta);
					__m128i x2 = _mm_sub_epi8(_mm_loadu_si128((const __m128i*)(ptr + pixel[2 * quarterPatternSize])), delta);
					__m128i x3 = _mm_sub_epi8(_mm_loadu_si128((const __m128i*)(ptr + pixel[3 * quarterPatternSize])), delta);
					m0 = _mm_and_si128(_mm_cmpgt_epi8(x0, v0), _mm_cmpgt_epi8(x1, v0));
					m1 = _mm_and_si128(_mm_cmpgt_epi8(v1, x0), _mm_cmpgt_epi8(v1, x1));
					m0 = _mm_or_si128(m0, _mm_and_si128(_mm_cmpgt_epi8(x1, v0), _mm_cmpgt_epi8(x2, v0)));
					m1 = _mm_or_si128(m1, _mm_and_si128(_mm_cmpgt_epi8(v1, x1), _mm_cmpgt_epi8(v1, x2)));
					m0 = _mm_or_si128(m0, _mm_and_si128(_mm_cmpgt_epi8(x2, v0), _mm_cmpgt_epi8(x3, v0)));
					m1 = _mm_or_si128(m1, _mm_and_si128(_mm_cmpgt_epi8(v1, x2), _mm_cmpgt_epi8(v1, x3)));
					m0 = _mm_or_si128(m0, _mm_and_si128(_mm_cmpgt_epi8(x3, v0), _mm_cmpgt_epi8(x0, v0)));
					m1 = _mm_or_si128(m1, _mm_and_si128(_mm_cmpgt_epi8(v1, x3), _mm_cmpgt_epi8(v1, x0)));
					m0 = _mm_or_si128(m0, m1);
					int mask = _mm_movemask_epi8(m0);
					if (mask == 0)
						continue;
					if ((mask & 255) == 0)
					{
						j -= 8;
						ptr -= 8;
						continue;
					}

					__m128i c0 = _mm_setzero_si128(), c1 = c0, max0 = c0, max1 = c0;
					for (k = 0; k < N; k++)
					{
						__m128i x = _mm_xor_si128(_mm_loadu_si128((const __m128i*)(ptr + pixel[k])), delta);
						m0 = _mm_cmpgt_epi8(x, v0);
						m1 = _mm_cmpgt_epi8(v1, x);

						c0 = _mm_and_si128(_mm_sub_epi8(c0, m0), m0);
						c1 = _mm_and_si128(_mm_sub_epi8(c1, m1), m1);

						max0 = _mm_max_epu8(max0, c0);
						max1 = _mm_max_epu8(max1, c1);
					}

					max0 = _mm_max_epu8(max0, max1);
					int m = _mm_movemask_epi8(_mm_cmpgt_epi8(max0, K16));

					for (k = 0; m > 0 && k < 16; k++, m >>= 1)
						if (m & 1)
						{
							cornerpos[ncorners++] = j + k;
							if (nonmax_suppression)
								curr[j + k] = (uchar)cornerScore16(ptr + k, pixel, threshold);
						}
				}
			}
#endif
			for (; j < img.cols - 3; j++, ptr++)
			{
				int v = ptr[0];
				const uchar* tab = &threshold_tab[0] - v + 255;
				int d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];

				if (d == 0)
					continue;

				d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
				d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
				d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];

				if (d == 0)
					continue;

				d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
				d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
				d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
				d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];

				if (d & 1)
				{
					int vt = v - threshold, count = 0;

					for (k = 0; k < N; k++)
					{
						int x = ptr[pixel[k]];
						if (x < vt)
						{
							if (++count > K)
							{
								cornerpos[ncorners++] = j;
								if (nonmax_suppression)
									curr[j] = (uchar)cornerScore16(ptr, pixel, threshold);
								break;
							}
						}
						else
							count = 0;
					}
				}

				if (d & 2)
				{
					int vt = v + threshold, count = 0;

					for (k = 0; k < N; k++)
					{
						int x = ptr[pixel[k]];
						if (x > vt)
						{
							if (++count > K)
							{
								cornerpos[ncorners++] = j;
								if (nonmax_suppression)
									curr[j] = (uchar)cornerScore16(ptr, pixel, threshold);
								break;
							}
						}
						else
							count = 0;
					}
				}
			}
		}

		cornerpos[-1] = ncorners;

		if (i == 3)
			continue;

		const uchar* prev = buf[(i - 4 + 3) % 3];
		const uchar* pprev = buf[(i - 5 + 3) % 3];
		cornerpos = cpbuf[(i - 4 + 3) % 3];
		ncorners = cornerpos[-1];

		for (k = 0; k < ncorners; k++)
		{
			j = cornerpos[k];
			int score = prev[j];
			if (!nonmax_suppression ||
				(score > prev[j + 1] && score > prev[j - 1] &&
					score > pprev[j - 1] && score > pprev[j] && score > pprev[j + 1] &&
					score > curr[j - 1] && score > curr[j] && score > curr[j + 1]))
			{
				keypoints.push_back(KeyPoint((float)j, (float)(i - 1), 7.f, -1, (float)score));
			}
		}
	}
}

#endif  
