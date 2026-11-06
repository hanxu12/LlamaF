#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <iostream>
#include <vector>


#define GS 256  // GS=256, n=dim=5632, dim/GS=22
#define NGROUPS 22
typedef hls::vector<int8_t, GS> int8_vec_t;
typedef hls::vector<int16_t, GS> int16_vec_t;
typedef hls::vector<int32_t, GS> int32_vec_t;
typedef hls::vector<float, NGROUPS> float_vec_t;
#define DATA_SIZE 5632
const int c_size = DATA_SIZE;



static void read_w(int8_vec_t* w, 
	hls::stream<int8_vec_t>& w_stream,
	int d) {
Loop_read_w:
	for (int i = 0; i < d; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
		int in = i * NGROUPS;
		for (int j = 0; j < NGROUPS; j++) {
			w_stream << w[in + j];
		}
	}
}


static void castw(
	hls::stream<int8_vec_t>& w_stream,
	hls::stream<int16_vec_t>& w_stream16,
	int d) {
Loop_castw:
	for (int i = 0; i < d; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
		for (int j = 0; j < NGROUPS; j++) {
			#pragma HLS PIPELINE II=1
			int8_vec_t w = w_stream.read();
			int16_vec_t w16;
			for (int j = 0; j < GS; j++) {
				w16[j] = (int16_t)w[j];
			}
			w_stream16 << w16;
		}
	}
}

static void read_scale(float* scale,
	hls::stream<float>& xs_stream,
	hls::stream<float>& ws_stream,
	int d) {
Loop_read_ws:
	for (int i = 0; i < d; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
		for (int j = 0; j < NGROUPS; j++) {
			xs_stream << scale[j];
		}
		int in = (i + 1) * NGROUPS;
		for (int j = 0; j < NGROUPS; j++) {
			ws_stream << scale[in + j];
		}
	}
}

static void mul(int16_vec_t x_stream16_arr[NGROUPS],
				hls::stream<int16_vec_t>& w_stream,
				hls::stream<int32_t>& psum_stream,													
				int d) {
Loop_mul:
	for (int i = 0; i < d; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
		for (int j = 0; j < NGROUPS; j++) {
			int16_vec_t mul_res = x_stream16_arr[j] * w_stream.read();
			// Adder Tree
            int32_t sum1[128], sum2[64], sum3[32], sum4[16], sum5[8], sum6[4], sum7[2];

            // Level 1
            #pragma HLS ARRAY_PARTITION variable=sum1 complete
            for (int k = 0; k < 128; k++) {
                #pragma HLS UNROLL
                sum1[k] = (int32_t)mul_res[k] + (int32_t)mul_res[k + 128];
            }

			// Level 2
			#pragma HLS ARRAY_PARTITION variable=sum2 complete
			for (int k = 0; k < 64; k++) {
				#pragma HLS UNROLL
				sum2[k] = sum1[k] + sum1[k + 64];
			}

			// Level 3
			#pragma HLS ARRAY_PARTITION variable=sum3 complete
			for (int k = 0; k < 32; k++) {
				#pragma HLS UNROLL
				sum3[k] = sum2[k] + sum2[k + 32];
			}

			// Level 4
			#pragma HLS ARRAY_PARTITION variable=sum4 complete
			for (int k = 0; k < 16; k++) {
				#pragma HLS UNROLL
				sum4[k] = sum3[k] + sum3[k + 16];
			}

			// Level 5
			#pragma HLS ARRAY_PARTITION variable=sum5 complete
			for (int k = 0; k < 8; k++) {
				#pragma HLS UNROLL
				sum5[k] = sum4[k] + sum4[k + 8];
			}

			// Level 6
			#pragma HLS ARRAY_PARTITION variable=sum6 complete
			for (int k = 0; k < 4; k++) {
				#pragma HLS UNROLL
				sum6[k] = sum5[k] + sum5[k + 4];
			}

			// Level 7
			#pragma HLS ARRAY_PARTITION variable=sum7 complete
			for (int k = 0; k < 2; k++) {
				#pragma HLS UNROLL
				sum7[k] = sum6[k] + sum6[k + 2];
			}

			psum_stream << sum7[0] + sum7[1];
		}
	}
}


// TODO: make this vector operation?
static void acc2(float* xout,
				hls::stream<int32_t>& psum_stream,
				hls::stream<float>& xs_stream,
				hls::stream<float>& ws_stream,
				int d) {
Loop_acc2:
	for (int i = 0; i < d; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
		float sum = 0.0f;	
		for (int j = 0; j < NGROUPS; j++) {
			float psum = (float)psum_stream.read();
			sum += psum * ws_stream.read() * xs_stream.read();
		}
		xout[i] = sum;
	}
}


extern "C" {
	void matmul2(float* xout, int8_t *x, int8_vec_t *w, float* scale, int d) {
		// W (d,n) @ x (n,) -> xout (d,)
	#pragma HLS INTERFACE m_axi port = xout depth = 2048 bundle = aximm1   // float: dim rows (W2 only)
	#pragma HLS INTERFACE m_axi port = x depth = 5632 bundle = aximm1      // int8: hidden_dim
	#pragma HLS INTERFACE m_axi port = w depth = 45056 bundle = aximm2     // int8_vec_t (GS int8): 2048 rows x 22 groups
	#pragma HLS INTERFACE m_axi port = scale depth = 45078 bundle = aximm3 // float: first 22 are xs, then 22 ws per row (22 * 2048)

	hls::stream<int8_vec_t> w_stream;
	hls::stream<int16_vec_t> w_stream16;
	// Note: 
	// 1. Vitis HLS warns that xs_stream/ws_stream with the default depth (2) "can result
	// in deadlock", because read_scale writes a row's xs values before its ws values while
	// acc2 reads them in pairs. 
	// 2. However, C/RTL co-simulation (Vitis HLS 2022.2) passes without
	// deadlock and with the same latency as with depth-44 FIFOs, so we keep the default depth (2).
	hls::stream<float> xs_stream, ws_stream;
	hls::stream<int32_t> psum_stream;
	int16_vec_t x_stream16_arr[NGROUPS];

	for (int i = 0; i < NGROUPS; i++) {
		for (int j = 0; j < GS; j++) {
			x_stream16_arr[i][j] = (int16_t)x[j + i * GS];
		}
	}

	#pragma HLS dataflow
		read_w(w, w_stream, d);
		castw(w_stream, w_stream16, d);
		read_scale(scale, xs_stream, ws_stream, d);
		mul(x_stream16_arr, w_stream16, psum_stream, d);
		acc2(xout, psum_stream, xs_stream, ws_stream, d);
	}
}
