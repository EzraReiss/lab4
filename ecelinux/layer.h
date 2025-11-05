//===========================================================================
// layer.h
//===========================================================================
// @brief: This header file defines the interface for the core functions.

#ifndef LAYER_H
#define LAYER_H

#include "model.h"
#include "typedefs.h"

#include <assert.h>
typedef ap_uint<32> HLS_SIZE_T;
#include "hls/hls_video_mem.h"

//----------------------------------------------------------
// Padding
//----------------------------------------------------------
// @param[in] : input - input fmaps
//              M - number of input fmaps
//              I - width of input fmaps
// @param[out] : output - output fmaps
template <int M, int I>
void pad(bit input[M][I][I], bit output[M][I + F_PAD][I + F_PAD]) {
  #pragma HLS INLINE off
  #pragma HLS array_reshape variable = input complete dim = 1
  #pragma HLS array_reshape variable = output complete dim = 1
  for (int x = 0; x < I; x++) {
    for (int y = 0; y < I; y++) {
      #pragma HLS pipeline
      for (int m = 0; m < M; m++) {
        output[m][y + F_PAD / 2][x + F_PAD / 2] = input[m][y][x];
      }
    }
  }
}

//----------------------------------------------------------
// Initialize Padded Memory with Constant
//----------------------------------------------------------
// @param[in] : input - input fmaps to be initialized
// @param[out] : output - output fmaps
template <int M, int I, int C>
void initialize_padded_memory(bit input[M][I][I]) {
  #pragma HLS INLINE off
  #pragma HLS array_reshape variable = input complete dim = 1

  for (int x = 0; x < I; x++) {
    for (int y = 0; y < I; y++) {
      #pragma HLS pipeline
      for (int m = 0; m < M; m++) {
        input[m][x][y] = C;
      }
    }
  }
}


//----------------------------------------------------------
// Perform Convolution Layer
//----------------------------------------------------------
// @param[in] : input - input fmaps
//              threshold - threshold for batchnorm operation
//              M - number of input fmaps
//              N - number of output fmaps
//              I - width of input fmaps
//              weight - layer weights
//              T - tile size for the output features
// @param[out] : output - output fmaps
template <int M, int N, int I, int T>
void conv(
  bit input[M][I][I], 
  bit output[N][I - F + 1][I - F + 1],
  const bit8_t threshold[N], 
  const bit weight[M][N][F][F]
) {
  #pragma HLS inline off

  bit input_slice[M][F][F];

  // Adding BRAMs for rows and columns using cyclic partitioning
  // #pragma HLS array_partition variable = weight complete dim = 4
  // #pragma HLS array_partition variable = weight complete dim = 3
  // #pragma HLS array_partition variable = weight complete dim = 2
  #pragma HLS array_reshape variable = weight complete dim = 1
  #pragma HLS array_reshape variable = input complete dim = 1

  // Adding BRAMs for the N dimension (output features)
  // #pragma HLS array_partition variable = output complete dim = 1

  int num_accum = F * F * M;

  hls::Window<F, F, ap_uint<M>> window;
  hls::LineBuffer<F, I, ap_uint<M>> linebuf;

  // #pragma HLS array_partition variable = linebuf complete dim = 0
  // #pragma HLS array_partition variable = window complete dim = 0
  // #pragma HLS array_partition variable = threshold complete dim = 1
  // fill linebuf with first F rows form input
  initialize_linebuf:
  for (int x = 0; x < I; x++) {
    for (int y = 0; y < F; y++) {
      #pragma HLS pipeline
      linebuf.shift_pixels_up(x);

      ap_uint<M> pixel_val;
      for (int m = 0; m < M; m++) {
        pixel_val[m] = input[m][y][x];
      }
      
      linebuf.insert_bottom_row(pixel_val, x);
    }
  }

  // fill window with the first FxF pixels from linebuf
  initialize_window:
  for (int y = 0; y < F; y++) { 
    #pragma HLS pipeline
    for (int x = 0; x < F; x++) {
      window.insert_pixel(linebuf.getval(x, y), x, y);
    }
  }
  OUTPUT_LOOP:
  for (int y = 0; y < I - F + 1; y++) {
    for (int x = 0; x < I - F + 1; x++) {
      if (T == 1) {
        #pragma HLS pipeline II=1
      } 
      OUTPUT_FEATURES:
      for (int nt = 1; nt <= T; nt++) {
        #pragma HLS pipeline II=1
        int lower_bound = (nt-1) * (N / T); //Tx tile over N since resource util was high
        for (int ntt = 0; ntt < N/T; ntt++) {
          bit16_t accum = 0;
          int n = lower_bound + ntt;
          for (int c = 0; c < F; c++) {
            for (int r = 0; r < F; r++) {
              ap_uint<M> pixel = window.getval(r, c);
              for (int m = 0; m < M; m++) {
                accum += pixel[m] == weight[m][n][r][c];
              }
            }
          }

          accum = (accum << 1) - num_accum;
          output[n][y][x] = accum > threshold[n] ? 1 : 0;
        }
      }
      
      window.shift_pixels_left();
      for (int r = 0; r < F; r++) {
        window.insert_pixel(linebuf.getval(r, x + F), r, F - 1);
      }
    }
    
    for (int x = 0; x < I; x++) {
      #pragma HLS pipeline
      linebuf.shift_pixels_up(x);

      ap_uint<M> pixel_val;
      for (int m = 0; m < M; m++) {
        pixel_val[m] = input[m][y + F][x];
      }

      linebuf.insert_bottom_row(pixel_val, x);
    }
    
    for (int yy = 0; yy < F; yy++) {
      #pragma HLS pipeline
      for (int xx = 0; xx < F; xx++) {
        window.insert_pixel(linebuf.getval(xx, yy), xx, yy);
      }
    }
  }

}

//----------------------------------------------------------
// Max pooling
//----------------------------------------------------------
// @param[in] : input - input fmaps
//              M - number of input fmaps
//              I - width of input fmaps
// @param[out] : output - output fmaps
template <int M, int I>
void max_pool(bit input[M][I][I], bit output[M][I / 2][I / 2]) {
  #pragma HLS INLINE off

  #pragma HLS array_reshape variable=input complete dim = 1
  #pragma HLS array_reshape variable=output complete dim = 1

  for (int x = 0; x < I / 2; x++) {
    for (int y = 0; y < I / 2; y++) {
      #pragma HLS pipeline

      bit accum[M];
      for (int m = 0; m < M; m++) {
        accum[m] = 0;
      }

      for (int c = 0; c < 2; c++) {
        for (int r = 0; r < 2; r++) {
          for (int m = 0; m < M; m++) {
            #pragma HLS unroll
            accum[m] |= input[m][2 * y + r][2 * x + c];
          }
        }
      }

      for (int m = 0; m < M; m++) {
        #pragma HLS unroll
        output[m][y][x] = accum[m];
      }
    }
  }
}

//----------------------------------------------------------
// Flatten the Output from Conv Layer
//----------------------------------------------------------
// @param[in] : input - output fmaps from the last conv layer
// @param[out] : output - input famps of the first dense layer
void flatten(bit input[O_CHANNEL2][O_WIDTH][O_WIDTH], bit output[I_UNITS1]) {
  #pragma HLS INLINE off

  #pragma HLS array_reshape variable=input complete dim=1
  #pragma HLS array_partition variable=input complete dim=2
  #pragma HLS array_partition variable=input complete dim=3
  #pragma HLS array_partition variable=output complete dim=1 //already partitioned in dense?


  // for (int c = 0; c < O_CHANNEL2; c++) {
  //   #pragma HLS pipeline
  //   for (int y = 0; y < O_WIDTH; y++) {
  //     for (int x = 0; x < O_WIDTH; x++) {
  //       int o_index = c + (x + y * O_WIDTH) * O_CHANNEL2;
  //       output[o_index] = input[c][y][x];
  //     }
  //   }
  // }

  for (int i = 0; i < I_UNITS1; ++i) {
    #pragma HLS unroll 
    //sams schitzo implementation
    int a = (i & 0b001100000) >> 5;
    int b = (i & 0b110000000) >> 7;
    int c = (i & 0b000011111) >> 0;
    output[i] = input[c][b][a];
  }
}

//----------------------------------------------------------
// Perform Sign Layer
//----------------------------------------------------------
// @param[in] : input - input fmaps
//              M - number of input and output channels
// @param[out] : output - output fmaps
template <int M> 
void sign(bit16_t input[M], bit output[M]) {
  // #pragma HLS INLINE off

  for (int m = 0; m < M; m++) {
    output[m] = (input[m] > 0) ? 1 : 0;
  }
}

//----------------------------------------------------------
// Perform Argmax Layer
//----------------------------------------------------------
// @param[in] : input - input channels
// @param[out] : output - argmax of the inputs
bit4_t argmax(bit16_t input[NUM_DIGITS]) {
  // #pragma HLS INLINE off
  bit16_t max = input[0];
  bit4_t max_id = 0;
  for (int i = 1; i < NUM_DIGITS; i++) {
    if (input[i] > max) {
      max = input[i];
      max_id = i;
    }
  }
  return max_id;
}

//----------------------------------------------------------
// Perform Dense Layer
//----------------------------------------------------------
// @param[in] : input - input fmaps
//              M - number of input fmaps
//              N - number of output fmaps
//              weight - layer weights
// @param[out] : output - output fmaps
template <int M, int N>
void dense(bit input[M], bit16_t output[N], const bit weight[M][N]) {
  #pragma HLS INLINE off
  #pragma HLS array_reshape variable=input complete dim=1
  #pragma HLS array_reshape variable=weight complete dim=1
  // #pragma HLS array_partition variable=output complete dim=1
  // #pragma HLS array_partition variable=weight complete dim=2
  // Partition over input and weight for higher parallelism
  // Don't do complete since M and N can be large

  for (int n = 0; n < N; n++) {
    #pragma HLS pipeline
    bit16_t accum = 0;
    for (int m = 0; m < M; m++) {
      #pragma HLS unroll 
      accum += input[m] == weight[m][n]; // XNOR
    }
    output[n] = (accum << 1) - M;
  }
}

#endif
