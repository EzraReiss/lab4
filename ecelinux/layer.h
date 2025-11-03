//===========================================================================
// layer.h
//===========================================================================
// @brief: This header file defines the interface for the core functions.

#ifndef LAYER_H
#define LAYER_H

#include "model.h"
#include "typedefs.h"

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
  for (int m = 0; m < M; m++) {
    for (int x = 0; x < I; x++) {
      for (int y = 0; y < I; y++) {
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
  for (int m = 0; m < M; m++) {
    for (int x = 0; x < I; x++) {
      for (int y = 0; y < I; y++) {
        input[m][x][y] = C;
      }
    }
  }
}

template <
  typename T,
  int M
> 
void initialize_line_buffer(
  T line_buffer[M],
  T initial_values[M]
) {
  for (int m = 0; m < M; m++) {
    line_buffer[m] = initial_values[m];
  }
}

template<
  typename T,
  int M
> void shift_line_buffer(
  T line_buffer[M],
  T new_value
) {
  for (int m = M - 1; m > 0; m--) {
    line_buffer[m] = line_buffer[m - 1];
  }
  line_buffer[0] = new_value;
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
// @param[out] : output - output fmaps
template <int M, int N, int I>
void conv(
  bit input[M][I][I], 
  bit output[N][I - F + 1][I - F + 1],
  const bit8_t threshold[N], 
  const bit weight[M][N][F][F]
) {
  #pragma HLS INLINE off

  bit input_slice[M][F][F];

    // Adding BRAMs for rows and columns using cyclic partitioning
  #pragma HLS array_partition variable = weight complete dim = 4
  #pragma HLS array_partition variable = weight complete dim = 3
  #pragma HLS array_partition variable = weight complete dim = 2
  #pragma HLS array_reshape variable = weight complete dim = 1

  #pragma HLS array_partition variable = input_slice complete dim = 3
  #pragma HLS array_partition variable = input_slice complete dim = 2
  #pragma HLS array_reshape variable = input_slice complete dim = 1

  #pragma HLS array_reshape variable = input complete dim = 1

  // Adding BRAMs for the N dimension (output features)
  #pragma HLS array_partition variable = output complete dim = 1

  int num_accum = F * F * M;


  for (int x = 0; x < I - F + 1; x++)
  {
    for (int c = 0; c < F; c++) {
      for (int r = 0; r < F; r++) {
        for (int m = 0; m < M; m++) {
          #pragma HLS unroll
          input_slice[m][r][c] = input[m][r][x + c];
        }
      }
    }

    for (int y = 0; y < I - F + 1; y++) {
      for (int n = 0; n < N; n++) {
        #pragma HLS pipeline

        bit16_t accum = 0;

        for (int c = 0; c < F; c++) {
          for (int r = 0; r < F; r++) {
            for (int m = 0; m < M; m++) {
              accum += input_slice[m][r][c] == weight[m][n][r][c];
            }
          }
        }

        accum = (accum << 1) - num_accum;
        output[n][y][x] = accum > threshold[n] ? 1 : 0;
      }

      // Shift input slice left by one column
      for (int r = 0; r < F - 1; r++) {
        #pragma HLS unroll
        for (int c = 0; c < F; c++) {
          #pragma HLS unroll
          for (int m = 0; m < M; m++) {
            #pragma HLS unroll
            input_slice[m][r][c] = input_slice[m][r + 1][c];
          }
        }
      }
      
      // Load new column into input slice
      for (int c = 0; c < F; c++) {
        for (int m = 0; m < M; m++) {
          #pragma HLS unroll
          input_slice[m][F - 1][c] = input[m][y + F][x + c];
        }
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
  for (int m = 0; m < M; m++) {
    for (int x = 0; x < I / 2; x++) {
      for (int y = 0; y < I / 2; y++) {
        bit max = 0;
        for (int c = 0; c < 2; c++) {
          for (int r = 0; r < 2; r++) {
            if (input[m][2 * y + r][2 * x + c])
              max = 1;
          }
        }
        output[m][y][x] = max;
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
  for (int c = 0; c < O_CHANNEL2; c++) {
    for (int y = 0; y < O_WIDTH; y++) {
      for (int x = 0; x < O_WIDTH; x++) {
        int o_index = c + (x + y * O_WIDTH) * O_CHANNEL2;
        output[o_index] = input[c][y][x];
      }
    }
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
  #pragma HLS INLINE off
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
  #pragma HLS INLINE off
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
  // Partition over input and weight for higher parallelism
  // Don't do complete since M and N can be large
  #pragma HLS array_reshape variable=input complete dim=1
  #pragma HLS array_reshape variable=weight complete dim=1

  for (int n = 0; n < N; n++) {
    bit16_t accum = 0;
    for (int m = 0; m < M; m++) {
      #pragma HLS unroll
      accum += input[m] == weight[m][n]; // XNOR
    }
    output[n] = (accum << 1) - M;
  }
}

#endif
