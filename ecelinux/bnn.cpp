//==========================================================================
// bnn.cpp
//==========================================================================
// @brief: A convolution kernel for CNN on digit recognition

#include "bnn.h"
#include "layer.h"
#include "model.h"
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

//----------------------------------------------------------
// Top function
//----------------------------------------------------------

void dut(hls::stream<bit32_t> &strm_in, hls::stream<bit32_t> &strm_out) {
  bit input[1][I_WIDTH1][I_WIDTH1];
  bit32_t input_l;
  bit32_t output;

  // read one test image into digit
  int bitcount = 0;
  for (int i = 0; i < I_WIDTH1 * I_WIDTH1 / BUS_WIDTH; i++) {
    input_l = strm_in.read();
    for (int j = 0; j < BUS_WIDTH; j++) {
      input[0][bitcount / I_WIDTH1][bitcount % I_WIDTH1] = input_l(j, j);
      bitcount++;
    }
  }
  // call bnn
  output = bnn_xcel(input);

  // write out the result
  strm_out.write(output);
}

//----------------------------------------------------------
// BNN Accelerator
//----------------------------------------------------------
// @param[in] : input - the testing instance
// @return : the predicted digit

bit32_t bnn_xcel(bit input[1][I_WIDTH1][I_WIDTH1]) {
  bit input_padded[I_CHANNEL1][I_WIDTH1 + F_PAD][I_WIDTH1 + F_PAD];

  // INFO: [HLS 200-42] -- Implementing module 'initialize_padded_me_1'
  initialize_padded_memory<
    I_CHANNEL1,               // M
    I_WIDTH1 + F_PAD,         // I
    1                         // C
  >(
    input_padded              // input[M][I][I]
  );
  
  bit conv1[O_CHANNEL1][I_WIDTH1][I_WIDTH1];
  bit conv1_pooled[O_CHANNEL1][I_WIDTH2][I_WIDTH2];
  bit conv1_pooled_padded[O_CHANNEL1][I_WIDTH2 + F_PAD][I_WIDTH2 + F_PAD];

  // INFO: [HLS 200-42] -- Implementing module 'initialize_padded_me'
  initialize_padded_memory<
    O_CHANNEL1,               // M
    I_WIDTH2 + F_PAD,         // I
    0                         // C
  >(
    conv1_pooled_padded       // input[M][I][I]
  );

  bit conv2[O_CHANNEL2][I_WIDTH2][I_WIDTH2];
  bit conv2_pooled[O_CHANNEL2][O_WIDTH][O_WIDTH];

  bit reshaped[I_UNITS1];
  bit16_t dense1[I_UNITS2];
  bit signed1[I_UNITS2];
  bit16_t dense2[NUM_DIGITS];
  bit32_t output;

  /* First Conv Layer */
  // INFO: [HLS 200-42] -- Implementing module 'pad_1_16_s'
  pad<
    I_CHANNEL1,               // M
    I_WIDTH1                  // I
  >(
    input,                    // input[M][I][I]
    input_padded              // output[M][I+F_PAD][I+F_PAD]  (see model.h)
  );
  
  // INFO: [HLS 200-42] -- Implementing module 'conv_1_16_18_s'
  conv<
    I_CHANNEL1,               // M
    O_CHANNEL1,               // N
    I_WIDTH1 + F_PAD,         // I
    1                         // T
  >(
    input_padded,             // input[M][I][I]
    conv1,                    // output[N][I - F + 1][I - F + 1]
    threshold_conv1,          // threshold[N] 
    w_conv1                   // weights[M][N][F][F]
  );

  // INFO: [HLS 200-42] -- Implementing module 'max_pool_16_16_s' 
  max_pool<
    O_CHANNEL1,               // M
    I_WIDTH1                  // I 
  >(
    conv1,                    // input[M][I][I]
    conv1_pooled              // output[M][I/2][I/2]
  );

  /* Second Conv Layer */
  // INFO: [HLS 200-42] -- Implementing module 'pad_16_8_s' 
  pad<
    O_CHANNEL1,               // M
    I_WIDTH2                  // I
  >(
    conv1_pooled,             // input[M][I][I]
    conv1_pooled_padded       // output[M][I+F_PAD][I+F_PAD]
  );

  // INFO: [HLS 200-42] -- Implementing module 'conv_16_32_10_s'
  conv<
    O_CHANNEL1,               // M
    O_CHANNEL2,               // N
    I_WIDTH2 + F_PAD,         // I
    4                         // T
  >(
    conv1_pooled_padded,      // input[M][I][I]
    conv2,                    // output[N][I - F + 1][I - F + 1] 
    threshold_conv2,          // threshold[N]
    w_conv2                   // weights[M][N][F][F] 
  );

  // INFO: [HLS 200-10] -- Generating RTL for module 'max_pool_32_8_s'
  max_pool<
    O_CHANNEL2,               // M
    I_WIDTH2                  // I
  >(
    conv2,                    // input[M][I][I]
    conv2_pooled              // output[M][I/2][I/2]
  );

  // INFO: [HLS 200-10] -- Generating RTL for module 'flatten'
  flatten(
    conv2_pooled,             // input[O_CHANNEL2][O_WIDTH][O_WIDTH]
    reshaped                  // output[I_UNITS1]
  );

  /* Dense Layers */
  // INFO: [HLS 200-10] -- Generating RTL for module 'dense_512_256_s'
  dense<
    I_UNITS1,                 // M
    I_UNITS2                  // N
  >(
    reshaped,                 // input[M]
    dense1,                   // output[N]
    w_fc1                     // weights[M][N]
  );

  // INFO: [HLS 200-10] -- Generating RTL for module 'sign_256_s'
  sign<
    I_UNITS2                  // M
  >(
    dense1,                   // input[M] 
    signed1                   // output[M]  
  );

  // INFO: [HLS 200-10] -- Generating RTL for module 'dense_256_10_s'
  dense<
    I_UNITS2,                 // M  
    NUM_DIGITS                // N 
  >(
    signed1,                  // input[M]
    dense2,                   // output[N]
    w_fc2                     // weights[M][N]
  );

  // INFO: [HLS 200-10] -- Generating RTL for module 'argmax'
  output = argmax(dense2);

  return output;
}


void dense_layer_2(
  bit input[I_UNITS1], 
  bit16_t output[I_UNITS2]
) {
  dense<
    I_UNITS1,                 // M  
    I_UNITS2                  // N 
  >(
    input,                    // input[M]
    output,                   // output[N]
    w_fc1                     // weights[M][N]
  );
}

void conv_layer_2(
  bit input[O_CHANNEL1][I_WIDTH2 + F_PAD][I_WIDTH2 + F_PAD], 
  bit output[O_CHANNEL2][I_WIDTH2 + F_PAD - F + 1][I_WIDTH2 + F_PAD - F + 1]
) {
  conv<
    O_CHANNEL1,               // M
    O_CHANNEL2,               // N
    I_WIDTH2 + F_PAD,         // I
    4
  >(
    input,                    // input[M][I][I]
    output,                   // output[N][I - F + 1][I - F + 1]
    threshold_conv2,          // threshold[N]
    w_conv2                   // weights[M][N][F][F] 
  );
}

void conv_layer_1(
  bit input[1][I_WIDTH1 + F_PAD][I_WIDTH1 + F_PAD], 
  bit output[O_CHANNEL1][I_WIDTH1 + F_PAD - F + 1][I_WIDTH1 + F_PAD - F + 1]
) {
  conv<
    I_CHANNEL1,               // M
    O_CHANNEL1,               // N
    I_WIDTH1 + F_PAD,         // I
    1                         // T
  >(
    input,                    // input[M][I][I]
    output,                   // output[N][I - F + 1][I - F + 1]
    threshold_conv1,          // threshold[N] 
    w_conv1                   // weights[M][N][F][F]
  );
}

void flatten_layer(
  bit input[O_CHANNEL2][O_WIDTH][O_WIDTH], 
  bit output[I_UNITS1]
) {
  flatten(
    input,                    // input[O_CHANNEL2][O_WIDTH][O_WIDTH]
    output                    // output[I_UNITS1]
  );
}