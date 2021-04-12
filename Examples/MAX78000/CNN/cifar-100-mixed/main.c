/*******************************************************************************
* Copyright (C) Maxim Integrated Products, Inc., All rights Reserved.
*
* This software is protected by copyright laws of the United States and
* of foreign countries. This material may also be protected by patent laws
* and technology transfer regulations of the United States and of foreign
* countries. This software is furnished under a license agreement and/or a
* nondisclosure agreement and may only be used or reproduced in accordance
* with the terms of those agreements. Dissemination of this information to
* any party or parties not specified in the license agreement and/or
* nondisclosure agreement is expressly prohibited.
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
* OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*
* Except as contained in this notice, the name of Maxim Integrated
* Products, Inc. shall not be used except as stated in the Maxim Integrated
* Products, Inc. Branding Policy.
*
* The mere transfer of this software does not imply any licenses
* of trade secrets, proprietary technology, copyrights, patents,
* trademarks, maskwork rights, or any other form of intellectual
* property whatsoever. Maxim Integrated Products, Inc. retains all
* ownership rights.
*******************************************************************************/

// cifar-100-mixed
// Created using ./ai8xize.py --verbose --log --test-dir sdk/Examples/MAX78000/CNN --prefix cifar-100-mixed --checkpoint-file trained/ai85-cifar100-qat-mixed-q.pth.tar --config-file networks/cifar100-simple.yaml --softmax --device MAX78000 --compact-data --mexpress --timer 0 --display-checkpoint --boost 2.5

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "sampledata.h"

volatile uint32_t cnn_time; // Stopwatch

void fail(void)
{
  printf("\n*** FAIL ***\n\n");
  while (1);
}

// 3-channel 32x32 data input (3072 bytes total / 1024 bytes per channel):
// HWC 32x32, channels 0 to 2
static const uint32_t input_0[] = SAMPLE_INPUT_0;

void load_input(void)
{
  // This function loads the sample data input -- replace with actual data

  memcpy32((uint32_t *) 0x50400000, input_0, 1024);
}

// Expected output of layer 13 for cifar-100-mixed given the sample input
int check_output(void)
{
  if ((*((volatile uint32_t *) 0x50400000)) != 0x000035cf) return CNN_FAIL; // 0,0,0
  if ((*((volatile uint32_t *) 0x50400004)) != 0xfffffb43) return CNN_FAIL; // 0,0,1
  if ((*((volatile uint32_t *) 0x50400008)) != 0xfffffb13) return CNN_FAIL; // 0,0,2
  if ((*((volatile uint32_t *) 0x5040000c)) != 0xffffe062) return CNN_FAIL; // 0,0,3
  if ((*((volatile uint32_t *) 0x50408000)) != 0xffffda2c) return CNN_FAIL; // 0,0,4
  if ((*((volatile uint32_t *) 0x50408004)) != 0xffffffa2) return CNN_FAIL; // 0,0,5
  if ((*((volatile uint32_t *) 0x50408008)) != 0xffffe51c) return CNN_FAIL; // 0,0,6
  if ((*((volatile uint32_t *) 0x5040800c)) != 0xfffff290) return CNN_FAIL; // 0,0,7
  if ((*((volatile uint32_t *) 0x50410000)) != 0xffffdecb) return CNN_FAIL; // 0,0,8
  if ((*((volatile uint32_t *) 0x50410004)) != 0x00000944) return CNN_FAIL; // 0,0,9
  if ((*((volatile uint32_t *) 0x50410008)) != 0x00000646) return CNN_FAIL; // 0,0,10
  if ((*((volatile uint32_t *) 0x5041000c)) != 0xfffff718) return CNN_FAIL; // 0,0,11
  if ((*((volatile uint32_t *) 0x50418000)) != 0xffffd776) return CNN_FAIL; // 0,0,12
  if ((*((volatile uint32_t *) 0x50418004)) != 0x00000074) return CNN_FAIL; // 0,0,13
  if ((*((volatile uint32_t *) 0x50418008)) != 0xfffffcd9) return CNN_FAIL; // 0,0,14
  if ((*((volatile uint32_t *) 0x5041800c)) != 0xfffff044) return CNN_FAIL; // 0,0,15
  if ((*((volatile uint32_t *) 0x50800000)) != 0xfffffa70) return CNN_FAIL; // 0,0,16
  if ((*((volatile uint32_t *) 0x50800004)) != 0xffffe083) return CNN_FAIL; // 0,0,17
  if ((*((volatile uint32_t *) 0x50800008)) != 0xfffff605) return CNN_FAIL; // 0,0,18
  if ((*((volatile uint32_t *) 0x5080000c)) != 0xffffea8f) return CNN_FAIL; // 0,0,19
  if ((*((volatile uint32_t *) 0x50808000)) != 0xffffeba1) return CNN_FAIL; // 0,0,20
  if ((*((volatile uint32_t *) 0x50808004)) != 0xffffd3a1) return CNN_FAIL; // 0,0,21
  if ((*((volatile uint32_t *) 0x50808008)) != 0xfffff70d) return CNN_FAIL; // 0,0,22
  if ((*((volatile uint32_t *) 0x5080800c)) != 0xffffe1cb) return CNN_FAIL; // 0,0,23
  if ((*((volatile uint32_t *) 0x50810000)) != 0xffffe4dd) return CNN_FAIL; // 0,0,24
  if ((*((volatile uint32_t *) 0x50810004)) != 0x00000dfb) return CNN_FAIL; // 0,0,25
  if ((*((volatile uint32_t *) 0x50810008)) != 0xfffff185) return CNN_FAIL; // 0,0,26
  if ((*((volatile uint32_t *) 0x5081000c)) != 0xffffd1f7) return CNN_FAIL; // 0,0,27
  if ((*((volatile uint32_t *) 0x50818000)) != 0xfffff632) return CNN_FAIL; // 0,0,28
  if ((*((volatile uint32_t *) 0x50818004)) != 0xffffe66c) return CNN_FAIL; // 0,0,29
  if ((*((volatile uint32_t *) 0x50818008)) != 0xffffaa7e) return CNN_FAIL; // 0,0,30
  if ((*((volatile uint32_t *) 0x5081800c)) != 0xffffd6b0) return CNN_FAIL; // 0,0,31
  if ((*((volatile uint32_t *) 0x50c00000)) != 0x000003ae) return CNN_FAIL; // 0,0,32
  if ((*((volatile uint32_t *) 0x50c00004)) != 0xfffff3b5) return CNN_FAIL; // 0,0,33
  if ((*((volatile uint32_t *) 0x50c00008)) != 0xffffda8b) return CNN_FAIL; // 0,0,34
  if ((*((volatile uint32_t *) 0x50c0000c)) != 0xfffff62d) return CNN_FAIL; // 0,0,35
  if ((*((volatile uint32_t *) 0x50c08000)) != 0xfffff0d4) return CNN_FAIL; // 0,0,36
  if ((*((volatile uint32_t *) 0x50c08004)) != 0xffffeb44) return CNN_FAIL; // 0,0,37
  if ((*((volatile uint32_t *) 0x50c08008)) != 0xffffc8af) return CNN_FAIL; // 0,0,38
  if ((*((volatile uint32_t *) 0x50c0800c)) != 0xffffe8c1) return CNN_FAIL; // 0,0,39
  if ((*((volatile uint32_t *) 0x50c10000)) != 0xfffff1d9) return CNN_FAIL; // 0,0,40
  if ((*((volatile uint32_t *) 0x50c10004)) != 0x00000032) return CNN_FAIL; // 0,0,41
  if ((*((volatile uint32_t *) 0x50c10008)) != 0xffffcccb) return CNN_FAIL; // 0,0,42
  if ((*((volatile uint32_t *) 0x50c1000c)) != 0xffffd161) return CNN_FAIL; // 0,0,43
  if ((*((volatile uint32_t *) 0x50c18000)) != 0xffffe9f1) return CNN_FAIL; // 0,0,44
  if ((*((volatile uint32_t *) 0x50c18004)) != 0xfffff391) return CNN_FAIL; // 0,0,45
  if ((*((volatile uint32_t *) 0x50c18008)) != 0xffffef6a) return CNN_FAIL; // 0,0,46
  if ((*((volatile uint32_t *) 0x50c1800c)) != 0xfffffa9a) return CNN_FAIL; // 0,0,47
  if ((*((volatile uint32_t *) 0x51000000)) != 0xffffdc45) return CNN_FAIL; // 0,0,48
  if ((*((volatile uint32_t *) 0x51000004)) != 0xffffd3a3) return CNN_FAIL; // 0,0,49
  if ((*((volatile uint32_t *) 0x51000008)) != 0xffffdba5) return CNN_FAIL; // 0,0,50
  if ((*((volatile uint32_t *) 0x5100000c)) != 0xffffff91) return CNN_FAIL; // 0,0,51
  if ((*((volatile uint32_t *) 0x50400010)) != 0xffffefd5) return CNN_FAIL; // 0,0,52
  if ((*((volatile uint32_t *) 0x50400014)) != 0x00001712) return CNN_FAIL; // 0,0,53
  if ((*((volatile uint32_t *) 0x50400018)) != 0xfffff200) return CNN_FAIL; // 0,0,54
  if ((*((volatile uint32_t *) 0x5040001c)) != 0xffffdf9e) return CNN_FAIL; // 0,0,55
  if ((*((volatile uint32_t *) 0x50408010)) != 0xffffded5) return CNN_FAIL; // 0,0,56
  if ((*((volatile uint32_t *) 0x50408014)) != 0x00002730) return CNN_FAIL; // 0,0,57
  if ((*((volatile uint32_t *) 0x50408018)) != 0x00000cc5) return CNN_FAIL; // 0,0,58
  if ((*((volatile uint32_t *) 0x5040801c)) != 0xffffed8b) return CNN_FAIL; // 0,0,59
  if ((*((volatile uint32_t *) 0x50410010)) != 0xffffec1e) return CNN_FAIL; // 0,0,60
  if ((*((volatile uint32_t *) 0x50410014)) != 0xfffff466) return CNN_FAIL; // 0,0,61
  if ((*((volatile uint32_t *) 0x50410018)) != 0xfffffba7) return CNN_FAIL; // 0,0,62
  if ((*((volatile uint32_t *) 0x5041001c)) != 0xffffe367) return CNN_FAIL; // 0,0,63
  if ((*((volatile uint32_t *) 0x50418010)) != 0xffffd9dd) return CNN_FAIL; // 0,0,64
  if ((*((volatile uint32_t *) 0x50418014)) != 0xfffff535) return CNN_FAIL; // 0,0,65
  if ((*((volatile uint32_t *) 0x50418018)) != 0xffffc4f7) return CNN_FAIL; // 0,0,66
  if ((*((volatile uint32_t *) 0x5041801c)) != 0xffffefda) return CNN_FAIL; // 0,0,67
  if ((*((volatile uint32_t *) 0x50800010)) != 0xffffd881) return CNN_FAIL; // 0,0,68
  if ((*((volatile uint32_t *) 0x50800014)) != 0xffffedda) return CNN_FAIL; // 0,0,69
  if ((*((volatile uint32_t *) 0x50800018)) != 0x00000460) return CNN_FAIL; // 0,0,70
  if ((*((volatile uint32_t *) 0x5080001c)) != 0xffffe12e) return CNN_FAIL; // 0,0,71
  if ((*((volatile uint32_t *) 0x50808010)) != 0xffffe282) return CNN_FAIL; // 0,0,72
  if ((*((volatile uint32_t *) 0x50808014)) != 0xffffdb7b) return CNN_FAIL; // 0,0,73
  if ((*((volatile uint32_t *) 0x50808018)) != 0xffffd621) return CNN_FAIL; // 0,0,74
  if ((*((volatile uint32_t *) 0x5080801c)) != 0xffffd0ff) return CNN_FAIL; // 0,0,75
  if ((*((volatile uint32_t *) 0x50810010)) != 0xffffd1d3) return CNN_FAIL; // 0,0,76
  if ((*((volatile uint32_t *) 0x50810014)) != 0xfffffc2f) return CNN_FAIL; // 0,0,77
  if ((*((volatile uint32_t *) 0x50810018)) != 0xffffe7b3) return CNN_FAIL; // 0,0,78
  if ((*((volatile uint32_t *) 0x5081001c)) != 0xffffdbd7) return CNN_FAIL; // 0,0,79
  if ((*((volatile uint32_t *) 0x50818010)) != 0xffffe706) return CNN_FAIL; // 0,0,80
  if ((*((volatile uint32_t *) 0x50818014)) != 0xfffff7b2) return CNN_FAIL; // 0,0,81
  if ((*((volatile uint32_t *) 0x50818018)) != 0xffffec89) return CNN_FAIL; // 0,0,82
  if ((*((volatile uint32_t *) 0x5081801c)) != 0x00002fb9) return CNN_FAIL; // 0,0,83
  if ((*((volatile uint32_t *) 0x50c00010)) != 0xfffff6f7) return CNN_FAIL; // 0,0,84
  if ((*((volatile uint32_t *) 0x50c00014)) != 0xfffff029) return CNN_FAIL; // 0,0,85
  if ((*((volatile uint32_t *) 0x50c00018)) != 0xffffe23f) return CNN_FAIL; // 0,0,86
  if ((*((volatile uint32_t *) 0x50c0001c)) != 0xffffef3b) return CNN_FAIL; // 0,0,87
  if ((*((volatile uint32_t *) 0x50c08010)) != 0xffffdbc1) return CNN_FAIL; // 0,0,88
  if ((*((volatile uint32_t *) 0x50c08014)) != 0x0000078a) return CNN_FAIL; // 0,0,89
  if ((*((volatile uint32_t *) 0x50c08018)) != 0xfffff627) return CNN_FAIL; // 0,0,90
  if ((*((volatile uint32_t *) 0x50c0801c)) != 0xffffee1a) return CNN_FAIL; // 0,0,91
  if ((*((volatile uint32_t *) 0x50c10010)) != 0x00000adb) return CNN_FAIL; // 0,0,92
  if ((*((volatile uint32_t *) 0x50c10014)) != 0xffffe080) return CNN_FAIL; // 0,0,93
  if ((*((volatile uint32_t *) 0x50c10018)) != 0xfffff50d) return CNN_FAIL; // 0,0,94
  if ((*((volatile uint32_t *) 0x50c1001c)) != 0xffffc670) return CNN_FAIL; // 0,0,95
  if ((*((volatile uint32_t *) 0x50c18010)) != 0xfffff508) return CNN_FAIL; // 0,0,96
  if ((*((volatile uint32_t *) 0x50c18014)) != 0xffffc60e) return CNN_FAIL; // 0,0,97
  if ((*((volatile uint32_t *) 0x50c18018)) != 0xfffff57f) return CNN_FAIL; // 0,0,98
  if ((*((volatile uint32_t *) 0x50c1801c)) != 0xffffe971) return CNN_FAIL; // 0,0,99

  return CNN_OK;
}

// Classification layer:
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

void softmax_layer(void)
{
  cnn_unload((uint32_t *) ml_data);
  softmax_q17p14_q15((const q31_t *) ml_data, CNN_NUM_OUTPUTS, ml_softmax);
}

int main(void)
{
  int i;
  int digs, tens;

  MXC_ICC_Enable(MXC_ICC0); // Enable cache

  // Switch to 100 MHz clock
  MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
  SystemCoreClockUpdate();

  printf("Waiting...\n");

  // DO NOT DELETE THIS LINE:
  MXC_Delay(SEC(2)); // Let debugger interrupt if needed

  // Enable peripheral, enable CNN interrupt, turn on CNN clock
  // CNN clock: 50 MHz div 1
  cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
  cnn_boost_enable(MXC_GPIO2, MXC_GPIO_PIN_5); // Turn on the boost circuit

  printf("\n*** CNN Inference Test ***\n");

  cnn_init(); // Bring state machine into consistent state
  cnn_load_weights(); // Load kernels
  // cnn_load_bias(); // Not used in this network
  cnn_configure(); // Configure state machine
  load_input(); // Load data input
  cnn_start(); // Start CNN processing

  while (cnn_time == 0)
    __WFI(); // Wait for CNN

  cnn_boost_disable(MXC_GPIO2, MXC_GPIO_PIN_5); // Turn off the boost circuit

  if (check_output() != CNN_OK) fail();
  softmax_layer();

  printf("\n*** PASS ***\n\n");

#ifdef CNN_INFERENCE_TIMER
  printf("Approximate inference time: %u us\n\n", cnn_time);
#endif

  cnn_disable(); // Shut down CNN clock, disable peripheral

  printf("Classification results:\n");
  for (i = 0; i < CNN_NUM_OUTPUTS; i++) {
    digs = (1000 * ml_softmax[i] + 0x4000) >> 15;
    tens = digs % 10;
    digs = digs / 10;
    printf("[%7d] -> Class %d: %d.%d%%\n", ml_data[i], i, digs, tens);
  }

  return 0;
}

/*
  SUMMARY OF OPS
  Hardware: 18,607,744 ops (18,461,184 macc; 146,560 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 165,228 bytes out of 442,368 bytes total (37%)
  Bias memory:   0 bytes out of 2,048 bytes total (0%)
*/

