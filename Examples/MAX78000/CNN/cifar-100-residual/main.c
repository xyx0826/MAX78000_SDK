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

// cifar-100-residual
// Created using ./ai8xize.py --verbose --log --test-dir sdk/Examples/MAX78000/CNN --prefix cifar-100-residual --checkpoint-file trained/ai85-cifar100-residual-qat8-q.pth.tar --config-file networks/cifar100-ressimplenet.yaml --softmax --device MAX78000 --compact-data --mexpress --timer 0 --display-checkpoint --boost 2.5

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
static const uint32_t input_60[] = SAMPLE_INPUT_60;

void load_input(void)
{
  // This function loads the sample data input -- replace with actual data

  memcpy32((uint32_t *) 0x51018000, input_60, 1024);
}

// Expected output of layer 16 for cifar-100-residual given the sample input
int check_output(void)
{
  if ((*((volatile uint32_t *) 0x50400000)) != 0x00045c97) return CNN_FAIL; // 0,0,0
  if ((*((volatile uint32_t *) 0x50400004)) != 0xffff50dc) return CNN_FAIL; // 0,0,1
  if ((*((volatile uint32_t *) 0x50400008)) != 0xffffd94b) return CNN_FAIL; // 0,0,2
  if ((*((volatile uint32_t *) 0x5040000c)) != 0xfffeb72a) return CNN_FAIL; // 0,0,3
  if ((*((volatile uint32_t *) 0x50408000)) != 0xfffe61dc) return CNN_FAIL; // 0,0,4
  if ((*((volatile uint32_t *) 0x50408004)) != 0x00006068) return CNN_FAIL; // 0,0,5
  if ((*((volatile uint32_t *) 0x50408008)) != 0xfffe20c8) return CNN_FAIL; // 0,0,6
  if ((*((volatile uint32_t *) 0x5040800c)) != 0xffff4c4a) return CNN_FAIL; // 0,0,7
  if ((*((volatile uint32_t *) 0x50410000)) != 0xfffd8b6c) return CNN_FAIL; // 0,0,8
  if ((*((volatile uint32_t *) 0x50410004)) != 0x00016986) return CNN_FAIL; // 0,0,9
  if ((*((volatile uint32_t *) 0x50410008)) != 0x00017012) return CNN_FAIL; // 0,0,10
  if ((*((volatile uint32_t *) 0x5041000c)) != 0xffff4b0e) return CNN_FAIL; // 0,0,11
  if ((*((volatile uint32_t *) 0x50418000)) != 0xfffdc9b6) return CNN_FAIL; // 0,0,12
  if ((*((volatile uint32_t *) 0x50418004)) != 0xffff301c) return CNN_FAIL; // 0,0,13
  if ((*((volatile uint32_t *) 0x50418008)) != 0xffffb5d8) return CNN_FAIL; // 0,0,14
  if ((*((volatile uint32_t *) 0x5041800c)) != 0xffff8951) return CNN_FAIL; // 0,0,15
  if ((*((volatile uint32_t *) 0x50800000)) != 0x0000476a) return CNN_FAIL; // 0,0,16
  if ((*((volatile uint32_t *) 0x50800004)) != 0xfffe8cf2) return CNN_FAIL; // 0,0,17
  if ((*((volatile uint32_t *) 0x50800008)) != 0xffff33b5) return CNN_FAIL; // 0,0,18
  if ((*((volatile uint32_t *) 0x5080000c)) != 0xffff108e) return CNN_FAIL; // 0,0,19
  if ((*((volatile uint32_t *) 0x50808000)) != 0xffff89e2) return CNN_FAIL; // 0,0,20
  if ((*((volatile uint32_t *) 0x50808004)) != 0xfffe87ff) return CNN_FAIL; // 0,0,21
  if ((*((volatile uint32_t *) 0x50808008)) != 0x000043cb) return CNN_FAIL; // 0,0,22
  if ((*((volatile uint32_t *) 0x5080800c)) != 0xffff8baa) return CNN_FAIL; // 0,0,23
  if ((*((volatile uint32_t *) 0x50810000)) != 0xfffec23b) return CNN_FAIL; // 0,0,24
  if ((*((volatile uint32_t *) 0x50810004)) != 0x000133c2) return CNN_FAIL; // 0,0,25
  if ((*((volatile uint32_t *) 0x50810008)) != 0xfffeacf2) return CNN_FAIL; // 0,0,26
  if ((*((volatile uint32_t *) 0x5081000c)) != 0xfffe0ec9) return CNN_FAIL; // 0,0,27
  if ((*((volatile uint32_t *) 0x50818000)) != 0x0000183f) return CNN_FAIL; // 0,0,28
  if ((*((volatile uint32_t *) 0x50818004)) != 0xffff4e73) return CNN_FAIL; // 0,0,29
  if ((*((volatile uint32_t *) 0x50818008)) != 0xfffcb0ea) return CNN_FAIL; // 0,0,30
  if ((*((volatile uint32_t *) 0x5081800c)) != 0xfffe1f34) return CNN_FAIL; // 0,0,31
  if ((*((volatile uint32_t *) 0x50c00000)) != 0x00003692) return CNN_FAIL; // 0,0,32
  if ((*((volatile uint32_t *) 0x50c00004)) != 0x00000e3a) return CNN_FAIL; // 0,0,33
  if ((*((volatile uint32_t *) 0x50c00008)) != 0xfffe0f44) return CNN_FAIL; // 0,0,34
  if ((*((volatile uint32_t *) 0x50c0000c)) != 0xffff19b1) return CNN_FAIL; // 0,0,35
  if ((*((volatile uint32_t *) 0x50c08000)) != 0xffffb972) return CNN_FAIL; // 0,0,36
  if ((*((volatile uint32_t *) 0x50c08004)) != 0xfffec431) return CNN_FAIL; // 0,0,37
  if ((*((volatile uint32_t *) 0x50c08008)) != 0xfffdc38f) return CNN_FAIL; // 0,0,38
  if ((*((volatile uint32_t *) 0x50c0800c)) != 0xffffe2ab) return CNN_FAIL; // 0,0,39
  if ((*((volatile uint32_t *) 0x50c10000)) != 0xffffe83a) return CNN_FAIL; // 0,0,40
  if ((*((volatile uint32_t *) 0x50c10004)) != 0xffff1ae7) return CNN_FAIL; // 0,0,41
  if ((*((volatile uint32_t *) 0x50c10008)) != 0xfffd6497) return CNN_FAIL; // 0,0,42
  if ((*((volatile uint32_t *) 0x50c1000c)) != 0xfffec44b) return CNN_FAIL; // 0,0,43
  if ((*((volatile uint32_t *) 0x50c18000)) != 0xfffee832) return CNN_FAIL; // 0,0,44
  if ((*((volatile uint32_t *) 0x50c18004)) != 0xffffb304) return CNN_FAIL; // 0,0,45
  if ((*((volatile uint32_t *) 0x50c18008)) != 0xfffe799a) return CNN_FAIL; // 0,0,46
  if ((*((volatile uint32_t *) 0x50c1800c)) != 0x0001090d) return CNN_FAIL; // 0,0,47
  if ((*((volatile uint32_t *) 0x51000000)) != 0xfffdf22b) return CNN_FAIL; // 0,0,48
  if ((*((volatile uint32_t *) 0x51000004)) != 0xfffdf5d6) return CNN_FAIL; // 0,0,49
  if ((*((volatile uint32_t *) 0x51000008)) != 0xffff96b2) return CNN_FAIL; // 0,0,50
  if ((*((volatile uint32_t *) 0x5100000c)) != 0x00010abe) return CNN_FAIL; // 0,0,51
  if ((*((volatile uint32_t *) 0x50400010)) != 0xffff14b7) return CNN_FAIL; // 0,0,52
  if ((*((volatile uint32_t *) 0x50400014)) != 0x00020dd6) return CNN_FAIL; // 0,0,53
  if ((*((volatile uint32_t *) 0x50400018)) != 0x00000923) return CNN_FAIL; // 0,0,54
  if ((*((volatile uint32_t *) 0x5040001c)) != 0xffff1f44) return CNN_FAIL; // 0,0,55
  if ((*((volatile uint32_t *) 0x50408010)) != 0xffff75c1) return CNN_FAIL; // 0,0,56
  if ((*((volatile uint32_t *) 0x50408014)) != 0x000333d1) return CNN_FAIL; // 0,0,57
  if ((*((volatile uint32_t *) 0x50408018)) != 0xfffed669) return CNN_FAIL; // 0,0,58
  if ((*((volatile uint32_t *) 0x5040801c)) != 0x00003338) return CNN_FAIL; // 0,0,59
  if ((*((volatile uint32_t *) 0x50410010)) != 0x00000809) return CNN_FAIL; // 0,0,60
  if ((*((volatile uint32_t *) 0x50410014)) != 0x0000774a) return CNN_FAIL; // 0,0,61
  if ((*((volatile uint32_t *) 0x50410018)) != 0xfffffc45) return CNN_FAIL; // 0,0,62
  if ((*((volatile uint32_t *) 0x5041001c)) != 0xfffedf67) return CNN_FAIL; // 0,0,63
  if ((*((volatile uint32_t *) 0x50418010)) != 0xfffe9f3f) return CNN_FAIL; // 0,0,64
  if ((*((volatile uint32_t *) 0x50418014)) != 0x000014ce) return CNN_FAIL; // 0,0,65
  if ((*((volatile uint32_t *) 0x50418018)) != 0xfffd41bb) return CNN_FAIL; // 0,0,66
  if ((*((volatile uint32_t *) 0x5041801c)) != 0xffff6e1c) return CNN_FAIL; // 0,0,67
  if ((*((volatile uint32_t *) 0x50800010)) != 0xfffe2113) return CNN_FAIL; // 0,0,68
  if ((*((volatile uint32_t *) 0x50800014)) != 0xffff0c0e) return CNN_FAIL; // 0,0,69
  if ((*((volatile uint32_t *) 0x50800018)) != 0x0001235e) return CNN_FAIL; // 0,0,70
  if ((*((volatile uint32_t *) 0x5080001c)) != 0xfffebc38) return CNN_FAIL; // 0,0,71
  if ((*((volatile uint32_t *) 0x50808010)) != 0xffff3d90) return CNN_FAIL; // 0,0,72
  if ((*((volatile uint32_t *) 0x50808014)) != 0xfffdcf31) return CNN_FAIL; // 0,0,73
  if ((*((volatile uint32_t *) 0x50808018)) != 0xfffedf99) return CNN_FAIL; // 0,0,74
  if ((*((volatile uint32_t *) 0x5080801c)) != 0xfffd6c3f) return CNN_FAIL; // 0,0,75
  if ((*((volatile uint32_t *) 0x50810010)) != 0xfffec72b) return CNN_FAIL; // 0,0,76
  if ((*((volatile uint32_t *) 0x50810014)) != 0x00012dfb) return CNN_FAIL; // 0,0,77
  if ((*((volatile uint32_t *) 0x50810018)) != 0xffff57c5) return CNN_FAIL; // 0,0,78
  if ((*((volatile uint32_t *) 0x5081001c)) != 0xfffece62) return CNN_FAIL; // 0,0,79
  if ((*((volatile uint32_t *) 0x50818010)) != 0xffff6656) return CNN_FAIL; // 0,0,80
  if ((*((volatile uint32_t *) 0x50818014)) != 0xffff1aa9) return CNN_FAIL; // 0,0,81
  if ((*((volatile uint32_t *) 0x50818018)) != 0xffff766e) return CNN_FAIL; // 0,0,82
  if ((*((volatile uint32_t *) 0x5081801c)) != 0x00030f59) return CNN_FAIL; // 0,0,83
  if ((*((volatile uint32_t *) 0x50c00010)) != 0x0000a622) return CNN_FAIL; // 0,0,84
  if ((*((volatile uint32_t *) 0x50c00014)) != 0xffffed29) return CNN_FAIL; // 0,0,85
  if ((*((volatile uint32_t *) 0x50c00018)) != 0xffff3e08) return CNN_FAIL; // 0,0,86
  if ((*((volatile uint32_t *) 0x50c0001c)) != 0xfffff18d) return CNN_FAIL; // 0,0,87
  if ((*((volatile uint32_t *) 0x50c08010)) != 0xffffaa76) return CNN_FAIL; // 0,0,88
  if ((*((volatile uint32_t *) 0x50c08014)) != 0xffff8af6) return CNN_FAIL; // 0,0,89
  if ((*((volatile uint32_t *) 0x50c08018)) != 0xfffed64f) return CNN_FAIL; // 0,0,90
  if ((*((volatile uint32_t *) 0x50c0801c)) != 0xffff101d) return CNN_FAIL; // 0,0,91
  if ((*((volatile uint32_t *) 0x50c10010)) != 0x000199ec) return CNN_FAIL; // 0,0,92
  if ((*((volatile uint32_t *) 0x50c10014)) != 0xfffde94a) return CNN_FAIL; // 0,0,93
  if ((*((volatile uint32_t *) 0x50c10018)) != 0x0000553e) return CNN_FAIL; // 0,0,94
  if ((*((volatile uint32_t *) 0x50c1001c)) != 0xfffd13b9) return CNN_FAIL; // 0,0,95
  if ((*((volatile uint32_t *) 0x50c18010)) != 0x00012af6) return CNN_FAIL; // 0,0,96
  if ((*((volatile uint32_t *) 0x50c18014)) != 0xfffe47a4) return CNN_FAIL; // 0,0,97
  if ((*((volatile uint32_t *) 0x50c18018)) != 0xffff511c) return CNN_FAIL; // 0,0,98
  if ((*((volatile uint32_t *) 0x50c1801c)) != 0xffffe876) return CNN_FAIL; // 0,0,99

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
  Hardware: 18,636,416 ops (18,461,184 macc; 146,560 comp; 28,672 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 381,792 bytes out of 442,368 bytes total (86%)
  Bias memory:   0 bytes out of 2,048 bytes total (0%)
*/

