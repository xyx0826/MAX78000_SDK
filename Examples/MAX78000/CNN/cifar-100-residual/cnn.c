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

// DO NOT EDIT - regenerate this file instead!

// Configuring 17 layers:
// Layer 0: 3x32x32 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, 16x32x32 output
// Layer 1: 16x32x32 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, 20x32x32 output
// Layer 2: 20x32x32 (HWC data), no pooling, no convolution, 20x32x32 output
// Layer 3: 20x32x32 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, 20x32x32 output
// Layer 4: 2x20x32x32 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, 20x32x32 output
// Layer 5: 20x32x32 (HWC data), 2x2 max pool with stride 2/2, conv2d with kernel size 3x3, stride 1/1, pad 1/1, 20x16x16 output
// Layer 6: 20x16x16 (HWC data), no pooling, no convolution, 20x16x16 output
// Layer 7: 20x16x16 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, 20x16x16 output
// Layer 8: 2x20x16x16 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, 44x16x16 output
// Layer 9: 44x16x16 (HWC data), 2x2 max pool with stride 2/2, conv2d with kernel size 3x3, stride 1/1, pad 1/1, 48x8x8 output
// Layer 10: 48x8x8 (HWC data), no pooling, no convolution, 48x8x8 output
// Layer 11: 48x8x8 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, 48x8x8 output
// Layer 12: 2x48x8x8 (HWC data), 2x2 max pool with stride 2/2, conv2d with kernel size 3x3, stride 1/1, pad 1/1, 96x4x4 output
// Layer 13: 96x4x4 (HWC data), 2x2 max pool with stride 2/2, conv2d with kernel size 1x1, stride 1/1, pad 0/0, 512x2x2 output
// Layer 14: 512x2x2 (HWC data), no pooling, conv2d with kernel size 1x1, stride 1/1, pad 0/0, 128x2x2 output
// Layer 15: 128x2x2 (HWC data), 2x2 max pool with stride 2/2, conv2d with kernel size 3x3, stride 1/1, pad 1/1, 128x1x1 output
// Layer 16: 128x1x1 (HWC data), no pooling, conv2d with kernel size 1x1, stride 1/1, pad 0/0, 100x1x1 output

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "gcfr_regs.h"
#include "cnn.h"
#include "weights.h"

void CNN_ISR(void)
{
  // Acknowledge interrupt to all groups
  *((volatile uint32_t *) 0x50100000) &= ~((1<<12) | 1);
  *((volatile uint32_t *) 0x50500000) &= ~((1<<12) | 1);
  *((volatile uint32_t *) 0x50900000) &= ~((1<<12) | 1);
  *((volatile uint32_t *) 0x50d00000) &= ~((1<<12) | 1);

  CNN_COMPLETE; // Signal that processing is complete
#ifdef CNN_INFERENCE_TIMER
  cnn_time = MXC_TMR_SW_Stop(CNN_INFERENCE_TIMER);
#else
  cnn_time = 1;
#endif
}

int cnn_continue(void)
{
  cnn_time = 0;

  *((volatile uint32_t *) 0x50100000) |= 1; // Re-enable group 0

  return CNN_OK;
}

int cnn_stop(void)
{
  *((volatile uint32_t *) 0x50100000) &= ~1; // Disable group 0

  return CNN_OK;
}

void memcpy32(uint32_t *dst, const uint32_t *src, int n)
{
  while (n-- > 0) {
    *dst++ = *src++;
  }
}

// Kernels:
static const uint32_t kernels_0[] = KERNELS_0;
static const uint32_t kernels_1[] = KERNELS_1;
static const uint32_t kernels_2[] = KERNELS_2;
static const uint32_t kernels_3[] = KERNELS_3;
static const uint32_t kernels_4[] = KERNELS_4;
static const uint32_t kernels_5[] = KERNELS_5;
static const uint32_t kernels_6[] = KERNELS_6;
static const uint32_t kernels_7[] = KERNELS_7;
static const uint32_t kernels_8[] = KERNELS_8;
static const uint32_t kernels_9[] = KERNELS_9;
static const uint32_t kernels_10[] = KERNELS_10;
static const uint32_t kernels_11[] = KERNELS_11;
static const uint32_t kernels_12[] = KERNELS_12;
static const uint32_t kernels_13[] = KERNELS_13;
static const uint32_t kernels_14[] = KERNELS_14;
static const uint32_t kernels_15[] = KERNELS_15;
static const uint32_t kernels_16[] = KERNELS_16;
static const uint32_t kernels_17[] = KERNELS_17;
static const uint32_t kernels_18[] = KERNELS_18;
static const uint32_t kernels_19[] = KERNELS_19;
static const uint32_t kernels_20[] = KERNELS_20;
static const uint32_t kernels_21[] = KERNELS_21;
static const uint32_t kernels_22[] = KERNELS_22;
static const uint32_t kernels_23[] = KERNELS_23;
static const uint32_t kernels_24[] = KERNELS_24;
static const uint32_t kernels_25[] = KERNELS_25;
static const uint32_t kernels_26[] = KERNELS_26;
static const uint32_t kernels_27[] = KERNELS_27;
static const uint32_t kernels_28[] = KERNELS_28;
static const uint32_t kernels_29[] = KERNELS_29;
static const uint32_t kernels_30[] = KERNELS_30;
static const uint32_t kernels_31[] = KERNELS_31;
static const uint32_t kernels_32[] = KERNELS_32;
static const uint32_t kernels_33[] = KERNELS_33;
static const uint32_t kernels_34[] = KERNELS_34;
static const uint32_t kernels_35[] = KERNELS_35;
static const uint32_t kernels_36[] = KERNELS_36;
static const uint32_t kernels_37[] = KERNELS_37;
static const uint32_t kernels_38[] = KERNELS_38;
static const uint32_t kernels_39[] = KERNELS_39;
static const uint32_t kernels_40[] = KERNELS_40;
static const uint32_t kernels_41[] = KERNELS_41;
static const uint32_t kernels_42[] = KERNELS_42;
static const uint32_t kernels_43[] = KERNELS_43;
static const uint32_t kernels_44[] = KERNELS_44;
static const uint32_t kernels_45[] = KERNELS_45;
static const uint32_t kernels_46[] = KERNELS_46;
static const uint32_t kernels_47[] = KERNELS_47;
static const uint32_t kernels_48[] = KERNELS_48;
static const uint32_t kernels_49[] = KERNELS_49;
static const uint32_t kernels_50[] = KERNELS_50;
static const uint32_t kernels_51[] = KERNELS_51;
static const uint32_t kernels_52[] = KERNELS_52;
static const uint32_t kernels_53[] = KERNELS_53;
static const uint32_t kernels_54[] = KERNELS_54;
static const uint32_t kernels_55[] = KERNELS_55;
static const uint32_t kernels_56[] = KERNELS_56;
static const uint32_t kernels_57[] = KERNELS_57;
static const uint32_t kernels_58[] = KERNELS_58;
static const uint32_t kernels_59[] = KERNELS_59;
static const uint32_t kernels_60[] = KERNELS_60;
static const uint32_t kernels_61[] = KERNELS_61;
static const uint32_t kernels_62[] = KERNELS_62;
static const uint32_t kernels_63[] = KERNELS_63;

int cnn_load_weights(void)
{
  *((volatile uint8_t *) 0x50180001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50180000, kernels_0, 1728);
  *((volatile uint8_t *) 0x50184001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50184000, kernels_1, 1728);
  *((volatile uint8_t *) 0x50188001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50188000, kernels_2, 1728);
  *((volatile uint8_t *) 0x5018c001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x5018c000, kernels_3, 1728);
  *((volatile uint8_t *) 0x50190001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50190000, kernels_4, 1728);
  *((volatile uint8_t *) 0x50194001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50194000, kernels_5, 1728);
  *((volatile uint8_t *) 0x50198001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50198000, kernels_6, 1728);
  *((volatile uint8_t *) 0x5019c001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x5019c000, kernels_7, 1728);
  *((volatile uint8_t *) 0x501a0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501a0000, kernels_8, 1728);
  *((volatile uint8_t *) 0x501a4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501a4000, kernels_9, 1728);
  *((volatile uint8_t *) 0x501a8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501a8000, kernels_10, 1728);
  *((volatile uint8_t *) 0x501ac001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501ac000, kernels_11, 1728);
  *((volatile uint8_t *) 0x501b0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501b0000, kernels_12, 1728);
  *((volatile uint8_t *) 0x501b4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501b4000, kernels_13, 1728);
  *((volatile uint8_t *) 0x501b8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501b8000, kernels_14, 1728);
  *((volatile uint8_t *) 0x501bc001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501bc000, kernels_15, 1728);
  *((volatile uint8_t *) 0x50580001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50580000, kernels_16, 1728);
  *((volatile uint8_t *) 0x50584001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50584000, kernels_17, 1728);
  *((volatile uint8_t *) 0x50588001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50588000, kernels_18, 1728);
  *((volatile uint8_t *) 0x5058c001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x5058c000, kernels_19, 1728);
  *((volatile uint8_t *) 0x50590001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50590000, kernels_20, 1728);
  *((volatile uint8_t *) 0x50594001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50594000, kernels_21, 1728);
  *((volatile uint8_t *) 0x50598001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50598000, kernels_22, 1728);
  *((volatile uint8_t *) 0x5059c001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x5059c000, kernels_23, 1728);
  *((volatile uint8_t *) 0x505a0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505a0000, kernels_24, 1728);
  *((volatile uint8_t *) 0x505a4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505a4000, kernels_25, 1728);
  *((volatile uint8_t *) 0x505a8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505a8000, kernels_26, 1728);
  *((volatile uint8_t *) 0x505ac001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505ac000, kernels_27, 1728);
  *((volatile uint8_t *) 0x505b0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505b0000, kernels_28, 1728);
  *((volatile uint8_t *) 0x505b4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505b4000, kernels_29, 1728);
  *((volatile uint8_t *) 0x505b8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505b8000, kernels_30, 1728);
  *((volatile uint8_t *) 0x505bc001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505bc000, kernels_31, 1728);
  *((volatile uint8_t *) 0x50980001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50980000, kernels_32, 1728);
  *((volatile uint8_t *) 0x50984001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50984000, kernels_33, 1728);
  *((volatile uint8_t *) 0x50988001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50988000, kernels_34, 1728);
  *((volatile uint8_t *) 0x5098c001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x5098c000, kernels_35, 1728);
  *((volatile uint8_t *) 0x50990001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50990000, kernels_36, 1728);
  *((volatile uint8_t *) 0x50994001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50994000, kernels_37, 1728);
  *((volatile uint8_t *) 0x50998001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50998000, kernels_38, 1728);
  *((volatile uint8_t *) 0x5099c001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x5099c000, kernels_39, 1728);
  *((volatile uint8_t *) 0x509a0101) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509a0000, kernels_40, 1584);
  *((volatile uint8_t *) 0x509a4101) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509a4000, kernels_41, 1584);
  *((volatile uint8_t *) 0x509a8101) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509a8000, kernels_42, 1584);
  *((volatile uint8_t *) 0x509ac101) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509ac000, kernels_43, 1584);
  *((volatile uint8_t *) 0x509b0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509b0000, kernels_44, 1728);
  *((volatile uint8_t *) 0x509b4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509b4000, kernels_45, 1728);
  *((volatile uint8_t *) 0x509b8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509b8000, kernels_46, 1728);
  *((volatile uint8_t *) 0x509bc001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509bc000, kernels_47, 1728);
  *((volatile uint8_t *) 0x50d80001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d80000, kernels_48, 1728);
  *((volatile uint8_t *) 0x50d84001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d84000, kernels_49, 1728);
  *((volatile uint8_t *) 0x50d88001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d88000, kernels_50, 1728);
  *((volatile uint8_t *) 0x50d8c001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d8c000, kernels_51, 1728);
  *((volatile uint8_t *) 0x50d90001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d90000, kernels_52, 1728);
  *((volatile uint8_t *) 0x50d94001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d94000, kernels_53, 1728);
  *((volatile uint8_t *) 0x50d98001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d98000, kernels_54, 1728);
  *((volatile uint8_t *) 0x50d9c001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d9c000, kernels_55, 1728);
  *((volatile uint8_t *) 0x50da0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50da0000, kernels_56, 1728);
  *((volatile uint8_t *) 0x50da4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50da4000, kernels_57, 1728);
  *((volatile uint8_t *) 0x50da8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50da8000, kernels_58, 1728);
  *((volatile uint8_t *) 0x50dac001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50dac000, kernels_59, 1728);
  *((volatile uint8_t *) 0x50db0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50db0000, kernels_60, 1728);
  *((volatile uint8_t *) 0x50db4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50db4000, kernels_61, 1728);
  *((volatile uint8_t *) 0x50db8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50db8000, kernels_62, 1728);
  *((volatile uint8_t *) 0x50dbc051) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50dbc000, kernels_63, 1683);

  return CNN_OK;
}

int cnn_load_bias(void)
{
  // Not used in this network
  return CNN_OK;
}

int cnn_init(void)
{
  *((volatile uint32_t *) 0x50001000) = 0x00000000; // AON control
  *((volatile uint32_t *) 0x50100000) = 0x00100008; // Stop SM
  *((volatile uint32_t *) 0x50100004) = 0x0000040e; // SRAM control
  *((volatile uint32_t *) 0x50100008) = 0x00000010; // Layer count
  *((volatile uint32_t *) 0x50500000) = 0x00100008; // Stop SM
  *((volatile uint32_t *) 0x50500004) = 0x0000040e; // SRAM control
  *((volatile uint32_t *) 0x50500008) = 0x00000010; // Layer count
  *((volatile uint32_t *) 0x50900000) = 0x00100008; // Stop SM
  *((volatile uint32_t *) 0x50900004) = 0x0000040e; // SRAM control
  *((volatile uint32_t *) 0x50900008) = 0x00000010; // Layer count
  *((volatile uint32_t *) 0x50d00000) = 0x00100008; // Stop SM
  *((volatile uint32_t *) 0x50d00004) = 0x0000040e; // SRAM control
  *((volatile uint32_t *) 0x50d00008) = 0x00000010; // Layer count

  return CNN_OK;
}

int cnn_configure(void)
{
  // Layer 0 group 0
  *((volatile uint32_t *) 0x50100010) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x50100090) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50100310) = 0x00016800; // SRAM write ptr
  *((volatile uint32_t *) 0x50100410) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50100590) = 0x00008b20; // Layer control
  *((volatile uint32_t *) 0x50100a10) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50100610) = 0x00000078; // Mask offset and count
  *((volatile uint32_t *) 0x50100690) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x50100790) = 0x00024000; // Post processing register

  // Layer 0 group 1
  *((volatile uint32_t *) 0x50500010) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x50500090) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50500310) = 0x00016800; // SRAM write ptr
  *((volatile uint32_t *) 0x50500410) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50500590) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a10) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50500610) = 0x00000078; // Mask offset and count
  *((volatile uint32_t *) 0x50500690) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x50500790) = 0x00024000; // Post processing register

  // Layer 0 group 2
  *((volatile uint32_t *) 0x50900010) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x50900090) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50900310) = 0x00016800; // SRAM write ptr
  *((volatile uint32_t *) 0x50900410) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50900590) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a10) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50900610) = 0x00000078; // Mask offset and count
  *((volatile uint32_t *) 0x50900690) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x50900790) = 0x00024000; // Post processing register

  // Layer 0 group 3
  *((volatile uint32_t *) 0x50d00010) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x50d00090) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50d00310) = 0x00016800; // SRAM write ptr
  *((volatile uint32_t *) 0x50d00410) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d00590) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a10) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50d00610) = 0x00000078; // Mask offset and count
  *((volatile uint32_t *) 0x50d00690) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x50d00790) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50d00710) = 0x70007000; // Mask and processor enables

  // Layer 1 group 0
  *((volatile uint32_t *) 0x50100014) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x50100094) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50100414) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50100514) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50100594) = 0x0000cb20; // Layer control
  *((volatile uint32_t *) 0x50100a14) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50100614) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x50100694) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x50100794) = 0x00024000; // Post processing register

  // Layer 1 group 1
  *((volatile uint32_t *) 0x50500014) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x50500094) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50500414) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50500514) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50500594) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a14) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50500614) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x50500694) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x50500794) = 0x00024000; // Post processing register

  // Layer 1 group 2
  *((volatile uint32_t *) 0x50900014) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x50900094) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50900414) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50900514) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50900594) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a14) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50900614) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x50900694) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x50900794) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50900714) = 0xf000f000; // Mask and processor enables

  // Layer 1 group 3
  *((volatile uint32_t *) 0x50d00014) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x50d00094) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50d00414) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d00514) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50d00594) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a14) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50d00614) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x50d00694) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x50d00794) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50d00714) = 0x0fff0fff; // Mask and processor enables

  // Layer 2 group 0
  *((volatile uint32_t *) 0x50100018) = 0x0000001f; // Rows
  *((volatile uint32_t *) 0x50100098) = 0x0000001f; // Columns
  *((volatile uint32_t *) 0x50100318) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50100398) = 0x00002000; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50100598) = 0x00000920; // Layer control
  *((volatile uint32_t *) 0x50100a18) = 0x00009810; // Layer control 2
  *((volatile uint32_t *) 0x50100118) = 0x00000103; // 1D
  *((volatile uint32_t *) 0x50100798) = 0x03000000; // Post processing register
  *((volatile uint32_t *) 0x50100718) = 0x0000ffff; // Mask and processor enables

  // Layer 2 group 1
  *((volatile uint32_t *) 0x50500018) = 0x0000001f; // Rows
  *((volatile uint32_t *) 0x50500098) = 0x0000001f; // Columns
  *((volatile uint32_t *) 0x50500318) = 0x00008800; // SRAM write ptr
  *((volatile uint32_t *) 0x50500398) = 0x00002000; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50500598) = 0x00000920; // Layer control
  *((volatile uint32_t *) 0x50500a18) = 0x00009810; // Layer control 2
  *((volatile uint32_t *) 0x50500118) = 0x00000103; // 1D
  *((volatile uint32_t *) 0x50500798) = 0x03000000; // Post processing register
  *((volatile uint32_t *) 0x50500718) = 0x0000000f; // Mask and processor enables

  // Layer 2 group 2
  *((volatile uint32_t *) 0x50900018) = 0x0000001f; // Rows
  *((volatile uint32_t *) 0x50900098) = 0x0000001f; // Columns
  *((volatile uint32_t *) 0x50900318) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50900398) = 0x00002000; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50900598) = 0x00000920; // Layer control
  *((volatile uint32_t *) 0x50900a18) = 0x00009810; // Layer control 2
  *((volatile uint32_t *) 0x50900118) = 0x00000103; // 1D
  *((volatile uint32_t *) 0x50900798) = 0x03000000; // Post processing register

  // Layer 2 group 3
  *((volatile uint32_t *) 0x50d00018) = 0x0000001f; // Rows
  *((volatile uint32_t *) 0x50d00098) = 0x0000001f; // Columns
  *((volatile uint32_t *) 0x50d00318) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50d00398) = 0x00002000; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d00598) = 0x00000920; // Layer control
  *((volatile uint32_t *) 0x50d00a18) = 0x00009810; // Layer control 2
  *((volatile uint32_t *) 0x50d00118) = 0x00000103; // 1D
  *((volatile uint32_t *) 0x50d00798) = 0x03000000; // Post processing register

  // Layer 3 group 0
  *((volatile uint32_t *) 0x5010001c) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x5010009c) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x5010031c) = 0x00000801; // SRAM write ptr
  *((volatile uint32_t *) 0x5010041c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5010059c) = 0x00002b20; // Layer control
  *((volatile uint32_t *) 0x50100a1c) = 0x00009810; // Layer control 2
  *((volatile uint32_t *) 0x5010061c) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x5010069c) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x5010079c) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x5010071c) = 0xffffffff; // Mask and processor enables

  // Layer 3 group 1
  *((volatile uint32_t *) 0x5050001c) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x5050009c) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x5050031c) = 0x00000801; // SRAM write ptr
  *((volatile uint32_t *) 0x5050041c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5050059c) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a1c) = 0x00009810; // Layer control 2
  *((volatile uint32_t *) 0x5050061c) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x5050069c) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x5050079c) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x5050071c) = 0x000f000f; // Mask and processor enables

  // Layer 3 group 2
  *((volatile uint32_t *) 0x5090001c) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x5090009c) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x5090031c) = 0x00000801; // SRAM write ptr
  *((volatile uint32_t *) 0x5090041c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5090059c) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a1c) = 0x00009810; // Layer control 2
  *((volatile uint32_t *) 0x5090061c) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x5090069c) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x5090079c) = 0x00024000; // Post processing register

  // Layer 3 group 3
  *((volatile uint32_t *) 0x50d0001c) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x50d0009c) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50d0031c) = 0x00000801; // SRAM write ptr
  *((volatile uint32_t *) 0x50d0041c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d0059c) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a1c) = 0x00009810; // Layer control 2
  *((volatile uint32_t *) 0x50d0061c) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x50d0069c) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x50d0079c) = 0x00024000; // Post processing register

  // Layer 4 group 0
  *((volatile uint32_t *) 0x50100020) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x501000a0) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50100320) = 0x00016000; // SRAM write ptr
  *((volatile uint32_t *) 0x50100420) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50100520) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x501005a0) = 0x00002b20; // Layer control
  *((volatile uint32_t *) 0x50100a20) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50100620) = 0x00a00138; // Mask offset and count
  *((volatile uint32_t *) 0x50100120) = 0x00066000; // 1D
  *((volatile uint32_t *) 0x501006a0) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x501007a0) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50100720) = 0xffffffff; // Mask and processor enables

  // Layer 4 group 1
  *((volatile uint32_t *) 0x50500020) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x505000a0) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50500320) = 0x00016000; // SRAM write ptr
  *((volatile uint32_t *) 0x50500420) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50500520) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x505005a0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a20) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50500620) = 0x00a00138; // Mask offset and count
  *((volatile uint32_t *) 0x50500120) = 0x00066000; // 1D
  *((volatile uint32_t *) 0x505006a0) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x505007a0) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50500720) = 0x000f000f; // Mask and processor enables

  // Layer 4 group 2
  *((volatile uint32_t *) 0x50900020) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x509000a0) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50900320) = 0x00016000; // SRAM write ptr
  *((volatile uint32_t *) 0x50900420) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50900520) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x509005a0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a20) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50900620) = 0x00a00138; // Mask offset and count
  *((volatile uint32_t *) 0x50900120) = 0x00066000; // 1D
  *((volatile uint32_t *) 0x509006a0) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x509007a0) = 0x00024000; // Post processing register

  // Layer 4 group 3
  *((volatile uint32_t *) 0x50d00020) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x50d000a0) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50d00320) = 0x00016000; // SRAM write ptr
  *((volatile uint32_t *) 0x50d00420) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d00520) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50d005a0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a20) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50d00620) = 0x00a00138; // Mask offset and count
  *((volatile uint32_t *) 0x50d00120) = 0x00066000; // 1D
  *((volatile uint32_t *) 0x50d006a0) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007a0) = 0x00024000; // Post processing register

  // Layer 5 group 0
  *((volatile uint32_t *) 0x50100024) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x501000a4) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x501001a4) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50100224) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x501002a4) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50100324) = 0x0000a800; // SRAM write ptr
  *((volatile uint32_t *) 0x50100424) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501005a4) = 0x0000cba0; // Layer control
  *((volatile uint32_t *) 0x50100a24) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50100624) = 0x00a00138; // Mask offset and count
  *((volatile uint32_t *) 0x501006a4) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x501007a4) = 0x00022000; // Post processing register

  // Layer 5 group 1
  *((volatile uint32_t *) 0x50500024) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x505000a4) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x505001a4) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50500224) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x505002a4) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50500324) = 0x0000a800; // SRAM write ptr
  *((volatile uint32_t *) 0x50500424) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505005a4) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50500a24) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50500624) = 0x00a00138; // Mask offset and count
  *((volatile uint32_t *) 0x505006a4) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x505007a4) = 0x00022000; // Post processing register

  // Layer 5 group 2
  *((volatile uint32_t *) 0x50900024) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x509000a4) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x509001a4) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50900224) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x509002a4) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50900324) = 0x0000a800; // SRAM write ptr
  *((volatile uint32_t *) 0x50900424) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509005a4) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50900a24) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50900624) = 0x00a00138; // Mask offset and count
  *((volatile uint32_t *) 0x509006a4) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x509007a4) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50900724) = 0xf000f000; // Mask and processor enables

  // Layer 5 group 3
  *((volatile uint32_t *) 0x50d00024) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x50d000a4) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50d001a4) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50d00224) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50d002a4) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50d00324) = 0x0000a800; // SRAM write ptr
  *((volatile uint32_t *) 0x50d00424) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d005a4) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50d00a24) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50d00624) = 0x00a00138; // Mask offset and count
  *((volatile uint32_t *) 0x50d006a4) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007a4) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50d00724) = 0xffffffff; // Mask and processor enables

  // Layer 6 group 0
  *((volatile uint32_t *) 0x50100028) = 0x0000000f; // Rows
  *((volatile uint32_t *) 0x501000a8) = 0x0000000f; // Columns
  *((volatile uint32_t *) 0x501003a8) = 0x00002000; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50100528) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x501005a8) = 0x00000920; // Layer control
  *((volatile uint32_t *) 0x50100a28) = 0x00009810; // Layer control 2
  *((volatile uint32_t *) 0x50100128) = 0x00000102; // 1D
  *((volatile uint32_t *) 0x501007a8) = 0x03000000; // Post processing register

  // Layer 6 group 1
  *((volatile uint32_t *) 0x50500028) = 0x0000000f; // Rows
  *((volatile uint32_t *) 0x505000a8) = 0x0000000f; // Columns
  *((volatile uint32_t *) 0x50500328) = 0x0000a000; // SRAM write ptr
  *((volatile uint32_t *) 0x505003a8) = 0x00002000; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50500528) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x505005a8) = 0x00000920; // Layer control
  *((volatile uint32_t *) 0x50500a28) = 0x00009810; // Layer control 2
  *((volatile uint32_t *) 0x50500128) = 0x00000102; // 1D
  *((volatile uint32_t *) 0x505007a8) = 0x03000000; // Post processing register
  *((volatile uint32_t *) 0x50500728) = 0x0000fff0; // Mask and processor enables

  // Layer 6 group 2
  *((volatile uint32_t *) 0x50900028) = 0x0000000f; // Rows
  *((volatile uint32_t *) 0x509000a8) = 0x0000000f; // Columns
  *((volatile uint32_t *) 0x50900328) = 0x00010000; // SRAM write ptr
  *((volatile uint32_t *) 0x509003a8) = 0x00002000; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50900528) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x509005a8) = 0x00000920; // Layer control
  *((volatile uint32_t *) 0x50900a28) = 0x00009810; // Layer control 2
  *((volatile uint32_t *) 0x50900128) = 0x00000102; // 1D
  *((volatile uint32_t *) 0x509007a8) = 0x03000000; // Post processing register
  *((volatile uint32_t *) 0x50900728) = 0x000000ff; // Mask and processor enables

  // Layer 6 group 3
  *((volatile uint32_t *) 0x50d00028) = 0x0000000f; // Rows
  *((volatile uint32_t *) 0x50d000a8) = 0x0000000f; // Columns
  *((volatile uint32_t *) 0x50d003a8) = 0x00002000; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d00528) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50d005a8) = 0x00000920; // Layer control
  *((volatile uint32_t *) 0x50d00a28) = 0x00009810; // Layer control 2
  *((volatile uint32_t *) 0x50d00128) = 0x00000102; // 1D
  *((volatile uint32_t *) 0x50d007a8) = 0x03000000; // Post processing register

  // Layer 7 group 0
  *((volatile uint32_t *) 0x5010002c) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x501000ac) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x5010032c) = 0x0000a001; // SRAM write ptr
  *((volatile uint32_t *) 0x5010042c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5010052c) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x501005ac) = 0x00006b20; // Layer control
  *((volatile uint32_t *) 0x50100a2c) = 0x00009810; // Layer control 2
  *((volatile uint32_t *) 0x5010062c) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x501006ac) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x501007ac) = 0x00024000; // Post processing register

  // Layer 7 group 1
  *((volatile uint32_t *) 0x5050002c) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x505000ac) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x5050032c) = 0x0000a001; // SRAM write ptr
  *((volatile uint32_t *) 0x5050042c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5050052c) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x505005ac) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a2c) = 0x00009810; // Layer control 2
  *((volatile uint32_t *) 0x5050062c) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x505006ac) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x505007ac) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x5050072c) = 0xfff0fff0; // Mask and processor enables

  // Layer 7 group 2
  *((volatile uint32_t *) 0x5090002c) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x509000ac) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x5090032c) = 0x0000a001; // SRAM write ptr
  *((volatile uint32_t *) 0x5090042c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5090052c) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x509005ac) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a2c) = 0x00009810; // Layer control 2
  *((volatile uint32_t *) 0x5090062c) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x509006ac) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x509007ac) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x5090072c) = 0x00ff00ff; // Mask and processor enables

  // Layer 7 group 3
  *((volatile uint32_t *) 0x50d0002c) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x50d000ac) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x50d0032c) = 0x0000a001; // SRAM write ptr
  *((volatile uint32_t *) 0x50d0042c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d0052c) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50d005ac) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a2c) = 0x00009810; // Layer control 2
  *((volatile uint32_t *) 0x50d0062c) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x50d006ac) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007ac) = 0x00024000; // Post processing register

  // Layer 8 group 0
  *((volatile uint32_t *) 0x50100030) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x501000b0) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x50100330) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50100430) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501005b0) = 0x00006b20; // Layer control
  *((volatile uint32_t *) 0x50100a30) = 0x00015800; // Layer control 2
  *((volatile uint32_t *) 0x50100630) = 0x00a001f8; // Mask offset and count
  *((volatile uint32_t *) 0x50100130) = 0x00066000; // 1D
  *((volatile uint32_t *) 0x501006b0) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x501007b0) = 0x00024000; // Post processing register

  // Layer 8 group 1
  *((volatile uint32_t *) 0x50500030) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x505000b0) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x50500330) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50500430) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505005b0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a30) = 0x00015800; // Layer control 2
  *((volatile uint32_t *) 0x50500630) = 0x00a001f8; // Mask offset and count
  *((volatile uint32_t *) 0x50500130) = 0x00066000; // 1D
  *((volatile uint32_t *) 0x505006b0) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x505007b0) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50500730) = 0xfff0fff0; // Mask and processor enables

  // Layer 8 group 2
  *((volatile uint32_t *) 0x50900030) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x509000b0) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x50900330) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50900430) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509005b0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a30) = 0x00015800; // Layer control 2
  *((volatile uint32_t *) 0x50900630) = 0x00a001f8; // Mask offset and count
  *((volatile uint32_t *) 0x50900130) = 0x00066000; // 1D
  *((volatile uint32_t *) 0x509006b0) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x509007b0) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50900730) = 0x00ff00ff; // Mask and processor enables

  // Layer 8 group 3
  *((volatile uint32_t *) 0x50d00030) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x50d000b0) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x50d00330) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50d00430) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d005b0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a30) = 0x00015800; // Layer control 2
  *((volatile uint32_t *) 0x50d00630) = 0x00a001f8; // Mask offset and count
  *((volatile uint32_t *) 0x50d00130) = 0x00066000; // 1D
  *((volatile uint32_t *) 0x50d006b0) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007b0) = 0x00024000; // Post processing register

  // Layer 9 group 0
  *((volatile uint32_t *) 0x50100034) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x501000b4) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x501001b4) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50100234) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x501002b4) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50100434) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50100534) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x501005b4) = 0x00006ba0; // Layer control
  *((volatile uint32_t *) 0x50100a34) = 0x00017800; // Layer control 2
  *((volatile uint32_t *) 0x50100634) = 0x02000378; // Mask offset and count
  *((volatile uint32_t *) 0x501006b4) = 0x00000007; // TRAM ptr max
  *((volatile uint32_t *) 0x501007b4) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50100734) = 0xffffffff; // Mask and processor enables

  // Layer 9 group 1
  *((volatile uint32_t *) 0x50500034) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x505000b4) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x505001b4) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50500234) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x505002b4) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50500434) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50500534) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x505005b4) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50500a34) = 0x00017800; // Layer control 2
  *((volatile uint32_t *) 0x50500634) = 0x02000378; // Mask offset and count
  *((volatile uint32_t *) 0x505006b4) = 0x00000007; // TRAM ptr max
  *((volatile uint32_t *) 0x505007b4) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50500734) = 0xffffffff; // Mask and processor enables

  // Layer 9 group 2
  *((volatile uint32_t *) 0x50900034) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x509000b4) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x509001b4) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50900234) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x509002b4) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50900434) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50900534) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x509005b4) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50900a34) = 0x00017800; // Layer control 2
  *((volatile uint32_t *) 0x50900634) = 0x02000378; // Mask offset and count
  *((volatile uint32_t *) 0x509006b4) = 0x00000007; // TRAM ptr max
  *((volatile uint32_t *) 0x509007b4) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50900734) = 0x0fff0fff; // Mask and processor enables

  // Layer 9 group 3
  *((volatile uint32_t *) 0x50d00034) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x50d000b4) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x50d001b4) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50d00234) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50d002b4) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50d00434) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d00534) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50d005b4) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50d00a34) = 0x00017800; // Layer control 2
  *((volatile uint32_t *) 0x50d00634) = 0x02000378; // Mask offset and count
  *((volatile uint32_t *) 0x50d006b4) = 0x00000007; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007b4) = 0x00024000; // Post processing register

  // Layer 10 group 0
  *((volatile uint32_t *) 0x50100038) = 0x00000007; // Rows
  *((volatile uint32_t *) 0x501000b8) = 0x00000007; // Columns
  *((volatile uint32_t *) 0x50100338) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x501003b8) = 0x00002000; // Write ptr time slot offs
  *((volatile uint32_t *) 0x501005b8) = 0x00000920; // Layer control
  *((volatile uint32_t *) 0x50100a38) = 0x00017810; // Layer control 2
  *((volatile uint32_t *) 0x50100138) = 0x00000103; // 1D
  *((volatile uint32_t *) 0x501007b8) = 0x03000000; // Post processing register
  *((volatile uint32_t *) 0x50100738) = 0x0000ffff; // Mask and processor enables

  // Layer 10 group 1
  *((volatile uint32_t *) 0x50500038) = 0x00000007; // Rows
  *((volatile uint32_t *) 0x505000b8) = 0x00000007; // Columns
  *((volatile uint32_t *) 0x50500338) = 0x00008800; // SRAM write ptr
  *((volatile uint32_t *) 0x505003b8) = 0x00002000; // Write ptr time slot offs
  *((volatile uint32_t *) 0x505005b8) = 0x00000920; // Layer control
  *((volatile uint32_t *) 0x50500a38) = 0x00017810; // Layer control 2
  *((volatile uint32_t *) 0x50500138) = 0x00000103; // 1D
  *((volatile uint32_t *) 0x505007b8) = 0x03000000; // Post processing register
  *((volatile uint32_t *) 0x50500738) = 0x0000ffff; // Mask and processor enables

  // Layer 10 group 2
  *((volatile uint32_t *) 0x50900038) = 0x00000007; // Rows
  *((volatile uint32_t *) 0x509000b8) = 0x00000007; // Columns
  *((volatile uint32_t *) 0x50900338) = 0x00010800; // SRAM write ptr
  *((volatile uint32_t *) 0x509003b8) = 0x00002000; // Write ptr time slot offs
  *((volatile uint32_t *) 0x509005b8) = 0x00000920; // Layer control
  *((volatile uint32_t *) 0x50900a38) = 0x00017810; // Layer control 2
  *((volatile uint32_t *) 0x50900138) = 0x00000103; // 1D
  *((volatile uint32_t *) 0x509007b8) = 0x03000000; // Post processing register
  *((volatile uint32_t *) 0x50900738) = 0x0000ffff; // Mask and processor enables

  // Layer 10 group 3
  *((volatile uint32_t *) 0x50d00038) = 0x00000007; // Rows
  *((volatile uint32_t *) 0x50d000b8) = 0x00000007; // Columns
  *((volatile uint32_t *) 0x50d00338) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50d003b8) = 0x00002000; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d005b8) = 0x00000920; // Layer control
  *((volatile uint32_t *) 0x50d00a38) = 0x00017810; // Layer control 2
  *((volatile uint32_t *) 0x50d00138) = 0x00000103; // 1D
  *((volatile uint32_t *) 0x50d007b8) = 0x03000000; // Post processing register

  // Layer 11 group 0
  *((volatile uint32_t *) 0x5010003c) = 0x00010009; // Rows
  *((volatile uint32_t *) 0x501000bc) = 0x00010009; // Columns
  *((volatile uint32_t *) 0x5010033c) = 0x00000801; // SRAM write ptr
  *((volatile uint32_t *) 0x5010043c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501005bc) = 0x00006b20; // Layer control
  *((volatile uint32_t *) 0x50100a3c) = 0x00017810; // Layer control 2
  *((volatile uint32_t *) 0x5010063c) = 0x038004f8; // Mask offset and count
  *((volatile uint32_t *) 0x501006bc) = 0x00000007; // TRAM ptr max
  *((volatile uint32_t *) 0x501007bc) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x5010073c) = 0xffffffff; // Mask and processor enables

  // Layer 11 group 1
  *((volatile uint32_t *) 0x5050003c) = 0x00010009; // Rows
  *((volatile uint32_t *) 0x505000bc) = 0x00010009; // Columns
  *((volatile uint32_t *) 0x5050033c) = 0x00000801; // SRAM write ptr
  *((volatile uint32_t *) 0x5050043c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505005bc) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a3c) = 0x00017810; // Layer control 2
  *((volatile uint32_t *) 0x5050063c) = 0x038004f8; // Mask offset and count
  *((volatile uint32_t *) 0x505006bc) = 0x00000007; // TRAM ptr max
  *((volatile uint32_t *) 0x505007bc) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x5050073c) = 0xffffffff; // Mask and processor enables

  // Layer 11 group 2
  *((volatile uint32_t *) 0x5090003c) = 0x00010009; // Rows
  *((volatile uint32_t *) 0x509000bc) = 0x00010009; // Columns
  *((volatile uint32_t *) 0x5090033c) = 0x00000801; // SRAM write ptr
  *((volatile uint32_t *) 0x5090043c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509005bc) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a3c) = 0x00017810; // Layer control 2
  *((volatile uint32_t *) 0x5090063c) = 0x038004f8; // Mask offset and count
  *((volatile uint32_t *) 0x509006bc) = 0x00000007; // TRAM ptr max
  *((volatile uint32_t *) 0x509007bc) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x5090073c) = 0xffffffff; // Mask and processor enables

  // Layer 11 group 3
  *((volatile uint32_t *) 0x50d0003c) = 0x00010009; // Rows
  *((volatile uint32_t *) 0x50d000bc) = 0x00010009; // Columns
  *((volatile uint32_t *) 0x50d0033c) = 0x00000801; // SRAM write ptr
  *((volatile uint32_t *) 0x50d0043c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d005bc) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a3c) = 0x00017810; // Layer control 2
  *((volatile uint32_t *) 0x50d0063c) = 0x038004f8; // Mask offset and count
  *((volatile uint32_t *) 0x50d006bc) = 0x00000007; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007bc) = 0x00024000; // Post processing register

  // Layer 12 group 0
  *((volatile uint32_t *) 0x50100040) = 0x00010009; // Rows
  *((volatile uint32_t *) 0x501000c0) = 0x00010009; // Columns
  *((volatile uint32_t *) 0x501001c0) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50100240) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x501002c0) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50100440) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501004c0) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x50100540) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x501005c0) = 0x00006ba0; // Layer control
  *((volatile uint32_t *) 0x50100a40) = 0x00017810; // Layer control 2
  *((volatile uint32_t *) 0x50100640) = 0x050007f8; // Mask offset and count
  *((volatile uint32_t *) 0x50100140) = 0x00066000; // 1D
  *((volatile uint32_t *) 0x501006c0) = 0x00000003; // TRAM ptr max
  *((volatile uint32_t *) 0x501007c0) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50100740) = 0xffffffff; // Mask and processor enables

  // Layer 12 group 1
  *((volatile uint32_t *) 0x50500040) = 0x00010009; // Rows
  *((volatile uint32_t *) 0x505000c0) = 0x00010009; // Columns
  *((volatile uint32_t *) 0x505001c0) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50500240) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x505002c0) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50500440) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505004c0) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x50500540) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x505005c0) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50500a40) = 0x00017810; // Layer control 2
  *((volatile uint32_t *) 0x50500640) = 0x050007f8; // Mask offset and count
  *((volatile uint32_t *) 0x50500140) = 0x00066000; // 1D
  *((volatile uint32_t *) 0x505006c0) = 0x00000003; // TRAM ptr max
  *((volatile uint32_t *) 0x505007c0) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50500740) = 0xffffffff; // Mask and processor enables

  // Layer 12 group 2
  *((volatile uint32_t *) 0x50900040) = 0x00010009; // Rows
  *((volatile uint32_t *) 0x509000c0) = 0x00010009; // Columns
  *((volatile uint32_t *) 0x509001c0) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50900240) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x509002c0) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50900440) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509004c0) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x50900540) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x509005c0) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50900a40) = 0x00017810; // Layer control 2
  *((volatile uint32_t *) 0x50900640) = 0x050007f8; // Mask offset and count
  *((volatile uint32_t *) 0x50900140) = 0x00066000; // 1D
  *((volatile uint32_t *) 0x509006c0) = 0x00000003; // TRAM ptr max
  *((volatile uint32_t *) 0x509007c0) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50900740) = 0xffffffff; // Mask and processor enables

  // Layer 12 group 3
  *((volatile uint32_t *) 0x50d00040) = 0x00010009; // Rows
  *((volatile uint32_t *) 0x50d000c0) = 0x00010009; // Columns
  *((volatile uint32_t *) 0x50d001c0) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50d00240) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50d002c0) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50d00440) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d004c0) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x50d00540) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50d005c0) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50d00a40) = 0x00017810; // Layer control 2
  *((volatile uint32_t *) 0x50d00640) = 0x050007f8; // Mask offset and count
  *((volatile uint32_t *) 0x50d00140) = 0x00066000; // 1D
  *((volatile uint32_t *) 0x50d006c0) = 0x00000003; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007c0) = 0x00022000; // Post processing register

  // Layer 13 group 0
  *((volatile uint32_t *) 0x50100044) = 0x00000003; // Rows
  *((volatile uint32_t *) 0x501000c4) = 0x00000003; // Columns
  *((volatile uint32_t *) 0x501001c4) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50100244) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x501002c4) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50100344) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x501003c4) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50100444) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501004c4) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x501005c4) = 0x00006ba0; // Layer control
  *((volatile uint32_t *) 0x50100a44) = 0x0001f871; // Layer control 2
  *((volatile uint32_t *) 0x50100644) = 0x480067f8; // Mask offset and count
  *((volatile uint32_t *) 0x50100144) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x501007c4) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50100744) = 0xffffffff; // Mask and processor enables

  // Layer 13 group 1
  *((volatile uint32_t *) 0x50500044) = 0x00000003; // Rows
  *((volatile uint32_t *) 0x505000c4) = 0x00000003; // Columns
  *((volatile uint32_t *) 0x505001c4) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50500244) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x505002c4) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50500344) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x505003c4) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50500444) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505004c4) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x505005c4) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50500a44) = 0x0001f871; // Layer control 2
  *((volatile uint32_t *) 0x50500644) = 0x480067f8; // Mask offset and count
  *((volatile uint32_t *) 0x50500144) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x505007c4) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50500744) = 0xffffffff; // Mask and processor enables

  // Layer 13 group 2
  *((volatile uint32_t *) 0x50900044) = 0x00000003; // Rows
  *((volatile uint32_t *) 0x509000c4) = 0x00000003; // Columns
  *((volatile uint32_t *) 0x509001c4) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50900244) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x509002c4) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50900344) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x509003c4) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50900444) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509004c4) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x509005c4) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50900a44) = 0x0001f871; // Layer control 2
  *((volatile uint32_t *) 0x50900644) = 0x480067f8; // Mask offset and count
  *((volatile uint32_t *) 0x50900144) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x509007c4) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50900744) = 0xffffffff; // Mask and processor enables

  // Layer 13 group 3
  *((volatile uint32_t *) 0x50d00044) = 0x00000003; // Rows
  *((volatile uint32_t *) 0x50d000c4) = 0x00000003; // Columns
  *((volatile uint32_t *) 0x50d001c4) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50d00244) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50d002c4) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50d00344) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50d003c4) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d00444) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d004c4) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x50d005c4) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50d00a44) = 0x0001f871; // Layer control 2
  *((volatile uint32_t *) 0x50d00644) = 0x480067f8; // Mask offset and count
  *((volatile uint32_t *) 0x50d00144) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50d007c4) = 0x00022000; // Post processing register

  // Layer 14 group 0
  *((volatile uint32_t *) 0x50100048) = 0x00000001; // Rows
  *((volatile uint32_t *) 0x501000c8) = 0x00000001; // Columns
  *((volatile uint32_t *) 0x501003c8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50100448) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501004c8) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x50100548) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x501005c8) = 0x0000eb20; // Layer control
  *((volatile uint32_t *) 0x50100a48) = 0x0001f817; // Layer control 2
  *((volatile uint32_t *) 0x50100648) = 0x68a08898; // Mask offset and count
  *((volatile uint32_t *) 0x50100148) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x501007c8) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50100748) = 0xffffffff; // Mask and processor enables

  // Layer 14 group 1
  *((volatile uint32_t *) 0x50500048) = 0x00000001; // Rows
  *((volatile uint32_t *) 0x505000c8) = 0x00000001; // Columns
  *((volatile uint32_t *) 0x505003c8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50500448) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505004c8) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x50500548) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x505005c8) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a48) = 0x0001f817; // Layer control 2
  *((volatile uint32_t *) 0x50500648) = 0x68a08898; // Mask offset and count
  *((volatile uint32_t *) 0x50500148) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x505007c8) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50500748) = 0xffffffff; // Mask and processor enables

  // Layer 14 group 2
  *((volatile uint32_t *) 0x50900048) = 0x00000001; // Rows
  *((volatile uint32_t *) 0x509000c8) = 0x00000001; // Columns
  *((volatile uint32_t *) 0x509003c8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50900448) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509004c8) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x50900548) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x509005c8) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a48) = 0x0001f817; // Layer control 2
  *((volatile uint32_t *) 0x50900648) = 0x68a08898; // Mask offset and count
  *((volatile uint32_t *) 0x50900148) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x509007c8) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50900748) = 0xffffffff; // Mask and processor enables

  // Layer 14 group 3
  *((volatile uint32_t *) 0x50d00048) = 0x00000001; // Rows
  *((volatile uint32_t *) 0x50d000c8) = 0x00000001; // Columns
  *((volatile uint32_t *) 0x50d003c8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d00448) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d004c8) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x50d00548) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50d005c8) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a48) = 0x0001f817; // Layer control 2
  *((volatile uint32_t *) 0x50d00648) = 0x68a08898; // Mask offset and count
  *((volatile uint32_t *) 0x50d00148) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50d007c8) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50d00748) = 0xffffffff; // Mask and processor enables

  // Layer 15 group 0
  *((volatile uint32_t *) 0x5010004c) = 0x00010003; // Rows
  *((volatile uint32_t *) 0x501000cc) = 0x00010003; // Columns
  *((volatile uint32_t *) 0x501001cc) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x5010024c) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x501002cc) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x5010034c) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x5010044c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501004cc) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x501005cc) = 0x0000eba0; // Layer control
  *((volatile uint32_t *) 0x50100a4c) = 0x0001f811; // Layer control 2
  *((volatile uint32_t *) 0x5010064c) = 0x0f401738; // Mask offset and count
  *((volatile uint32_t *) 0x501007cc) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x5010074c) = 0xffffffff; // Mask and processor enables

  // Layer 15 group 1
  *((volatile uint32_t *) 0x5050004c) = 0x00010003; // Rows
  *((volatile uint32_t *) 0x505000cc) = 0x00010003; // Columns
  *((volatile uint32_t *) 0x505001cc) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x5050024c) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x505002cc) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x5050034c) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x5050044c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505004cc) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x505005cc) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50500a4c) = 0x0001f811; // Layer control 2
  *((volatile uint32_t *) 0x5050064c) = 0x0f401738; // Mask offset and count
  *((volatile uint32_t *) 0x505007cc) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x5050074c) = 0xffffffff; // Mask and processor enables

  // Layer 15 group 2
  *((volatile uint32_t *) 0x5090004c) = 0x00010003; // Rows
  *((volatile uint32_t *) 0x509000cc) = 0x00010003; // Columns
  *((volatile uint32_t *) 0x509001cc) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x5090024c) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x509002cc) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x5090034c) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x5090044c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509004cc) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x509005cc) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50900a4c) = 0x0001f811; // Layer control 2
  *((volatile uint32_t *) 0x5090064c) = 0x0f401738; // Mask offset and count
  *((volatile uint32_t *) 0x509007cc) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x5090074c) = 0xffffffff; // Mask and processor enables

  // Layer 15 group 3
  *((volatile uint32_t *) 0x50d0004c) = 0x00010003; // Rows
  *((volatile uint32_t *) 0x50d000cc) = 0x00010003; // Columns
  *((volatile uint32_t *) 0x50d001cc) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50d0024c) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50d002cc) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50d0034c) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50d0044c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d004cc) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x50d005cc) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50d00a4c) = 0x0001f811; // Layer control 2
  *((volatile uint32_t *) 0x50d0064c) = 0x0f401738; // Mask offset and count
  *((volatile uint32_t *) 0x50d007cc) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50d0074c) = 0xffffffff; // Mask and processor enables

  // Layer 16 group 0
  *((volatile uint32_t *) 0x501003d0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50100450) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501004d0) = 0x00000004; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x50100550) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x501005d0) = 0x0001e920; // Layer control
  *((volatile uint32_t *) 0x50100a50) = 0x00019811; // Layer control 2
  *((volatile uint32_t *) 0x50100650) = 0xd140d7b8; // Mask offset and count
  *((volatile uint32_t *) 0x50100150) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50100750) = 0xffffffff; // Mask and processor enables

  // Layer 16 group 1
  *((volatile uint32_t *) 0x505003d0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50500450) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505004d0) = 0x00000004; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x50500550) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x505005d0) = 0x00010920; // Layer control
  *((volatile uint32_t *) 0x50500a50) = 0x00019811; // Layer control 2
  *((volatile uint32_t *) 0x50500650) = 0xd140d7b8; // Mask offset and count
  *((volatile uint32_t *) 0x50500150) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50500750) = 0xffffffff; // Mask and processor enables

  // Layer 16 group 2
  *((volatile uint32_t *) 0x509003d0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50900450) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509004d0) = 0x00000004; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x50900550) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x509005d0) = 0x00010920; // Layer control
  *((volatile uint32_t *) 0x50900a50) = 0x00019811; // Layer control 2
  *((volatile uint32_t *) 0x50900650) = 0xd140d7b8; // Mask offset and count
  *((volatile uint32_t *) 0x50900150) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50900750) = 0xffffffff; // Mask and processor enables

  // Layer 16 group 3
  *((volatile uint32_t *) 0x50d003d0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d00450) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d004d0) = 0x00000004; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x50d00550) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50d005d0) = 0x00010920; // Layer control
  *((volatile uint32_t *) 0x50d00a50) = 0x00019811; // Layer control 2
  *((volatile uint32_t *) 0x50d00650) = 0xd140d7b8; // Mask offset and count
  *((volatile uint32_t *) 0x50d00150) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50d00750) = 0xffffffff; // Mask and processor enables


  return CNN_OK;
}

int cnn_start(void)
{
  cnn_time = 0;

  *((volatile uint32_t *) 0x50100000) = 0x00100808; // Enable group 0
  *((volatile uint32_t *) 0x50500000) = 0x00100809; // Enable group 1
  *((volatile uint32_t *) 0x50900000) = 0x00100809; // Enable group 2
  *((volatile uint32_t *) 0x50d00000) = 0x00100809; // Enable group 3

#ifdef CNN_INFERENCE_TIMER
  MXC_TMR_SW_Start(CNN_INFERENCE_TIMER);
#endif

  CNN_START; // Allow capture of processing time
  *((volatile uint32_t *) 0x50100000) = 0x00100009; // Master enable group 0

  return CNN_OK;
}

// Custom unload for this network: 32-bit data, shape: [100, 1, 1]
int cnn_unload(uint32_t *out_buf)
{
  volatile uint32_t *addr;
  addr = (volatile uint32_t *) 0x50400000;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50408000;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50410000;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50418000;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50800000;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50808000;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50810000;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50818000;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50c00000;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50c08000;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50c10000;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50c18000;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x51000000;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50400010;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50408010;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50410010;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50418010;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50800010;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50808010;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50810010;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50818010;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50c00010;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50c08010;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50c10010;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50c18010;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;

  return CNN_OK;
}

int cnn_enable(uint32_t clock_source, uint32_t clock_divider)
{
  // Reset all domains, restore power to CNN
  MXC_GCFR->reg3 = 0xf; // Reset
  MXC_GCFR->reg1 = 0xf; // Mask memory
  MXC_GCFR->reg0 = 0xf; // Power
  MXC_GCFR->reg2 = 0x0; // Iso
  MXC_GCFR->reg3 = 0x0; // Reset

  MXC_GCR->pclkdiv = (MXC_GCR->pclkdiv & ~(MXC_F_GCR_PCLKDIV_CNNCLKDIV | MXC_F_GCR_PCLKDIV_CNNCLKSEL))
                     | clock_divider | clock_source;
  MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CNN); // Enable CNN clock

  NVIC_SetVector(CNN_IRQn, CNN_ISR); // Set CNN complete vector

  return CNN_OK;
}

int cnn_boost_enable(mxc_gpio_regs_t *port, uint32_t pin)
{
  mxc_gpio_cfg_t gpio_out;
  gpio_out.port = port;
  gpio_out.mask = pin;
  gpio_out.pad = MXC_GPIO_PAD_NONE;
  gpio_out.func = MXC_GPIO_FUNC_OUT;
  MXC_GPIO_Config(&gpio_out);
  MXC_GPIO_OutSet(gpio_out.port, gpio_out.mask);

  return CNN_OK;
}

int cnn_boost_disable(mxc_gpio_regs_t *port, uint32_t pin)
{
  mxc_gpio_cfg_t gpio_out;
  gpio_out.port = port;
  gpio_out.mask = pin;
  gpio_out.pad = MXC_GPIO_PAD_NONE;
  gpio_out.func = MXC_GPIO_FUNC_OUT;
  MXC_GPIO_Config(&gpio_out);
  MXC_GPIO_OutSet(gpio_out.port, gpio_out.mask);

  return CNN_OK;
}

int cnn_disable(void)
{
  // Disable CNN clock
  MXC_SYS_ClockDisable(MXC_SYS_PERIPH_CLOCK_CNN);

  // Disable power to CNN
  MXC_GCFR->reg3 = 0xf; // Reset
  MXC_GCFR->reg1 = 0x0; // Mask memory
  MXC_GCFR->reg0 = 0x0; // Power
  MXC_GCFR->reg2 = 0xf; // Iso
  MXC_GCFR->reg3 = 0x0; // Reset

  return CNN_OK;
}

