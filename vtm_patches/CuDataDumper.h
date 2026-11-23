#pragma once
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include "CommonLib/CommonDef.h"
#include "CommonLib/Buffer.h"
#include "CommonLib/CodingStructure.h"

class CuDataDumper {
private:
  template <int SIZE>
  static void writeBlock(FILE* fp, const CodingStructure& bestCS, const CompArea& lumaArea) {
    uint8_t splitLabel = 0;
    if (bestCS.cus.size() > 1) {
      splitLabel = 1; // 預設 QT
      bool hasFullW = false, hasFullH = false;
      bool hasQuarterW = false, hasQuarterH = false;

      for (const auto& cu : bestCS.cus) {
        const Area c = cu->Y();
        if (c.width == SIZE) hasFullW = true;
        if (c.height == SIZE) hasFullH = true;
        if (c.width == SIZE / 4) hasQuarterW = true;
        if (c.height == SIZE / 4) hasQuarterH = true;
      }

      if (hasFullW && !hasFullH) {
        splitLabel = hasQuarterH ? 4 : 2; // 4: TTH, 2: BTH
      } else if (!hasFullW && hasFullH) {
        splitLabel = hasQuarterW ? 5 : 3; // 5: TTV, 3: BTV
      } else {
        splitLabel = 1; // 1: QT
      }
    }

    const CPelBuf orgY = bestCS.picture->getOrigBuf(lumaArea);
    uint8_t patch[SIZE * SIZE];
    const int bitDepth = bestCS.sps->getBitDepth(static_cast<ChannelType>(0));
    const int shift = (bitDepth > 8) ? (bitDepth - 8) : 0;

    for (int y = 0; y < SIZE; ++y) {
      const Pel* row = orgY.buf + y * orgY.stride;
      for (int x = 0; x < SIZE; ++x) {
        patch[y * SIZE + x] = static_cast<uint8_t>(row[x] >> shift);
      }
    }

    uint8_t qp = static_cast<uint8_t>(bestCS.cus[0]->qp);
    uint16_t meta[3] = {
      static_cast<uint16_t>(bestCS.picture->poc),
      static_cast<uint16_t>(lumaArea.x),
      static_cast<uint16_t>(lumaArea.y)
    };

    std::fwrite(patch, 1, SIZE * SIZE, fp);
    std::fwrite(&qp, 1, 1, fp);
    std::fwrite(&splitLabel, 1, 1, fp);
    std::fwrite(meta, sizeof(uint16_t), 3, fp);
    std::fflush(fp);
  }

public:
  static void dump32x32IfTarget(const CodingStructure& bestCS, const CompArea& lumaArea) {
    if (!lumaArea.valid() || lumaArea.width != lumaArea.height || bestCS.cus.empty()) return;

    static bool init = false;
    static FILE *fp32 = nullptr, *fp16 = nullptr, *fp8 = nullptr;
    if (!init) {
      init = true;
      if (const char* p32 = std::getenv("VVC_DUMP_32X32_PATH"); p32 && p32[0]) fp32 = std::fopen(p32, "ab");
      if (const char* p16 = std::getenv("VVC_DUMP_16X16_PATH"); p16 && p16[0]) fp16 = std::fopen(p16, "ab");
      if (const char* p8  = std::getenv("VVC_DUMP_8X8_PATH");   p8  && p8[0])  fp8  = std::fopen(p8,  "ab");
    }

    if (lumaArea.width == 32 && fp32)      writeBlock<32>(fp32, bestCS, lumaArea);
    else if (lumaArea.width == 16 && fp16) writeBlock<16>(fp16, bestCS, lumaArea);
    else if (lumaArea.width == 8 && fp8)   writeBlock<8>(fp8, bestCS, lumaArea);
  }
};