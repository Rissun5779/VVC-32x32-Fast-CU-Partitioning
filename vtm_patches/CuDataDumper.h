#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "CommonLib/Buffer.h"
#include "CommonLib/CodingStructure.h"

class CuDataDumper {
public:
  static void dump32x32IfTarget(const CodingStructure& bestCS, const CompArea& lumaArea) {
      const char* envVal = std::getenv("VVC_DUMP_32X32_PATH");
      std::fprintf(stderr, "w=%d h=%d valid=%d env=%s\n",
                   lumaArea.width, lumaArea.height, lumaArea.valid(), envVal ? envVal : "NULL");
    }
    if (!lumaArea.valid() || lumaArea.width != 32 || lumaArea.height != 32) return;

    static FILE* fp = nullptr;
    if (!fp) {
      const char* outPath = std::getenv("VVC_DUMP_32X32_PATH");
      if (outPath && outPath[0] != '\0') {
        fp = std::fopen(outPath, "ab");
      }
    }
    if (!fp) return;

    uint8_t splitLabel = 0;
    if (bestCS.cus.size() > 1) {
      const Area c0 = bestCS.cus[0]->Y();
      if (c0.width == 32 && c0.height == 16)      splitLabel = 2;
      else if (c0.width == 16 && c0.height == 32) splitLabel = 3;
      else if (c0.width == 32 && c0.height == 8)  splitLabel = 4;
      else if (c0.width == 8  && c0.height == 32) splitLabel = 5;
      else                                        splitLabel = 1;
    }

    const CPelBuf orgY = bestCS.picture->getOrigBuf(lumaArea);
    uint8_t patch[1024];
    for (int y = 0; y < 32; ++y) {
      const Pel* row = orgY.buf + y * orgY.stride;
      for (int x = 0; x < 32; ++x) {
        int val = row[x];
        patch[y * 32 + x] = static_cast<uint8_t>(val > 255 ? (val >> 2) : val);
      }
    }
    uint8_t qp = static_cast<uint8_t>(bestCS.cus[0]->qp);
    std::fwrite(patch, 1, 1024, fp);
    std::fwrite(&qp, 1, 1, fp);
    std::fwrite(&splitLabel, 1, 1, fp);
    std::fflush(fp);
  }
};