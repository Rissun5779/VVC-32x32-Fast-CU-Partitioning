import pathlib, shutil

root = pathlib.Path(__file__).resolve().parent.parent
enc_dir = root / "external/VTM24.0/source/Lib/EncoderLib"
shutil.copy2(root / "vtm_patches/CuDataDumper.h", enc_dir / "CuDataDumper.h")

enc_cu = enc_dir / "EncCu.cpp"
# 清除所有先前注入的痕跡
lines = [l for l in enc_cu.read_text(encoding="utf-8").splitlines() if "dump32x32" not in l]
code = "\n".join(lines) + "\n"

# 確保 include 存在
if '#include "CuDataDumper.h"' not in code:
    code = code.replace('#include "EncCu.h"', '#include "EncCu.h"\n#include "CuDataDumper.h"', 1)

fn_pos = code.find("void EncCu::xCompressCU(")
assert fn_pos != -1, "xCompressCU not found"

anchor = 'CHECK( bestCS->cost             == MAX_DOUBLE                , "No possible encoding found" );'
target_pos = code.find(anchor, fn_pos)
assert target_pos != -1, "Anchor inside xCompressCU not found"

insert_pos = target_pos + len(anchor)
# 正確的 hook，包含 .currArea().Y()
hook = "\n  CuDataDumper::dump32x32IfTarget(*bestCS, partitioner.currArea().Y());"
code = code[:insert_pos] + hook + code[insert_pos:]

enc_cu.write_text(code, encoding="utf-8")
print("[OK] Hooked inside xCompressCU correctly.")