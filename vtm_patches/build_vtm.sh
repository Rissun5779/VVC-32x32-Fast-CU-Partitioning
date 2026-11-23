#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VTM_SRC="${PROJECT_ROOT}/external/VTM24.0"
BUILD_DIR="${PROJECT_ROOT}/build"
BIN_DIR="${PROJECT_ROOT}/bin"

mkdir -p "${BUILD_DIR}" "${BIN_DIR}"
cd "${BUILD_DIR}"

cmake "${VTM_SRC}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="${BIN_DIR}" \
  -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY="${BUILD_DIR}/lib" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${BUILD_DIR}/lib"

make -j4
mv -f "${VTM_SRC}/bin/"* "${BIN_DIR}/" 2>/dev/null || true
echo "[OK] EncoderAppStatic built directly at: ${BIN_DIR}/EncoderAppStatic"
