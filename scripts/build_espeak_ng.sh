#!/bin/bash
# Build espeak-ng from its official source (github.com/espeak-ng/espeak-ng) into a local
# prefix under the repo — no sudo/system install needed, and no third-party bundled-binary
# wrapper package (unlike e.g. `espeakng-loader`): this builds directly from the upstream
# project itself. Requires cmake + a C/C++ compiler + git (provided by pixi's conda
# dependencies — see pyproject.toml's [tool.pixi.dependencies]).
#
# Run via `pixi run build-espeak-ng`. Skips the build if already done (idempotent).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="$REPO_ROOT/.espeak-ng-prefix"
SRC_DIR="$REPO_ROOT/.espeak-ng-src"
ESPEAK_NG_REF="${ESPEAK_NG_REF:-1.52.0}"  # pin a release tag for reproducibility

if [ -f "$PREFIX/lib/libespeak-ng.so" ] || [ -f "$PREFIX/lib64/libespeak-ng.so" ]; then
    echo "espeak-ng already built at $PREFIX, skipping (delete it to force a rebuild)"
    exit 0
fi

if [ ! -d "$SRC_DIR" ]; then
    git clone --branch "$ESPEAK_NG_REF" --depth 1 \
        https://github.com/espeak-ng/espeak-ng.git "$SRC_DIR"
fi

cmake -S "$SRC_DIR" -B "$SRC_DIR/build" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_ASYNC=OFF \
    -DUSE_MBROLA=OFF \
    -DUSE_LIBSONIC=OFF \
    -DUSE_LIBPCAUDIO=OFF \
    -DUSE_KLATT=OFF \
    -DUSE_SPEECHPLAYER=OFF \
    -DBUILD_SHARED_LIBS=ON

cmake --build "$SRC_DIR/build" --target espeak-ng -j"$(nproc)"
cmake --build "$SRC_DIR/build" --target data
cmake --install "$SRC_DIR/build"

echo "Built espeak-ng into $PREFIX"
echo "Library:   $(find "$PREFIX" -name 'libespeak-ng.so*' | head -1)"
echo "Data path: $PREFIX/share/espeak-ng-data (or wherever --target data installed it — check above)"
