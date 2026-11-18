#!/usr/bin/env bash
# Deploy ke npmjs. Pemakaian:
#   npm run deploy            → publish versi di package.json dengan tag "latest"
#   npm run deploy:dry        → simulasi (tidak publish)
#   npm run tag:legacy        → beri dist-tag "v0" ke 0.0.1 (npm i <pkg>@v0)
# Prasyarat: sudah `npm login` (akun dengan hak publish) dan working tree bersih.
set -euo pipefail

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="--dry-run"

NAME=$(node -p "require('./package.json').name")
VERSION=$(node -p "require('./package.json').version")

echo "→ Paket: $NAME@$VERSION"

if [ -z "$DRY" ]; then
  npm whoami >/dev/null 2>&1 || { echo "✗ Belum login. Jalankan: npm login"; exit 1; }
  [ -z "$(git status --porcelain)" ] || { echo "✗ Working tree belum bersih (commit dulu)"; exit 1; }
  if npm view "$NAME@$VERSION" version >/dev/null 2>&1; then
    echo "✗ $NAME@$VERSION sudah ada di npm. Naikkan versi: npm version <patch|minor|major>"; exit 1
  fi
fi

# prepublishOnly di package.json otomatis menjalankan `npm run build`
npm publish --access public --tag latest $DRY

if [ -z "$DRY" ]; then
  echo "✓ Terbit: $NAME@$VERSION (tag: latest)"
  echo "  Install: npm i $NAME@$VERSION"
fi
