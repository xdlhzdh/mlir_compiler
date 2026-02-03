#!/bin/sh
# 由 make run [DOMAIN=ast|pass] [PASS=simple_pass] 调用
# 参数：$1=CMAKE, $2=BINARY_DIR, $3=run.cmake, $4=DOMAIN, $5=PASS, $6=RUN_HAVE_PASS
# Make 空变量时参数会少：无 DOMAIN/PASS 时只有 4 个参数（$4=1），仅无 PASS 时 5 个（$5=1）
CMAKE="$1"
BINARY_DIR="$2"
SCRIPT="$3"
SAVE_DOMAIN="${DOMAIN:-}"
SAVE_PASS="${PASS:-}"
if [ -n "${6:-}" ]; then
  DOMAIN="${4:-}"
  PASS="${5:-}"
  RUN_HAVE_PASS="$6"
elif [ "$5" = "0" ] || [ "$5" = "1" ]; then
  DOMAIN="${4:-}"
  PASS=""
  RUN_HAVE_PASS="$5"
elif [ "$4" = "0" ] || [ "$4" = "1" ]; then
  DOMAIN=""
  PASS=""
  RUN_HAVE_PASS="$4"
else
  DOMAIN="${4:-}"
  PASS="${5:-}"
  RUN_HAVE_PASS="${6:-0}"
fi
[ "$DOMAIN" = '$(DOMAIN)' ] && DOMAIN=""
[ "$PASS" = '$(PASS)' ] && PASS=""
DOMAIN="${DOMAIN:-$SAVE_DOMAIN}"
PASS="${PASS:-$SAVE_PASS}"
exec "$CMAKE" -DDOMAIN="$DOMAIN" -DPASS="$PASS" -DRUN_HAVE_PASS="$RUN_HAVE_PASS" -DBINARY_DIR="$BINARY_DIR" -P "$SCRIPT"
