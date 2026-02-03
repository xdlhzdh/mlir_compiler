#!/bin/sh
# 由 make run_tests [DOMAIN=ast|pass] 调用；$4=DOMAIN
CMAKE="$1"
BINARY_DIR="$2"
SCRIPT="$3"
SAVE_DOMAIN="${DOMAIN:-}"
DOMAIN="${4:-}"
[ "$DOMAIN" = '$(DOMAIN)' ] && DOMAIN=""
DOMAIN="${DOMAIN:-$SAVE_DOMAIN}"
exec "$CMAKE" -DDOMAIN="$DOMAIN" -DBINARY_DIR="$BINARY_DIR" -P "$SCRIPT"
