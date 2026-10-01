#!/bin/bash
# The hold between Target C's volume window (target_c_volume_window.sh, on the box) and its Mac
# side (target_c_volume_mac.sh). Deployed next to the window as hold.sh.
#
# Each window holds in its own directory, hold/<its start stamp>/, never reused or removed. A
# second window, for a run the first didn't finish, can't mistake the first one's markers for its
# own, and every window's hold stays on record. The markers:
#   ready    the window's server is up; holds the hold's deadline, in epoch seconds (box clock)
#   started  the Mac's generator has begun
#   done     the Mac is finished, and the window can end
#   closed   the window has exited, whatever happened
#
#   hold.sh state           (the Mac) the newest hold, as "<dir> <deadline or -> <open|closed>";
#                           nothing before the first window
#   hold.sh wait DIR PATTERN [URL]
#                           (the window) waits, and prints why it stopped:
#                           - done;
#                           - timeout: the deadline in DIR/ready passed;
#                           - no client: no `started` NOCLIENT_MAX seconds after ready (1200);
#                           - idle: the server at URL processed no request for IDLE_MAX seconds
#                             (900) after `started`, as when the Mac died mid-run;
#                           - server gone: no process matches PATTERN.
#                           The last three end the hold early, so production isn't down for
#                           nothing. A server the thermal governor has paused still counts as
#                           busy: its /metrics doesn't answer.
#
# The holds are next to this script, where the window keeps them. POLL (seconds between checks,
# default 30) is set from the environment only to test the waiting.
set -u
D=$(cd "$(dirname "$0")" && pwd)
case "${1:-}" in
  state)
    h=$(ls -1d "$D"/hold/*/ 2>/dev/null | tail -1)
    h=${h%/}
    [ -n "$h" ] || exit 0
    echo "$h $(cat "$h/ready" 2>/dev/null || echo -) $([ -e "$h/closed" ] && echo closed || echo open)"
    ;;
  wait)
    dir=${2:?hold.sh wait DIR PATTERN [URL]}
    pattern=${3:?hold.sh wait DIR PATTERN [URL]}
    url=${4:-}
    deadline=$(cat "$dir/ready")
    since=$(date +%s)
    idle_since=
    while :; do
      [ -e "$dir/done" ] && { echo done; exit 0; }
      now=$(date +%s)
      [ "$now" -lt "$deadline" ] || { echo timeout; exit 0; }
      if [ ! -e "$dir/started" ]; then
        [ $((now - since)) -lt "${NOCLIENT_MAX:-1200}" ] || { echo "no client"; exit 0; }
      elif [ -n "$url" ]; then
        busy=$(curl -sf -m 10 "$url/metrics" |
               awk '$1 == "llamacpp:requests_processing" { print $2 }')
        if [ "$busy" = 0 ]; then idle_since=${idle_since:-$now}; else idle_since=; fi
        if [ -n "$idle_since" ] && [ $((now - idle_since)) -ge "${IDLE_MAX:-900}" ]; then
          echo idle; exit 0
        fi
      fi
      pgrep -f "$pattern" >/dev/null || { echo "server gone"; exit 0; }
      sleep "${POLL:-30}"
    done
    ;;
  *)
    echo "usage: hold.sh state | hold.sh wait DIR PATTERN [URL]" >&2
    exit 2
    ;;
esac
