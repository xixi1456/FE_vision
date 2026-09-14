sleep 5
cd "$(dirname "$(readlink -f "$0")")" || exit 1
screen \
    -L \
    -Logfile logs/$(date "+%Y-%m-%d_%H-%M-%S").screenlog \
    -d \
    -m \
    bash -c "./watchdog.sh"
