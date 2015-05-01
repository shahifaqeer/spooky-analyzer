#!/bin/sh

# This script will install spookyscan dependencies for Ubuntu linux.

set -e

if [ "$(id -u)" -ne 0 ]; then
    echo 'This script must be run as root.' 1>&2
    exit 1
fi

release="$(lsb_release -sc)"
apt-key adv --recv-keys --keyserver keyserver.ubuntu.com E084DAB9
printf "deb http://cran.us.r-project.org/bin/linux/ubuntu $release/\ndeb-src http://cran.us.r-project.org/bin/linux/ubuntu $release/\n" > /etc/apt/sources.list.d/spookyscan.list

sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get -y install r-base r-base-dev python-rpy2

echo 'install.packages("forecast", repos="http://cran.us.r-project.org")' | R --vanilla

