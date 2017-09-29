#!/bin/bash

echo "[ Installing prerequisites ]"

echo "[ Installing build tools and system packages ]"

if which apt-get &> /dev/null ; then
	echo "[ - Debian/Ubuntu based system detected ]"

	if [ "$upgrade" == 0 ]; then
		sudo apt-get update
		sudo apt-get install -y python-pip python-dev build-essential gfortran python-virtualenv 
		sudo apt-get build-dep -y python-numpy
		# ubuntu installs very old versions of pip
		sudo -H pip install pip --upgrade
		sudo -H pip install virtualenvwrapper --upgrade
	fi
elif yum --version &> /dev/null ; then
	echo "[ - RHEL based system detected ]"

	if [ "$upgrade" == 0 ]; then
		sudo yum groupinstall -y development
		sudo yum install -y blas-devel lapack-devel gfortran python-devel python-pip python-virtualenv 
	fi
else
	echo "[ - Unknown system. Some prerequisites will not be installed ]"
    read -p "Do you want to continue nevertheless?" yesno

    if [[ $yesno =~ ^[Yy] ]]; then
        echo "Continuing with the installation."
    else
        exit 1
    fi

fi


echo "[ Creating virtualenv in .env ]"
virtualenv .env

source .env/bin/activate

echo "[ Installing python packages ]"

pip install --upgrade pip
pip install -r requirements.txt --upgrade

echo "[ Building C modules ]"

make ccore

echo "[ Running tests ]"

py.test

echo "[ Done ]"
echo "Remember to execute 'source activate-env' to activate the virtualenv"

