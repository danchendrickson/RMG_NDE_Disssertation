sudo apt-get update --allow-releaseinfo-change -y
sudo apt-get remove --purge libreoffice* scratch3 vlc* wolfram* -y
sudo apt-get clean -y
sudo apt autoremove  -y
sudo apt-get --fix-missing -y
sudo apt-get dist-upgrade -y

sudo apt-get install build-essential python-dev python-pip -y
sudo apt-get install code xrdp -y
sudo apt-get install python-smbus -y
sudo apt-get remove python3-numpy -y

sudo pip3 install numpy matplotlib RPi.GPIO pandas smbus2

sudo apt-get install libatlas-base-dev -y

sudo raspi-config nonint do_i2c 0

cd /home/pi
mkdir Code
cd Code

git clone https://github.com/danchendrickson/RMG_NDE_Disssertation

cp '/RMG_NDE_Dissertation/Code on Pi Sensor/*' ~/Code
