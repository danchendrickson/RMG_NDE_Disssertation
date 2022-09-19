#update OS
sudo apt-get update --allow-releaseinfo-change -y
#Delete unneeded programs
sudo apt-get remove --purge libreoffice* scratch3 vlc* wolfram* -y
sudo apt-get clean -y
sudo apt autoremove  -y
sudo apt-get --fix-missing -y

#second software update
sudo apt-get dist-upgrade -y

#install needed software
sudo apt-get install build-essential python-dev python-pip -y # python
sudo apt-get install code xrdp -y #remote desktop
sudo apt-get install python-smbus -y #python packages
sudo apt-get remove python3-numpy -y #remove incorrect python package,w ill get new version

sudo pip3 install numpy matplotlib RPi.GPIO pandas smbus2 # use pip to get needed pacakges

sudo apt-get install libatlas-base-dev -y # get python package needed for sensor

sudo raspi-config nonint do_i2c 0 #enable sensor in code

# get Code needed
cd /home/pi
mkdir Code
cd Code

git clone https://github.com/danchendrickson/RMG_NDE_Disssertation

cp /home/pi/Code/RMG_NDE_Dissertation/SensorCode/* ~/Code

# allow for more than one sensor by changing secondary pins to be GPIO pins
cd /boot
sudo nano config.txt
#    Add lines: 
#    dtoverlay=i2c-gpio,bus=4,i2c_gpio_delay_us=1,i2c_gpio_sda=23,i2c_gpio_scl=24
#    dtoverlay=i2c-gpio,bus=3,i2c_gpio_delay_us=1,i2c_gpio_sda=17,i2c_gpio_scl=27

#set the python to run at startup
sudo nano /etc/rc.local
#    add line sudo python /home/pi/Code/Prospectus/RecordAcceleration.py

#turn on ssh to allow remote login without using windowed interface
sudo systemctl enable ssh
sudo systemctl start ssh

