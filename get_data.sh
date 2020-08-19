cd input_data

wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip
wget https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip 
wget https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip 

unzip train2014.zip
unzip val2014.zip

mv kfahv87nfe8ax910l85dksyl2q212voc.zip Set5_SR.zip
mv igsnfieh4lz68l926l8xbklwsnnk8we9.zip Set14_SR.zip
mv qgctsplb8txrksm9to9x01zfa4m61ngq.zip BSD100_SR.zip

mkdir BSD100

unzip Set5_SR.zip
unzip Set14_SR.zip
unzip BSD100_SR.zip

mv image_SRF_2 BSD100/
mv image_SRF_3 BSD100/
mv image_SRF_4 BSD100/

rm train2014.zip
rm val2014.zip
rm Set5_SR.zip
rm Set14_SR.zip
rm BSD100_SR.zip
rm readme.txt

cd ..