cd input_data

wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip

wget https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip
wget https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip

wget https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip

unzip train2014.zip
unzip val2014.zip

unzip Set5_SR.zip
unzip Set14_SR.zip
unzip BSD100_SR.zip

rm train2014.zip
rm val2014.zip

rm Set5_SR.zip
rm Set14_SR.zip
rm BSD100_SR.zip

cd ..