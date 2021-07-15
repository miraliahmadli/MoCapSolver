cd dataset
wget http://mocap.cs.cmu.edu/allc3d_0.zip http://mocap.cs.cmu.edu/allc3d_1a.zip http://mocap.cs.cmu.edu/allc3d_1b.zip http://mocap.cs.cmu.edu/allc3d_234.zip http://mocap.cs.cmu.edu/allc3d_56789.zip
unzip '*.zip'
mkdir all_c3d
mv subjects/ all_c3d/
rm *.zip
wget http://mocap.cs.cmu.edu/allasfamc.zip
unzip '*.zip'
rm *.zip
