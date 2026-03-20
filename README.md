# snow_simulation

Based on https://github.com/thu-ml/3D_Corruptions_AD

---

```
# PointPillars
==========BBOX_2D==========
Pedestrian AP@0.5: 64.6159 61.3713 57.6031
Cyclist AP@0.5: 86.2569 73.0707 70.1706
Car AP@0.7: 90.6471 89.3323 86.6528
==========AOS==========
Pedestrian AOS@0.5: 49.2289 46.5926 43.7520
Cyclist AOS@0.5: 85.0412 69.0913 66.2914
Car AOS@0.7: 90.4754 88.6834 85.7248
==========BBOX_BEV==========
Pedestrian AP@0.5: 59.1745 54.3432 50.5029
Cyclist AP@0.5: 84.4268 67.1347 63.7455
Car AP@0.7: 89.9664 87.9104 85.7638
==========BBOX_3D==========
Pedestrian AP@0.5: 51.4551 47.9575 43.8407
Cyclist AP@0.5: 81.8821 63.6698 60.9022
Car AP@0.7: 86.6348 76.7492 74.1609

==========Overall==========
bbox_2d AP: 80.5067 74.5914 71.4755
AOS AP: 74.9152 68.1224 65.2561
bbox_bev AP: 77.8559 69.7961 66.6707
bbox_3d AP: 73.3240 62.7922 59.6346
```

```
[folder setting]
git clone https://github.com/thu-ml/3D_Corruptions_AD
git clone --recursive https://github.com/SysCV/LiDAR_snow_sim.git 

[download snowflak]
wget https://avi.ethz.ch//publications/2022/lidar_snow_simulation/snowflakes.zip
```

**These all files are in the 3D_Corruptions_AD project folder!**

```
[running simulation codes on dataset]
python3 generate_snow_[dataset].py --severity 0
```

---
