# kappa-bend-model
End-to-end pipeline that learns tip angle and full curvature shape κ(s) from ImageJ Kappa data, then integrates to θ(s) and reconstructs XY.

The model learns data from train files. The excel contains curveture of materials under different conditions.

<img width="453" height="468" alt="Kappa_c_20C" src="https://github.com/user-attachments/assets/2a40a26d-fded-4935-b0f8-4e05ad3f58ac" />

After training the model, it can predict other ratio and temperature.

ratio=5:1 and temp = 20℃:<img width="100" height="100" alt="r5_T20_xy_from_theta" src="https://github.com/user-attachments/assets/a2ba44c5-6fa6-4b34-8986-be3d0a833ff0" />
