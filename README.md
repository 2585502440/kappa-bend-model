# kappa-bend-model
End-to-end pipeline that learns tip angle and full curvature shape κ(s) from ImageJ Kappa data, then integrates to θ(s) and reconstructs XY.

The model learns data from train files. The excel contains curveture of materials under different conditions.

<img width="453" height="468" alt="Kappa_c_20C" src="https://github.com/user-attachments/assets/2a40a26d-fded-4935-b0f8-4e05ad3f58ac" />

After training the model, it can predict other ratio and temperature.

<img width="100" height="100" alt="r5_T20_xy_from_theta" src="https://github.com/user-attachments/assets/a2ba44c5-6fa6-4b34-8986-be3d0a833ff0" />
ratio=5:1 and temp = 20℃
<img width="100" height="100" alt="r5_T30_xy_from_theta" src="https://github.com/user-attachments/assets/2ab7feba-5d5c-40ba-9620-b14389cc2987" />
ratio=5:1 and temp = 30℃
<img width="100" height="100" alt="r5_T40_xy_from_theta" src="https://github.com/user-attachments/assets/1267c6dc-4fba-42b9-88a4-6565e73e2596" />
ratio=5:1 and temp = 40℃
<img width="100" height="100" alt="r5_T50_xy_from_theta" src="https://github.com/user-attachments/assets/fde413a2-5148-4261-936a-a7f766aa7cb6" />
ratio=5:1 and temp = 50℃
<img width="100" height="100" alt="r5_T60_xy_from_theta" src="https://github.com/user-attachments/assets/81cf3419-06a9-42a6-b5a0-c464d5a80baa" />
ratio=5:1 and temp = 60℃
