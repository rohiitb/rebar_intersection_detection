# rebar_intersection_detection

<img src='images/reb_gif.gif'>
In this project, we propose a novel approach to find rebar intersections in a dense multilayered rebar network using Intel Realsense RGBD camera.
We obtained depth images from the RGBD camera and processed the noisy pointcloud to extract the rebar intersection points and their pose in the global frame.
<br>
<b>Click on the link to view the results : <a href="https://youtu.be/VeuRKfGhZqA"> Video</a></b>

## Results

For single layer rebar network : <br>

<table>
  <tr>
      <td align = "center"> <img src="./images/raw_image.PNG"> </td>
      <td align = "center"> <img src="./images/int_pose_2d.PNG"> </td>
  </tr>
  <tr>
      <td align = "center"> Raw images from the RGBD</td>
      <td align = "center"> Detected intersections</td>
  </tr>
</table>

<br>

For multilayer rebar network : <br>

<table>
  <tr>
      <td align = "center"> <img src="./images/with_obj_raw.PNG"> </td>
      <td align = "center"> <img src="./images/with_obj_int_pose_2d.PNG" width="500" height="320"> </td>
  </tr>
  <tr>
      <td align = "center"> Raw images from the RGBD</td>
      <td align = "center"> Detected intersections</td>
  </tr>
</table>

