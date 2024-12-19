# Live Traffic Camera Analysis for Improved Navigation

## Introduction

Traditional navigation systems (like Google Maps) rely on historical traffic data as well as real-time traffic information from users to estimate traffic. However, this approach is not always accurate as it may not include real-time traffic information from users who are using a different navigation system, or not using a navigation system at all. Additionally, it has been shown that this information can be [easily manipulated by malicious users](https://www.wired.com/story/99-phones-fake-google-maps-traffic-jam/) to create fake traffic jams, and make empty roads appear congested.

<video controls poster="cover.jfif" style="max-width: 100%">
  <source src="cover.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

In this project, we propose an approach that uses live traffic camera feeds to estimate traffic. We will use computer vision techniques to analyze live traffic camera feeds from US DOT APIs and estimate traffic based on the number of vehicles observed in the video feeds. This approach has several advantages over traditional navigation systems:

1. **Real-time Traffic Information**: By analyzing live traffic camera feeds, we can provide real-time traffic information to users, which is more accurate than historical traffic data or user-reported traffic information.
2. **No User Interaction Required**: Unlike traditional navigation systems, our approach does not require any user interaction. Users do not need to report traffic conditions to get real-time traffic information.
3. **Difficult to Manipulate**: Since our approach relies on live traffic camera feeds, it is difficult for malicious users to manipulate the traffic information.
4. **No additional infrastructure required**: Live traffic camera feeds are already installed in many cities around the world, and can be accessed for free or at a low cost. By using existing infrastructure, our approach is scalable and cost-effective.
5. **Privacy-preserving**: Our approach does not rely on user location data or personal information, and it does not use facial recognition or license plate recognition to identify individual vehicles. Therefore, it preserves user privacy.

## Related Work

Our approach is inspired by several previous works that have used computer vision techniques to analyze traffic from live camera feeds. We draw significant inspiration from Eckert et al. [[1]](#1) who developed a CV-based system that periodically extracts counts of different types of vehicles from traffic camera feeds. They used a YOLOv3 object detection model trained on the COCO dataset [[2]](#2) from imagery extracted and aggregated from multiple Canadian traffic APIs.

TrafficSensor [[3]](#3), a system for automatic vehicle tracking and classification on highways using deep learning algorithms captured by a fixed camera, also uses YOLOv3/v4-based trained networks. Their second module uses a spatial association algorithm along with a KLT tracker to implement vehicle tracking.

On the other hand, Peppa et al. [[4]](#4) propose an end-to-end framework for CCTV-based urban traffic volume detection and prediction that employs Faster R-CNN for vehicle detection, followed by SARIMAX, random forest, and LSTM models for predicting traffic volume up to 30 minutes ahead, demonstrating effectiveness across various traffic scenarios and weather conditions.

Hsieh et al. [[5]](#5) present a novel vehicle surveillance system that leverages adaptive image subtraction techniques for vehicle detection. One of the main contributions include an automatic scheme to detect lane dividing lines by analyzing trajectories, allowing lane width estimation without manual calibration. Vehicle tracking and classification employs a Kalman filter along with new features like “vehicle linearity” to distinguish between vehicle types. Through automation, the system 
exhibits significant improvements in accuracy, robustness and stability for traffic surveillance scenarios according to the experimental validation. 

Azimjonov et al. [[6]](#6) used a novel centroid-based vehicle tracking and data association algorithm for estimating vehicle trajectories in real-time. Their method outperforms traditional Kalman filter-based tracking algorithms by over 9% in terms of average tracking accuracy, and over 3 times in number of frames processed per second.

## Approach

Our approach consists of the following steps:

1. **Data Collection**: COCO dataset for pretrained YOLOv8 model and Top View vehicle dataset from kaggle for our implemetation. Video was obtained from iStockPhoto/ Stock images.
2. **Frame Extraction**: Static frame capture , for the video we used 5 frames per second
3. **Vehicle Detection**: We use a fine-tuned YOLOv8 model to detect vehicles in the extracted frames. Here, we aim to gauge traffic by counting the number of vehicles in the frame.
4. **Traffic Estimation**: We estimate traffic based on the number of vehicles detected in the images. We can use simple heuristics to classify traffic conditions (e.g., light, moderate, heavy) based on the number of vehicles detected.

 
 

## Experiments/Results
1. **Model Selection and Evaluation**: For the first approach we tried using YOLOv8, a popular real-time object detection renowned for its balance between speed and accuracy. We opted for YOLOv8 due to its proven track record in detecting vehicles accurately and swiftly, an important requirement for our traffic density estimation project. To ensure its efficacy, we rigorously evaluated its performance on the COCO dataset, focusing on its ability to detect vehicles with high precision. 

![baseline.png](/Images/baseline.png)

2. **Fine-tuning YOLO Model on Vehicle Images**: 
To improve the performance of the pre-trained YOLO model specifically for vehicle detection, we fine-tuned it using a dataset of vehicle images. We used optuna to fine tune the model.

Hyperparameters:
1. 'epochs': Range [50, 100, 150]
2. 'imgsz': Range [416, 640, 800]
3. 'patience': Range [30, 50, 70]
4. 'batch': Range [16, 32, 64]
5. 'optimizer':  ['SGD', 'Adam', 'AdamW']
6. 'lr0': Log-uniformly select a value between 1e-6 and 1e-2
7. 'lrf': Uniformly select a value between 0.1 and 1.0
8. 'dropout': Uniformly select a value between 0.1 and 0.5
9. 'seed': Choose from [0, 42, 123]

![val_batch1_pred.jpg](/Images/train/val_batch1_pred.jpg)

3. **Evaluating Accuracy** 
After fine-tuning, we evaluated the model's accuracy on a separate test set of 	vehicle images to assess its performance and ensure it meets the desired criteria. 
![img1](/Images/Comparison_Graphs/img1.png)
![img2](/Images/Comparison_Graphs/img2.png)
![img3](/Images/Comparison_Graphs/img3.png)
![img4](/Images/Comparison_Graphs/img4.png)

4. **Running Pre-trained Model on Video** 
We also ran the pre-trained YOLO model on a video file (MP4) to test its performance in real-time object detection and tracking scenarios. 

## What's Next?

1. **Fine-tuning vehicle detection and traffic estimation**: We plan to further fine-tune our vehicle detection model and traffic estimation algorithm to further improve the accuracy of our traffic estimation. Since this is a critical component of our system and requires more involvement, we plan to dedicate most of the time until the final deliverable to this task. Anticipated completion: March 10 (~2 weeks)
2. **Setting up an end-to-end pipeline**: Although we've setup the pipeline for steps 1-4, our experimentation uses pre-fetched data, and we need to set up a real-time pipeline that can process live traffic camera feeds. For our final deliverable, this will be an end-to-end pipeline that will accept a source and destination, and provide real-time traffic information about the suggested route based on live traffic camera feeds. We will do this in parallel with above task. Anticipated completion: March 10 (~2 weeks)
3. **Developing a web-based dashboard**: We plan to develop a web-based dashboard that will allow users to enter a source and destination, and show the live traffic conditions along the suggested route on a map. The dashboard will use the traffic information estimated from the live traffic camera feeds to show the traffic conditions in real-time. Since this is not a critical component of our system, we plan to work on this towards the end of the project. Anticipated completion: March 15.
4. **Finalizing the report**: We plan to finalize the report by summarizing our approach, discussing the final results, and outlining the future work. Anticipated completion: March 17.
5. **[OPTIONAL] Deployment**: We plan to deploy our system on a cloud platform (e.g., AWS, GCP). We will also explore the possibility of integrating our system with existing navigation apps (e.g., Google Maps) to provide real-time traffic information to users.

## Team Member Contributions

1. Aishwarya Vijaykumar Sheelvant: Tasks 1, 2, 4
2. Ayushi Chakrabarty: Tasks 1, 2, 4
3. Kaustubh Odak: Tasks 2, 3, 4

## References

<a id="1">[1]</a> J. Eckert and A. Al-Habashna, “Traffic volume estimation from traffic camera imagery: Toward real-time traffic data streams,” www150.statcan.gc.ca, 2022. https://www150.statcan.gc.ca/n1/pub/18-001-x/18-001-x2022001-eng.htm
(Accessed: 22 February 2024).

<a id="2">[2]</a> T.-Y. Lin et al., “Microsoft COCO: Common Objects in Context,” arXiv.org, 2014. https://arxiv.org/abs/1405.0312

<a id="3">[3]</a> J. Fernández, J. M. Cañas, V. Fernández, and S. Paniego, “Robust Real-Time Traffic Surveillance with Deep Learning,” Computational Intelligence and Neuroscience, vol. 2021, pp. 1–18, Dec. 2021, doi: https://doi.org/10.1155/2021/4632353.

<a id="4">[4]</a> M. V. Peppa et al., “Towards an End-to-End Framework of CCTV-Based Urban Traffic Volume Detection and Prediction,” Sensors, vol. 21, no. 2, p. 629, Jan. 2021, doi: https://doi.org/10.3390/s21020629.

<a id="5">[5]</a> J.-W. . Hsieh, S.-H. . Yu, Y.-S. . Chen, and W.-F. . Hu, “Automatic Traffic Surveillance System for Vehicle Tracking and Classification,” IEEE Transactions on Intelligent Transportation Systems, vol. 7, no. 2, pp. 175–187, Jun. 2006, doi: https://doi.org/10.1109/tits.2006.874722.

<a id="6">[6]</a> J. Azimjonov, A. Özmen, and M. Varan, “A vision-based real-time traffic flow monitoring system for road intersections,” Multimedia Tools and Applications, Feb. 2023, doi: https://doi.org/10.1007/s11042-023-14418-w.
