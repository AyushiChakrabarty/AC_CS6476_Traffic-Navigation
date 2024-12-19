# Live Traffic Camera Analysis for Improved Navigation

> ⚠️ Relevant rubric items have been referenced using the 🗒️ icon throughout the report.

## Introduction

> 🗒️ High-level topic description
 
The primary objective of this project is to evaluate the performance of Faster R-CNN and YOLOv8 in detecting vehicles from live traffic camera feeds and determine which model is more suitable for real-time traffic monitoring applications. The significance of this project lies in its potential to provide valuable insights into the performance of two widely used object detection models, Faster R-CNN and YOLOv8, in the context of real-time vehicle detection for traffic monitoring applications. By comparing the strengths and weaknesses of these models, the project aims to determine the most suitable option for delivering accurate and reliable real-time traffic information to commuters and transportation authorities.
 
> 🗒️ Includes:
> - Why should people care about this project? If you succeed, what benefit will you provide and to whom?
> - What precise problem are you tackling? What is the desired goal?
 
This is particularly important as it addresses the limitations of traditional navigation systems, which often rely on historical data or user-reported information, which can be [susceptible to manipulation](https://www.wired.com/story/99-phones-fake-google-maps-traffic-jam/) and may not always provide a comprehensive picture of current traffic conditions. By leveraging live camera feeds, this project offers a more robust and privacy-preserving approach to traffic monitoring, as the video feeds are less prone to manipulation and do not require user involvement, thereby enhancing the overall reliability and trustworthiness of the traffic information provided to end-users.
 
> 🗒️
> Visual
 
<video controls poster="cover.jfif" style="max-width: 100%">
  <source src="cover.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
 
> 🗒️
> What is the expected input and output?
 
The expected input for this project is:
1.  Live traffic camera feeds, which can be obtained from the US DOT. We use 511GA/GDOT’s API to obtain feeds for our proof-of-concept.
2.  A dataset of vehicle images i.e., Top-View Vehicle Detection Image Dataset from Kaggle used for fine-tuning the pre-trained (on the COCO dataset) Faster R-CNN and YOLOv8 models.
 
The expected output of this project is:
1.  A comparative analysis of the accuracy, speed, and robustness of Faster R-CNN and YOLOv8 in detecting vehicles from the live traffic camera feeds.
2.  Insights into the strengths and weaknesses of each model in the context of real-time traffic monitoring applications.
3.  A recommendation on the most suitable object detection model for real-time traffic estimation, based on the evaluation results.
4.  A scalable and cost-effective traffic monitoring solution that can be deployed in various cities and regions, leveraging the existing infrastructure of live traffic cameras.

## Related Work

> 🗒️ Explaining Context: Clearly state what other people have done before to solve your specific problem. If your project brings together multiple areas, you will need to describe each area and what people have done in each.
 
Our approach is inspired by several previous works that have used computer vision techniques to analyze traffic from live camera feeds. We draw significant inspiration from Eckert et al. [[1]](#1) who developed a CV-based system that periodically extracts counts of different types of vehicles from traffic camera feeds. They used a YOLOv3 object detection model trained on the COCO dataset [[2]](#2) from imagery extracted and aggregated from multiple Canadian traffic APIs.

TrafficSensor [[3]](#3), a system for automatic vehicle tracking and classification on highways using deep learning algorithms captured by a fixed camera, also uses YOLOv3/v4-based trained networks. Their second module uses a spatial association algorithm along with a KLT tracker to implement vehicle tracking.

On the other hand, Peppa et al. [[4]](#4) propose an end-to-end framework for CCTV-based urban traffic volume detection and prediction that employs Faster R-CNN for vehicle detection, followed by SARIMAX, random forest, and LSTM models for predicting traffic volume up to 30 minutes ahead, demonstrating effectiveness across various traffic scenarios and weather conditions.

Hsieh et al. [[5]](#5) present a novel vehicle surveillance system that leverages adaptive image subtraction techniques for vehicle detection. One of the main contributions include an automatic scheme to detect lane dividing lines by analyzing trajectories, allowing lane width estimation without manual calibration. Vehicle tracking and classification employs a Kalman filter along with new features like “vehicle linearity” to distinguish between vehicle types. Through automation, the system 
exhibits significant improvements in accuracy, robustness and stability for traffic surveillance scenarios according to the experimental validation. 

Azimjonov et al. [[6]](#6) used a novel centroid-based vehicle tracking and data association algorithm for estimating vehicle trajectories in real-time. Their method outperforms traditional Kalman filter-based tracking algorithms by over 9% in terms of average tracking accuracy, and over 3 times in number of frames processed per second.

## Approach

> 🗒️ Method Overview 

Our approach consists of the following steps:

Models used: 

**FASTER RCNN**[[9]](#9)

Faster R-CNN is an advanced model for object detection, known for its accuracy and efficiency. It builds upon Region-based Convolutional Neural Network architecture, addressing its drawbacks in terms of speed and complexity. Faster R-CNN introduces a novel approach to generate region proposals and perform object detection in a single, unified framework. These are the main components of this architecture that make this network work effectively: 

**1.  Region Proposal Network** 
The region proposal network efficiently generates potential object bounding boxes within an image. The RPN behaves as a neural network that scans the entire image and predicts regions that can contain projects. These regions, often referred to as region proposals or anchor boxes, are proposed at multiple scales and aspect ratios to handle objects of various sizes and shapes. 

**2.  Feature Pyramid Network** 
Faster R-CNN uses a Feature Pyramid Network (FPN) to provide reliable object detection at various scales. By creating a pyramid of feature maps with varying resolutions, FPN enables the network to extract high-level semantic information as well as fine-grained features. The model is capable of efficiently detecting objects at various scales because of its hierarchical representation. 

**3. ROI pooling or ROIAlign** 
Faster R-CNN uses a technique known as Region of Interest (RoI) pooling, or RoIAlign, to extract features from these regions. The suggested regions must be aligned with the feature maps that the convolutional layers produced in this stage.  

*Faster RCNN Architecture*
![fasterrcnn.png](/Images/fasterrcnn.png)
     


**YOLOv** [[7]](#7)[[8]](#8)

Yolov8 has been widely used in the field of object detection. Its lightweight architecture along with easy implementation makes it a popular choice. The new version of Yolo makes it suitable for this task because it has the following architectural nuances: 

**1. The backbone, feature extractor :** It is responsible for extracting meaningful features from the input.  This is used to capture simple patterns in the initial layers, such as edges and textures.
 
**2.  Neck :** The neck acts as a bridge between the backbone and the head and performs feature fusion operations integrating contextual information. Basically the Neck assembles feature pyramids by aggregating feature maps obtained by the Backbone, in other words, the neck collects feature maps from different stages of the backbone. 
It also reduces the spatial resolution and dimensionality of resources to facilitate computation, a fact that increases speed but can also reduce the model's quality. 

**3.  Head :** serves as the final stage responsible for generating key outputs crucial for object detection tasks. It creates bounding boxes that delineate potential objects within the input image, providing precise spatial localization information. Concurrently, confidence scores are assigned to each bounding box, indicating the likelihood that an object exists within the specified region. These confidence scores aid in determining the reliability of object detections, guiding subsequent decision-making processes. Additionally, the head assigns categorical labels to the objects enclosed within the bounding boxes, facilitating semantic understanding by identifying the specific type or class of each detected object, such as vehicles, pedestrians, or traffic signs. Through these coordinated activities, the head of the YOLOv8 network enables comprehensive object detection capabilities, empowering applications across various domains, including surveillance, autonomous driving, and image analysis. 

![yolov8.jpeg](/Images/yolov8.jpeg)
*YOLOv8 Architecture*


**Proof of Concept**

We demonstrate the capabilities of our work through a simple web app consisting of a RESTful Flask server for running on-demand and localized inference, and a MapBox-driven mapping client platform for navigation and overlaying computed information. 

**1. Server:** A simple RESTful Flask (Python) HTTP server for finding the cameras on the requested route using shapely and 511GA, and performing object detection on frames captured from the livestreams on these cameras. These frames are annotated, compressed, base64-encoded, and attached to the response along with the count of vehicles and camera metadata. Each camera is processed concurrently using process-based parallelism to minimize the response time. 

**2. Client:** A vanilla TypeScript web application served using a Vite dev-server. It displays a map with navigational controls and traffic information sourced from MapBox client libraries and APIs. End-users can provide a source and destination, which is converted to a route using MapBox APIs and sent over to the server. Using the camera metadata from the response, the client then adds data sources and layers on the map for displaying the counts and annotated frames from the livestreams of the cameras on the route. 

![inputPipeline.png](/Images/inputPipeline.png)


## Contribution

> 🗒️ Includes:
> - Related work > Your project in context
> - Methods/Approach > Contribution
> - Methods/Approach > Intuition


Our approach is poised to excel for a few key reasons. Firstly, our comparative analysis between Faster R-CNN and YOLOv8 fine-tuned models introduces a fresh perspective to the field. This comparison sheds light on how these models perform in real-time object detection, an aspect that's been less explored in previous research. Additionally, our fine-tuning of the YOLOv8 model, specifically tailored for object detection from live camera feeds, ensures that the model is finely tuned for our task, potentially boosting its accuracy and reliability.

While previous works utilized YOLOv3 and Faster R-CNN models, our approach fine-tuned these models to yield highly optimized performance values. This fine-tuning process involved training on specific datasets relevant to your traffic analysis task. We achieved higher precision, recall, mAP50, mAP50-95, and fitness scores compared to the original models and even surpassing the performance of other related works.

Unlike some of the cited works which have focused on offline evaluation or simulations, our approach tested the fine-tuned YOLOv8 and Fast R-CNN models on real-time traffic feeds. Our approach demonstrates the ability to generalize across various traffic scenarios, potentially including different traffic densities, vehicle types, and environmental conditions crucial for practical deployment in diverse urban environments.

To the best of our knowledge, no free, open-source and/or consumer-facing navigational system exists that uses live traffic camera feeds and computer vision to provide end-users with annotated and aggregated traffic information. Our work aims to fill this gap by leveraging existing infrastructure and open-source tools to build a scalable, cost-effective, and privacy-preserving navigational system that provides real-time traffic information to users. Through our work, we demonstrate the feasibility of such a system and provide a proof-of-concept implementation that can be further developed into a full-fledged navigational system.

By integrating the latest advancements in computer vision techniques and methodology, we aim to push the boundaries of what's possible in real-time object detection, contributing to advancements in the field. Overall, our work not only enhances our understanding of different model architectures but also paves the way for more effective solutions in object detection from live camera streams, impacting various domains like surveillance and autonomous driving.


**Pipeline of approach**

> 🗒️ Visuals

>  Provide a diagram that helps to explain your approach. This diagram could describe your approach pipeline, a key component you are introducing, or another visual that you feel helps best explain what you need to convey your approach.  

![FlowChartCV.drawio.png](/Images/FlowChartCV.drawio.png)


## Experiments

> 🗒️ Experiment Setup

In this project, we performed grid search-based optimized fine-tuning on the pretrained YOLOv8 and Fast R-CNN models to enhance their performance in traffic analysis tasks. The purpose of our experiments was to improve the precision, recall, and overall effectiveness of these models in detecting and classifying vehicles in live traffic camera feeds. 

We conducted a series of experiments where we systematically varied hyperparameters such as learning rates, batch sizes, and regularization techniques using grid search. This allowed us to identify the optimal configuration for fine-tuning each model on our specific traffic analysis dataset.  We used optuna for finetuning.  
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

Fine-tuning on these values visibly led to an enormous percentage increase (discussed in detail in the results section) in the performance for both the YOLOv8 and Fast R-CNN baseline models leading to clearly more accurate predictions for the real time feed. 

![realtime.png](/Images/realtime.png)

**Input**

> 🗒️ Input Description

We received input data from two sources: 

- A dataset of vehicle images i.e., Top-View Vehicle Detection Image Dataset from Kaggle (https://www.kaggle.com/datasets/farzadnekouei/top-view-vehicle-detection-image-dataset) used for fine-tuning the pre-trained Faster R-CNN and YOLOv8 models. 
1. Consists of 536 images for training the model and 90 images for validation. Each image is standardized to a 640 x 640 resolution and is roughly of the size of 70kB. The dataset i provided with labels. It is meticulously annotated in the YOLOv8 format for top-view vehicle detection. There is only label in the dataset: Vehicle which includes all the vehicles. We used this dataset because vehicle subcategories is not of importance to our use case. 
2. An extra sample image and sample video that is not a part of either the training or the 	validation dataset. 
3. Total dataset size is 48 MB.

- Live traffic camera feeds, which can be obtained from the US DOT. We use 511GA/GDOT’s API to obtain feeds for our proof-of-concept. 

![input.png](/Images/input.png)


**Output**

> 🗒️ Desired Output Description


The output is in both qualitative and quantitative terms to help interpret the effectiveness of the model in detail. The model is tested on a sample image that is not present in the training or the validation dataset. Accurate bounding box predictions of the various kinds of vehicles define the model’s effectiveness through visual perception. On the other hand, metrics such as Precision, Recall, mAP50, mAP50-95, and Fitness provide a more detailed insight into the reliability of the model’s predictions under different conditions. The output description is provided in the results section in much more detail.

> 🗒️ Metric for success

The metrics used to gauge the success of the model are Precision, Recall, mAP50, mAP50-95, and Fitness. Each of these metrics is explained as shown below: 

**1. Precision:** 
Precision measures the accuracy of positive predictions made by a model. It is calculated as the ratio of true positive predictions to the total number of positive predictions made by the model (true positives + false positives). It would indicate how accurately the model identifies vehicles among all the detected objects. Higher precision means fewer false positives, which is crucial for accurately estimating the number of vehicles on the road. 

**2. Recall:** 
Recall measures the ability of a model to correctly identify all positive instances in the dataset. It is calculated as the ratio of true positive predictions to the total number of actual positive instances in the dataset (true positives + false negatives). It indicates how effectively the model captures all the vehicles present in the scene. Higher recall means fewer missed vehicles, which is important for capturing the full extent of traffic density. 

**3. mAP50 (mean Average Precision at IoU threshold 0.50):** 
mAP50 is a common metric used in object detection tasks. It computes the average precision across different levels of confidence scores for detected objects, considering only those detections with Intersection over Union (IoU) overlap with ground truth bounding boxes of at least 0.50. It provides an overall assessment of the model's performance in detecting vehicles with a moderate level of confidence. It considers both precision and recall at a specific IoU threshold, providing a balanced evaluation. 

**4. mAP50-95 (mean Average Precision across IoU thresholds from 0.50 to 0.95):** 
mAP50-95 calculates the average precision over a range of IoU thresholds from 0.50 to 0.95, providing a broader assessment of the model's performance across various levels of overlap between predicted and ground truth bounding boxes. It evaluates the model's consistency in detecting vehicles across different levels of overlap. A high mAP50-95 score indicates robust performance across a wide range of detection scenarios. 

**5. Fitness:** 
Fitness averages the mAP50 and mAP50-95 scores and takes into account both the model's performance at a moderate IoU threshold (0.50) and its consistency across a range of IoU thresholds from 0.50 to 0.95. A higher fitness score indicates a more reliable and effective model for this task. 

 
These metrics collectively provide a comprehensive evaluation of the object detection models' performance in estimating traffic density. They help assess the models' accuracy, completeness, and robustness in detecting vehicles, which are essential factors for reliable traffic density estimation in real-world applications. 



## Results
> 🗒️
> Baselines

> How do prior works perform on this task? It’s best to have a quantitative comparison using your metric for success. If that is not possible, a qualitative result will suffice.  

Our experimentation leads to a drastically improved performance when compared to the baseline models. 

![img5](/Images/Final_Evaluation_Results/Yolov8_Comparison.png)

This graph presents a comparison of evaluation metric values for the YOLOV8 model, comparing the original model (baseline) and a fine-tuned version. The metrics shown are precision, recall, mAP50 (mean Average Precision at 50% IoU), mAP50-95 (mean Average Precision across IoU thresholds from 50% to 95%), and fitness. 

**Inference:**
1. The fine-tuned YOLOv8 model outperforms the original YOLOv8 model across all the evaluated metrics - precision, recall, mAP50, mAP50-95, and fitness. 
2. The most significant improvement is seen in the mAP50 metric, where the fine-tuned model achieves a value of 0.952, compared to 0.521 for the original model. This represents a substantial 83% increase in the mean Average Precision at 50% IoU. 
3. The recall metric also shows a notable improvement, increasing from 0.475 in the original model to 0.928 in the fine-tuned version, a nearly 95% relative increase. 
4. The precision, mAP50-95, and fitness metrics all see improvements in the range of 20-30% when comparing the fine-tuned model to the original. 

![img6](/Images/Final_Evaluation_Results/FastRCNN_Comparison.png) 

Similarly, this graph compares the evaluation metric values for the original Faster R-CNN model and a fine-tuned Faster R-CNN variant.

**Inference:**
1. The fine-tuned model achieves a significantly higher precision value of 0.843 compared to 0.159 for the baseline, representing a more than 5-fold improvement. 
2. The fine-tuned model has a recall value of 1.000, indicating perfect recall, whereas the baseline model has a recall of 0.475, a substantial difference. 
3. The fine-tuned model outperforms the original Faster R-CNN model in the mAP50 (mean Average Precision at 50% IoU) metric, with a value of 0.960 compared to 0.639 for the original. 
4. Similar to mAP50, the fine-tuned model achieves a higher mAP50-95 value of 1.000, compared to 0.843 for the original Faster R-CNN. 
5. The fitness metric, which is a composite measure, shows a considerable improvement from 0.819 for the original model to 0.980 for the Faster R-CNN. 

Analyzing the two graphs provided, we can draw the following overarching conclusions: 

The fine-tuning and optimization efforts have resulted in substantial performance improvements for both the YOLOv8 and Faster R-CNN object detection models, when compared to their respective original baseline versions. These quantitative results clearly indicate success in developing enhanced version models that significantly outperform the baselines. 


> 🗒️
> Key Result Presentation

> Clearly present your key result. This should include both your performance according to your metric of success defined above and a qualitative output example from your system.  

The quantitative and qualitative key results for the YOLOv8 and the Fast R-CNN models are encapsulated as shown below. 

**YOLOv8 Model** 
 
Quantitative performance comparison:

![img7](/Images/Final_Evaluation_Results/YOLOv8_Figure.png)

 Qualitative performance comparison: 
 
![img8](/Images/Final_Evaluation_Results/Yolov8_original_prediction.png)

![img9](/Images/Final_Evaluation_Results/Yolov8_fine-tuned_prediction.png)

The two images provided demonstrate the performance difference between the original YOLOv8 model and the fine-tuned YOLOv8 model in detecting objects in a sample image. 
The first image shows the results of the original YOLOv8 model on the same sample image. While the model is able to detect some vehicles, the bounding boxes and confidence scores are noticeably less precise compared to the fine-tuned version. In contrast, the second image shows how the fine-tuned YOLOv8 model is able to accurately detect and label various vehicles on the highway, including cars, vans, and a truck, with high confidence scores ranging from 0.70 to 0.89. The model's ability to precisely identify and locate these objects is clearly visible. 

**Fast R-CNN Model** 

Quantitative performance comparison: 

![img12](/Images/Final_Evaluation_Results/FastRCNN_Figure.png)

Qualitative performance comparison: 
![img13](/Images/Final_Evaluation_Results/FastRCNN_Prediction.png)

![img14](/Images/Final_Evaluation_Results/FasterRCNN_Prediction.png)

In this case, the two images provided demonstrate the performance difference between the original Faster R-CNN model and the fine-tuned Faster R-CNN model in detecting objects in a sample image. 

The first image shows the results of the original Faster R-CNN model on the same sample image. This model performs poorly as it overestimates several objects to be vehicles and misclassifies miscellaneous objects under the same class. This can prove to be detrimental in accurately estimating the real-time traffic density as the overall number would always demonstrate dense traffic. In contrast, the second image shows how the fine-tuned Faster R-CNN model is able to accurately detect and label various vehicles on the highway, including cars, vans, and trucks. It addresses the shortcoming of the baseline model predictions.  

 This indicates that the fine-tuning process has significantly improved the model's object detection capabilities, making it a more reliable and effective tool for real-world applications. 

> 🗒️
> Key Result Performance

> Includes:
> - Specify what variants of your approach you tried to make progress towards finding a solution.  
> - Ultimately, describe your final results. Did your approach work? If not, explain why you believe it did not.  

The variants of our approach include: 

- **Hyperparameter Tuning:** Experimenting with different combinations of learning rates, batch sizes, regularization techniques, etc., to optimize model performance. 
- **Architecture Modifications:** Adjusting the architecture of the models, such as adding or removing layers and changing activation functions. This step did not lead to much difference in the overall performance of the model.  
- **Data Augmentation:** A 50% probability of horizontal flip was exclusively applied to the training set.   
- **Fine-Tuning Strategies:** Exploring different fine-tuning approaches, such as gradual unfreezing, layer freezing, or differential learning rates. 

Overall comparison of all the models: 

> 🗒️ Discussion

![img15](/Images/Final_Evaluation_Results/Barchart_Comparison_all.png)

This stacked bar chart provides a comparative visualization of the evaluation metric values for the original and fine-tuned versions of the YOLOv8 and Faster R-CNN object detection models. 

The key insights that we can derive from the chart are: 

**YOLOv8:** 
- The fine-tuned YOLOv8 model outperforms the original YOLOv8 model across all the evaluation metrics, with significant improvements in precision, recall, mAP50, mAP50-95, and fitness. 
- The most noticeable difference is in the recall metric, where the fine-tuned model achieves a much higher value compared to the original. 

**Faster R-CNN:** 
- Similar to YOLOv8, the fine-tuned variant of the Faster R-CNN model demonstrates substantial improvements over the original Faster R-CNN across all the evaluated metrics. 
- The Faster R-CNN model shows particularly strong performance in precision, recall, and mAP50, significantly exceeding the original Faster R-CNN's results. 


The stacked bar chart format allows for a clear and concise comparison of the relative performance of the original and fine-tuned/optimized versions of the YOLOv8 and Faster R-CNN models. The visual representation highlights the effectiveness of the refinement processes in enhancing the object detection capabilities of these models. 


## Discussion

We demonstrate the capability of our fine-tuned models with a proof-of-concept web app that provides on-demand real-time traffic information. While developing it, we looked into several open-source libraries for mapping, geometry and image/video processing and learned about techniques for parallelism and optimization. In the future, we would like to replace our on-demand computation strategy with a background task-queue-based mechanism that would intermittently perform object detection on all cameras. This would help us significantly decrease the response time and eliminate redundant computations with larger user bases.       

## Challenges Encountered

While developing the proof-of-concept, our first challenge was finding a good mapping library. While Google Maps was our obvious first choice, we ended up going with MapBox because of its feature-rich free tier. Our main challenge was achieving a reasonable response time. We considered processing all cameras at once using a task-queue but found the process of setting up and deploying message brokers/backends too challenging and complex for a simple proof-of-concept. We also attempted concurrency-based background scheduling, but that also did not work since our machines do not have the compute to process 3000+ cameras at once. We also encountered race conditions with this approach. We eventually ended up settling with on-demand and localized processing, which only processes the cameras on the requested route.

## Team Member Contributions

We acknowledge that we have individually filled out the team member contributions form as per the provided template/rubric.

## References

<a id="1">[1]</a> Government of Canada, A.A.-H. and James Eckert (2022) Reports on special business projects traffic volume estimation from traffic camera imagery: Toward real-time traffic data streams, Traffic volume estimation from traffic camera imagery: Toward real-time traffic data streams. Available at: https://www150.statcan.gc.ca/n1/pub/18-001-x/18-001-x2022001-eng.htm (Accessed: 18 April 2024). 

<a id="2">[2]</a> Lin, T.-Y. et al. (2014) ‘Microsoft Coco: Common Objects in Context’, Computer Vision – ECCV 2014, pp. 740–755. doi:10.1007/978-3-319-10602-1_48.

<a id="3">[3]</a> J. Fernández, J. M. Cañas, V. Fernández, and S. Paniego, “Robust Real-Time Traffic Surveillance with Deep Learning,” Computational Intelligence and Neuroscience, vol. 2021, pp. 1–18, Dec. 2021, doi: https://doi.org/10.1155/2021/4632353.

<a id="4">[4]</a> M. V. Peppa et al., “Towards an End-to-End Framework of CCTV-Based Urban Traffic Volume Detection and Prediction,” Sensors, vol. 21, no. 2, p. 629, Jan. 2021, doi: https://doi.org/10.3390/s21020629.

<a id="5">[5]</a> J.-W. . Hsieh, S.-H. . Yu, Y.-S. . Chen, and W.-F. . Hu, “Automatic Traffic Surveillance System for Vehicle Tracking and Classification,” IEEE Transactions on Intelligent Transportation Systems, vol. 7, no. 2, pp. 175–187, Jun. 2006, doi: https://doi.org/10.1109/tits.2006.874722.

<a id="6">[6]</a> J. Azimjonov, A. Özmen, and M. Varan, “A vision-based real-time traffic flow monitoring system for road intersections,” Multimedia Tools and Applications, Feb. 2023, doi: https://doi.org/10.1007/s11042-023-14418-w.


<a id="6">[7]</a> Pedro, J. (2023) Detailed explanation of yolov8 architecture - part 1, Medium. Available at: https://medium.com/@juanpedro.bc22/detailed-explanation-of-yolov8-architecture-part-1-6da9296b954e (Accessed: 18 April 2024). 

<a id="6">[8]</a> Ultralytics (2024) Yolov8, Ultralytics YOLOv8 Docs. Available at: https://docs.ultralytics.com/models/yolov8/#performance-metrics (Accessed: 18 April 2024). 


<a id="6">[9]</a> Ren, S. et al. (2017) ‘Faster R-CNN: Towards real-time object detection with region proposal networks’, IEEE Transactions on Pattern Analysis and Machine Intelligence, 39(6), pp. 1137–1149. doi:10.1109/tpami.2016.2577031. 

## Supplementary Material

All of our code is available under the same GitHub organization in the [code](https://github.gatech.edu/cs6476-sp24-team26/code) repository.
