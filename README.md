# SmartCrops

[![N|Solid](https://images.squarespace-cdn.com/content/v1/59765fd317bffcafaf5ff75c/1547948344308-FXIB0Q1ZSGXUI6J8IF6M/ke17ZwdGBToddI8pDm48kIodKJTXk-sXLaP9zru2Ry17gQa3H78H3Y0txjaiv_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z5QHyNOqBUUEtDDsRWrJLTmQyViSO8WVy1F2YAzXWvEVL64wLxcPm3xHpUDy_3Ao5ByoNR57GLqMs6V0zcEsuee/5b368a729e5f0.image.jpg)]()


### Problem to solve
Normal solutions to pests and diseases in crops such as fumigation are unviable for urban farms due to their proximity to the community. Alternative methods such as manually checking the health of each of the plants is time-consuming and inefficient. 
### Existing Solution
Existing solutions include using computer vision to classify crops and weeds from digital images to identify whether there are any weeds affecting the health of the plants.
### Our Solution
For our solution, we will be expanding further upon the current existing solution above. We will be using computer vision, along with various sensors to provide real time updates on the health of the plants to the farmers. Through computer vision and machine learning, we will not only identify whether there are weeds, we will also be analysing the leaves of the plants for bite marks to identify any pest infestations. Through water and temperature sensors, we will be monitoring the soil conditions and taking actions to ensure optimal conditions for the plants to grow in. Finally we will be displaying the data in a dashboard which will provide alerts to the farmer about the health of his plants and show the location of the unhealthy plants if any. 

### Technology Stack
> Azure Cognitive Services
> Azure ML
> Azure Function Apps
> Azure SQL
> Power BI

## Usage

Structure your data as follows:

	data/
		training/
			class_a/
				class_a01.jpg
				class_a02.jpg
				...
			class_b/
				class_b01.jpg
				class_b02.jpg
				...
		validation/
			class_a/
				class_a01.jpg
				class_a02.jpg
				...
			class_b/
				class_b01.jpg
				class_b02.jpg
				...

### Features Extraction
Solution Architecture:

Function apps run code 1 every hour, code 2 every day, code 3 every week.
Code 1: Collect live data from the camera. Humidity. Temperature. Save into some database. Plug in.
Code 2: Classifies good and bad crops, crops with holes (To forecast nutritional needs)
Assume bad crops ratio is less than 20%. We use some unsupervised learning algo to divide data into 2 sets, the first set has 80% of data, the second set has 20% of data. Someone will pick the crops and decide which are truly bad crops.
Code 3: Classifies crop and weed seedlings. (Seedlings have different ages; Use RGB differentiate)
Check images of leaves for bite marks or signs of infection


### Model Results

### Implementation
  - Raspberry Pi (Adafruit)
```sh
$ pip install Adafruit
```

Sample data:
https://github.com/chuachinhon/weather_singapore_cch

To achieve superior performance without compromising the resource efficiency, EfficientNets rely on AutoML and compound scaling. The AutoML Mobile framework has helped to develop a baseline network of mobile sizes, EfficientNet-B0, which is then enhanced by the compound scaling method to obtain EfficientNet-B1 to B7.

With an order of magnitude better efficiency, EfficientNets achieves state-of-the-art precision on ImageNet: 

On ImageNet, with 66M parameters and 37B FLOPS, EfficientNet-B7 achieves the state-of-the-art 84.4 percent top-1/97.1 percent top-5 accuracy in a high precision system. At the same time the model on CPU inference is 8.4x smaller and 6.1x faster than the previous leader, Gpipe.

### Features
  - Temperature
  - Humidity
  - Camera Images
  - REST Computer Vision Image Analysis
https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/quickstarts/python-disk

### Architecture
Backend architecture proposed is similar to recommended architecture by large.
[![N|Solid](https://msdnshared.blob.core.windows.net/media/2018/04/pythonaci1.png)](https://msdnshared.blob.core.windows.net/media/2018/04/pythonaci1.png)

### Considerations
  - CV Image Analysis is used as the first layer of detection to inform end-users through Power Apps, if a stray object is located in some picture.
  - Architecture proposal is designed to work with assumptions that fundamental analysis will produce fruitful results.
  - Health metric confidence level is an estimate, produced by contextual anomaly detection of temperature/humidity time series data and machine vision.

**Academic Papers for Reference**

One research shows the use of support vector machine with specific feature extractions to classify crop and weed from digital images, with a high accuracy rate and no misclassifications in their test set.
https://www.cse.unr.edu/~bebis/CS479/PaperPresentations/WeedsClassificationSVM2012.pdf
Another research shows the use of k-means clustering technique to separate diseased crops from the healthy crops with image segmentation.
https://www.researchgate.net/profile/Tushar_Jaware/publication/326736136_Crop_Disease_Detection_using_Image_Segmentation/links/5b619a82458515c4b2574ec5/Crop-Disease-Detection-using-Image-Segmentation.pdf


Classification of crop types using machine learning methods for indirect parameter estimation.
https://www.researchgate.net/figure/Classification-of-crop-types-using-machine-learning-methods-for-indirect-parameter_tbl2_286088260
Crop Selection Method to maximize crop yield rate using machine learning technique 
We may not remove them once they are planted.
Use ML to find optimal time to harvest (Feature Extraction: Color, Size)
https://ieeexplore.ieee.org/abstract/document/7225403

A cognitive vision approach to early pest detection in greenhouse crops - https://www.sciencedirect.com/science/article/abs/pii/S0168169907002256
