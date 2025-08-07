# SolarGuard: Deep Learning for Solar Panel Defect Detection

## Overview  
SolarGuard automates solar panel inspections by using deep learning to classify and detect surface defects like dust, bird droppings, snow, and physical/electrical damage. This enables faster, cost-effective maintenance and maximizes energy output.

## Features  
- **Classification:** Categorizes panels into six conditions: Clean, Dusty, Bird-Drop, Electrical Damage, Physical Damage, Snow-Covered  
- **Object Detection:** (Optional) Localizes defects with bounding boxes for precise maintenance  
- **Streamlit App:** Upload images and get real-time predictions with visual defect highlights

## Use Cases  
- Automated inspections reducing manual effort  
- Optimized cleaning and repair scheduling  
- Efficiency monitoring for better energy yield  
- Integration with smart solar farm systems

## Dataset  
Custom annotated dataset with six classes representing common solar panel conditions and defects.

## Methodology  
1. **Data Prep:** Cleaning, augmentation, resizing, and annotation  
2. **Modeling:** CNNs (EfficientNet, ResNet, MobileNet) for classification; YOLOv8 for detection  
3. **Evaluation:** Accuracy, Precision, Recall, F1-score, mAP, IoU  
4. **Deployment:** Streamlit app for real-time use
