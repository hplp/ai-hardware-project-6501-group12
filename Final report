### Final Report: Building an AI-Powered Skin Disease Classification System  

**Hardware**: Google Coral Dev Board  
**Dataset**: ISIC 2016 Challenge Dataset  
- **Training Data**: 900 dermoscopic lesion images with ground truth malignant status  
- **Test Data**: 379 images in the same format as the training data  
**Model**: ResNet-18  
- Fine-tuned with added fully connected layers, achieving 80.7% accuracy.  
**Challenges**: Model overfitting, small dataset, and class imbalance issues. Additionally, image quality was a key factor, so uploading high-resolution images was used instead of real-time camera capture.  

---

### Background and Introduction  

Skin diseases are among the most common health concerns worldwide, ranging from benign conditions like acne to life-threatening ones like melanoma. Early and accurate detection is critical to prevent disease progression, reduce healthcare costs, and improve patient outcomes.  

This tutorial walks through the development of an AI-powered skin disease classification system. The system uses a deep learning model to classify skin conditions from dermoscopic images, providing a preliminary diagnosis that could serve as a decision-support tool for dermatologists or a self-assessment guide for patients.  

Our solution integrates AI model (Resnet) and accessible hardware (Google Coral Dev Board) to create a functional and practical tool.  

---

### Step 1: Data Preparation  

1. **Data Collection**  
   The ISIC 2016 Challenge Dataset was used, which provides labeled dermoscopic images for "benign" and "melanoma" categories.  
   - Training data: 900 images  
   - Test data: 379 images  

2. **Data Preprocessing**  
   - **Resizing**: Images were resized to 128*128 pixels.  
   - **Normalization**: Images were converted to tensors and normalized for input to the model.  
   - **Label Encoding**: Labels were encoded as `0` (benign) and `1` (melanoma).  

3. **Data Augmentation**  
   - Techniques like flipping, rotation, and brightness adjustment were used to artificially increase the size and variability of the dataset, improving the model's robustness.  

### Step 2: Model Design  

1. **Model Architecture**  
   - **Base Model**: ResNet-18 pre-trained on ImageNet.  
   - **Customization**: The fully connected (FC) layer was replaced with:  
     - A 512-unit dense layer with ReLU activation.  
     - A single-unit output layer with a Sigmoid activation function for binary classification.  

2. **Loss Function and Optimization**  
   - **Loss Function**: Binary Cross Entropy with Logits Loss (BCEWithLogitsLoss), with weighted penalties to address class imbalance.  
   - **Optimizer**: Adam optimizer with a learning rate of \(0.001\).  

3. **Hardware Acceleration**  
   - Training and inference were performed on the Google Coral Dev Board to leverage its edge computing capabilities.  


### Step 3: Model Training  

1. **Training Process**  
   - Batch size: 32  
   - Epochs: 3  
   - Metrics: Accuracy, precision, recall  

2. **Challenges and Observations**  
   - **Overfitting**: The model performed significantly better on the training data than on the test data.  
   - **Class Imbalance**: Malignant samples were underrepresented, causing prediction bias toward benign classifications.  


### Step 4: Deployment and User Interaction  

1. **Deployment**  
   - The model was deployed on the Google Coral Dev Board.  

2. **User Interaction**  
   - Instead of using a low-resolution camera, users uploaded images to ensure classification accuracy.  
   - A small LED indicator on the Coral Dev Board flashes green for malignant and no flash for benign results.  


### Challenges and Solutions  

1. **Small Dataset**  
   - **Challenge**: Limited images reduced the model’s ability to generalize.  
   - **Solution**:  
     - Expand the dataset by sourcing more images from later ISIC datasets.  
     - Use GANs to synthesize additional samples.  

2. **Class Imbalance**  
   - **Challenge**: Malignant cases were underrepresented, leading to biased predictions.  
   - **Solution**:  
     - Oversample malignant samples or undersample benign ones.  
     - Use focal loss or similar loss functions to dynamically focus on underrepresented classes.  

3. **Overfitting**  
   - **Challenge**: Model overfitted the training data, resulting in poor test performance.  
   - **Solution**:  
     - Implement stronger regularization techniques, such as dropout and L2 weight decay.  
     - Introduce early stopping to halt training before overfitting occurs.  

4. **Image Quality Dependency**  
   - **Challenge**: Poor-quality camera images affected the model’s accuracy.  
   - **Solution**:  
     - Improve preprocessing pipelines to handle noisy or low-resolution images.  
     - Train the model with augmented low-quality images to increase robustness.  

### Future Potential  

The following improvements can help address the current challenges and enhance the system:  

1. **Advanced Data Techniques**  
   - Integrate active learning to selectively annotate new, high-value images.  
   - Explore semi-supervised learning to leverage unlabeled data.  

2. **Robustness and Generalization**  
   - Train the model on multi-center datasets to account for geographic and demographic diversity.  
   - Add multi-task learning to classify multiple diseases simultaneously.  

3. **Enhanced Deployment**  
   - Develop mobile applications to make the system more accessible.  
   - Implement explainable AI techniques to provide users with insights into classification decisions.  

4. **Improved User Interaction**  
   - Integrate higher-resolution sensors for real-time analysis.  
   - Add advanced feedback mechanisms, such as detailed condition reports.  

