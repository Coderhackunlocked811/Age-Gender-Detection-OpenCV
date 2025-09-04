# 🧑‍🤖 Age & Gender Detection with OpenCV

This project demonstrates **real-time Age and Gender Detection** using **OpenCV** and pre-trained **Caffe deep learning models**.  
It captures live video from a webcam, detects faces, and predicts **gender (Male/Female)** and **age group** with high accuracy.  

---

## 🚀 Features
- Real-time face detection using Haar Cascade Classifier.  
- Predicts **Gender** → Male / Female.  
- Predicts **Age Groups** → (0–2), (4–6), (8–12), (15–20), (25–32), (38–43), (48–53), (60–100).  
- Uses pre-trained Caffe models with OpenCV’s `dnn` module.  
- Frame optimization → runs prediction every few frames for smoother performance.  

---

## 📂 Project Structure
Age-Gender-Detection-OpenCV/
│-- age_gender_detector.py # Main script
│-- age_deploy.prototxt # Model config (Age)
│-- age_net.caffemodel # Pre-trained model (Age)
│-- gender_deploy.prototxt # Model config (Gender)
│-- gender_net.caffemodel # Pre-trained model (Gender)


---

## 🔧 Requirements
- Python 3.x  
- OpenCV  
- NumPy  

Install dependencies:
```bash
pip install opencv-python numpy

**How to Run**

Clone the repository:

git clone https://github.com/<your-username>/Age-Gender-Detection-OpenCV.git
cd Age-Gender-Detection-OpenCV


Run the script:

python age_gender_detector.py


Press q to quit the webcam window.
