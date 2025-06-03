# Face Detection App (using DNN model)

[Deep Neural Networks (DNN)](https://en.wikipedia.org/wiki/Deep_learning#Deep_neural_networks)

[ResNet10 Algorithm (paper)](https://ieeexplore.ieee.org/document/9724001)

---

Fonte das imagens: https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset?resource=download

Código adaptado de:
 - https://github.com/sr6033/face-detection-with-OpenCV-and-DNN
 - https://drlee.io/building-a-face-detection-app-with-streamlit-and-opencv-dnn-946bc9994fcd

---

## Instalação

### Clonar o projeto
    
    git clone https://github.com/fahadkalil/face_detection_app.git
    cd face_detection_app

### Criar e ativar o virtualenv para o projeto

- Windows
    
      python -m venv .venv
      .venv/Scripts/activate.bat

- Linux / MacOS
  
      python -m venv .venv
      source .venv/bin/activate

### Instalar dependencias
    
    pip install -r requirements.txt

### Para rodar (confira a saída do comando)

    streamlit run streamlit_app.py
