# OBJECTIVE:
To develop a fake news detection model.
Reference Research Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10006567/
Dataset utilized:  Information Security and Object Technology (ISOT) Research Lab's Fake News Dataset
                   Link: https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/

# Challenges:
1. To decrease the complexity of the solution.
2. Hardware constraints for training and analyzing large datasets.
3. To improve F1 Score of the model.

# Methodology & Results:
Utilized the concept of Latent Variables for training the datasets which led to:
1. Reduction in the size of data being trained in terms of word length, which potentially countered
   the large training time due to hardware constraints.
2. Decreased the complexity of analyzing large datasets by reducing data size.
   Approximately took 15 min. to train on T4 GPU and over 50 min. on normal CPU.
3. Sharply enhanced F1 score of the model from 0.95 to 0.97.

# Tools Used:
Programming Language: Python.
Libraries: NLTK, Keras, TensorFlow, Scikit Learn, NumPy, Pandas, etc.
Domain: NLP.
Platform: Google Colab Workspace.

# New Development:
1. Integrated OCR to enable the working on image based texts.

# Use Case:
1. Can be utilized as a fact checker on news & social media platforms for countering missinformation and disinformation.

