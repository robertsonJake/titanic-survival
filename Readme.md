# Titanic Survival Classification

I used the classic Titanic dataset to predict whether or not a given passenger would survive.
I achieved 83.69% on a CV of f1_macro using SVM Classification. 
The confusion matrix gave me an F1 weighted average of:
SVM
              precision    recall  f1-score   support

           0       0.87      0.92      0.90       206
           1       0.87      0.80      0.83       139

   micro avg       0.87      0.87      0.87       345
   macro avg       0.87      0.86      0.87       345
weighted avg       0.87      0.87      0.87       345

('CV score: ', 0.8369020202473793)

If you are interested in more about this dataset and the Kaggle competition, check out https://www.kaggle.com/c/titanic 