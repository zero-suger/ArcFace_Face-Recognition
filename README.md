# ArcFace Face Recognition implementation with PyTorch  :umbrella:


------------

`ArcFace, or Additive Angular Margin Loss`, is a loss function used in face recognition tasks. The softmax is traditionally used in these tasks. However, the softmax loss function does not explicitly optimise the feature embedding to enforce higher similarity for intraclass samples and diversity for inter-class samples, which results in a performance gap for deep face recognition under large intra-class appearance variations. 

**Docs to read : **

1. https://paperswithcode.com/method/arcface
1. https://insightface.ai/arcface
1. https://arxiv.org/abs/1801.07698


------------

## WorkFlow :

1. `Gather custom data + CelebA ` datasets 

1.  Use `ms1mv3_r50_onegpu.py ` to train my dataset with batch 4 and epoch 50.

1.  Applying ** inference.py ** to turn my input value to embedings or numpy. 

1.  Apply  **angular_distance.py** to find angular distance

1.  Use   **similarity_percentage.py ** to find the percentage of similarity which is predicted 
 
    by weights.

------------

**In my case:** I just trained ArcFace with small dataset with 100 images in with one identity  So I get low similarity percentage. Use Feature Extraction and Fine Tune to improve model performance.


# Thank You :lollipop: :lollipop: :lollipop:

Contact with me to download custom trained weights (I did support the repo with weights link bcz the weights are trained with custom dataset which has my private images): **uacoding01@gmail.com**
