# MICD
source code and dataset for the paper titled "MICD: More Intra-Class Diversity in Few-shot Text Classification with Many Classes"
![image](https://github.com/rayjang/MICD/assets/9244296/139208f9-b010-4d34-9665-272bf496bca7)

### Abstract
Deep learning has been successfully applied to various text classification tasks, such as document categorization and intent classification. It is difficult to apply traditional deep learning models to various applications with limited data scenarios since deep learning models require large amounts of labeled data. To handle limited data scenarios, few-shot learning has gained interest and achieved remarkable performance. However, existing few-shot classification methods aim to classify a limited of classes, typically ranging from 5 to 10. Many real-world tasks require classification for many classes. Few-shot classification for many classes has rarely been studied and it is a challenging problem. Distinguishing differences among many classes is more difficult than distinguishing the differences among small classes. To address this problem, we newly propose a few-shot text classification model for many classes called MICD(More Intra-Class Diversity in Few-shot Text Classification with Many Classes). By enhancing intra-class diversity at both data and model levels, our MICD effectively finds generalizable decision boundaries even with a small amount of data. Experimental results on the real datasets show that our MICD outperforms the existing methods.

### Main Concept of the proposed MICD model
![image](https://github.com/rayjang/MICD/assets/9244296/f2301ece-7c2d-41a0-afb4-7262055d77d9)
To address a many-class few-shot text classification task, our proposed model MICD, as shown in the above figure, comprises of two parts for (1) intra-class diversity contrastive learning(ICDCL) and (2) intra-class augmentation(ICA)

- (1)  ICDCL produces an encoder that embeds all training samples to find generalizable decision boundaries well in a few-shot classifier. ICDCL consists of two components: text pair generation with hard positive sampling and a generalizable encoder for many classes with intra-class diversity contrastive loss.
- (2) In the meta-testing stage, ICA effectively classifies many classes based on a few support data by improving intra-class diversity of data, which resolves a key issue of data scarcity in a few-shot scenario. ICA is composed of two components: one is for support set generation, which
aims to select various samples that can represent data distribution. Another is augmenting the support sets via intra-class mix-up.
  
### Dataset
- Bank77
- Medium
- WOS(Web of Science)
- R&D (Our new dataset)


### Results
Our MICD surpasses all other comparison model on four datasets in the 30-way 5-shot setting as shown in the below table.
![image](https://github.com/rayjang/MICD/assets/9244296/cf6fe56c-4683-4d13-8450-40a265ef4a23)

### Source Code Detail
- Generate_dataset_with_large_classes.ipynb: the source code for creating a training dataset for the 30-way 5-shot classification task. The code combines the train and test data from the existing dataset and splits them in a way that the classes do not overlap between the train and test data. The number of classes in the test data is set to 30, while the number of classes in the train data is set to the total number of classes minus 30. You can generate multiple versions of the training dataset by using different random seeds for each dataset.

- without_iterative_sampling_for_support_set.py: train and inference source code of our MICD model with ICDCL and intra-class mix-up of ICA. In this case, we select support sets for training the classifier randomly instead of our selection method of diverse support sets)

- with_all_methods_including_iterative_sampling_for_support_set.py: train and inference source code of our MICD model with all techniques(ICDCL, ICA)

