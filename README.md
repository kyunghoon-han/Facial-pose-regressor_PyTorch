# Facial-pose-regressor_PyTorch

This is a PyTorch implementation of a facial pose regressor written first in Keras in [Dr. Gualberto
's Github Repository](https://github.com/arnaldog12/Deep-Learning/tree/master/problems/Regressor-Face%20Pose). The author of the original code has wrote a great explanation on his [Medium article](https://medium.com/analytics-vidhya/face-pose-estimation-with-deep-learning-eebd0e62dbaf). Essentially, the code tries to examine the facial pose by analysing the pairwise landmark distances. I highly recommend one to read the article, as it is very clear and easy-to-follow.
(just in case you didn't notice: you can click the Dr. Gualberto's name above to reach his Github repository)

As of the original code, Dr. Gualberto's code computes the facial position from its 3D principal axes (yaw, roll and pitch) only with 3 dense layers. However, this implementation is deeper and a bit more complicated as the model uses 4 2D convolution layers and 2 fully connected layers. 

The **model_eval.py** file contains the same functions Dr. Gualberto used in his Jupyter Notebook as 1) I could not undo my action of already seeing his code and 2) my code was worse than his haha...
In the model_eval.py file, one may need to change the following for his or her training procedures:
- model_number : this is the epoch one wants to reach to test the model
- model_file   : this is the file name of the saved model, if needed, please change this filename
- model_directory: the directory where the model is saved (it has to match the one specified by **simple_pose_estimator.py**, i.e. the main program)
- the_pics     : this is a list of pictures to be tested, this has to be changed if the number of pictures or the names of the pictures are different

I thank Dr. Gualberto for the code and the article as I was struggling quite a bit in this task. 
(I was about to purchase $500+ LIDAR camera before I encountered his Medium article!)
