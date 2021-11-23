import sys
sys.path.append("/opt/ros/melodic/lib/python2.7/dist-packages")
import rospy

from std_msgs.msg import Float32MultiArray

import numpy as np
from numpy import genfromtxt

from sklearn.svm import OneClassSVM
import _pickle as cPickle

# Data
num_train_data = 70000
num_joint = 7
num_seq = 5

class PandaOCSVM:

    def __init__(self):
        with open('./model/ocsvm_residual.pkl', 'rb') as fid:
            self.clf = cPickle.load(fid)
        print("Loaded Model!")

        self.output_max = np.tile(genfromtxt('./data/MinMax.csv', delimiter=",")[0],num_seq)
        self.output_min = np.tile(genfromtxt('./data/MinMax.csv', delimiter=",")[1],num_seq)

        self.threshold = -0.003

        print("OutMax: ", self.output_min)

        rospy.Subscriber("/panda/residual", Float32MultiArray, self.callback)

    def callback(self, data):
        self.residual = [2*(data.data - self.output_min) / (self.output_max - self.output_min) - 1]
        # collision_state = self.clf.predict(self.residual)
        anomaly_score = self.clf.decision_function(self.residual)
        if anomaly_score < self.threshold:
            print("Collision")
        
    def listener(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")


if __name__ == '__main__':
    rospy.init_node('one_class_svm', anonymous=True)
    oneClassSVM = PandaOCSVM()
    oneClassSVM.listener()