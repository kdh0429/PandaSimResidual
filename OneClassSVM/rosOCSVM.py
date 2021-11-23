import sys
sys.path.append("/opt/ros/melodic/lib/python2.7/dist-packages")
import rospy

from std_msgs.msg import Float32MultiArray

from numpy import genfromtxt

from sklearn.svm import OneClassSVM
import _pickle as cPickle

# Data
num_train_data = 70000
num_joint = 7

class PandaOCSVM:

    def __init__(self):
        with open('ocsvm_residual.pkl', 'rb') as fid:
            self.clf = cPickle.load(fid)
        print("Loaded Model!")

        self.output_max = genfromtxt('../MinMax.csv', delimiter=",")[0]

        # In ROS, nodes are uniquely named. If two nodes with the same
        # name are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        rospy.Subscriber("/panda/residual", Float32MultiArray, self.callback)

    def callback(self, data):
        self.residual = [data.data / self.output_max]
        collision_state = self.clf.predict(self.residual)
        if collision_state == -1:
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