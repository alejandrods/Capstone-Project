from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import rospy
import cv2

class TLClassifier(object):
    def __init__(self, is_site):
        if is_site:
            PATH_TO_MODEL = 'frozen_inference_graph_site.pb'
        else:
            PATH_TO_MODEL = 'frozen_inference_graph_sim.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)
        return

    def get_classification(self, img):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_debug = np.copy(img)
            rect_img = np.zeros_like(img_debug)
            img_expanded = np.expand_dims(img, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
            confidence = round(scores[0][0]*100)
            if (confidence < 50.0):
                signal = TrafficLight.UNKNOWN
                signal_str = 'Unknown'
                confidence = 100 - confidence
            elif (classes[0][0] == 1):
                signal = TrafficLight.GREEN
                signal_str = 'Green'
                overlay_color = (0,255,0)
            elif (classes[0][0] == 2):
                signal = TrafficLight.RED
                signal_str = 'Red'
                overlay_color = (255,0,0)
            elif (classes[0][0] == 3):
                signal = TrafficLight.YELLOW
                signal_str = 'Yellow'
                overlay_color = (255,255,0)
            else:
                signal = TrafficLight.UNKNOWN
                signal_str = 'Unknown'
            if signal == TrafficLight.GREEN or signal == TrafficLight.RED or signal == TrafficLight.YELLOW:
                height, width, channels = img.shape
                if (round(scores[0][0]*100) > 50.0):
            	    cv2.rectangle(rect_img, (int(boxes[0,0,1]*width),int(boxes[0,0,0]*height)),(int(boxes[0,0,3]*width),int(boxes[0,0,2]*height)), overlay_color, thickness=-1)
                    cv2.putText(rect_img,str(round(scores[0][0]*100)),(int(boxes[0,0,1]*width),int(boxes[0,0,0]*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,255,255),2,cv2.LINE_AA)
                if (round(scores[0][1]*100) > 50.0):
                    cv2.rectangle(rect_img, (int(boxes[0,1,1]*width),int(boxes[0,1,0]*height)),(int(boxes[0,1,3]*width),int(boxes[0,1,2]*height)), overlay_color, thickness=-1)
                    cv2.putText(rect_img,str(round(scores[0][1]*100)),(int(boxes[0,1,1]*width),int(boxes[0,1,0]*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,255,255),2,cv2.LINE_AA)
                if (round(scores[0][2]*100) > 50.0):
                    cv2.rectangle(rect_img, (int(boxes[0,2,1]*width),int(boxes[0,2,0]*height)),(int(boxes[0,2,3]*width),int(boxes[0,2,2]*height)), overlay_color, thickness=-1)
                    cv2.putText(rect_img,str(round(scores[0][2]*100)),(int(boxes[0,2,1]*width),int(boxes[0,2,0]*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,255,255),2,cv2.LINE_AA)
                cv2.addWeighted(img_debug, 1.0, rect_img, 0.5, 0, img_debug)
                

            #rospy.logdebug('signal:%s  confidence:%6.6f. ', signal_str, confidence)
        
        #TODO implement light color prediction
        return signal, img_debug
