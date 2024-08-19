import leap
import time
import cv2
import pdb
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

leap_hands = []

class LeapListener(leap.Listener):
    def __init__(self,):
        self.old_time = time.time()
        self.new_time = time.time()

    def on_connection_event(self, event):
        print("Connected")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()

        print(f"Found device {info.serial}")

    def on_tracking_event(self, event):
        global leap_hands
        leap_hands = event.hands

def get_hand(mode = 'right', conf_thre = 0.3):
    left_hands = []
    right_hands = []
    for hand in leap_hands:
        if hand.confidence > conf_thre:
            if str(hand.type) == "HandType.Left":
                left_hands.append(hand)
            elif str(hand.type) == "HandType.Right":
                right_hands.append(hand)
    if len(left_hands) >= 1:
        left_hands = sorted(left_hands, key=lambda x: x.confidence, reverse=True)
        left_hand = left_hands[0]
    else:
        left_hand =None
    if len(right_hands) >= 1:
        right_hands = sorted(right_hands, key=lambda x: x.confidence, reverse=True)
        right_hand = right_hands[0]
    else:
        right_hand = None

    assert mode == 'right', "Only support the right hand mode now."
    if mode == 'right':
        return right_hand
    elif mode == 'left':
        return left_hand
    elif mode == 'both':
        return left_hand, right_hand
    else:
        raise ValueError("Not supported mode: {}".format(mode))

def make_orthogonal(v):
    other = np.array([1, 0, 0])
    proj = np.dot(other, v) * v
    orth = other - proj
    return orth / np.linalg.norm(orth)

def vector_to_quaternion(direction_vector, normal_vector):
    # Ensure the vectors are normalized
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    if not np.isclose(np.dot(direction_vector, normal_vector), 0):
        normal_vector = normal_vector + make_orthogonal(direction_vector)

    # Compute the axis of rotation (cross product)
    axis = np.cross(direction_vector, normal_vector)
    axis = axis / np.linalg.norm(axis)

    # Compute the angle of rotation (dot product)
    angle = np.arccos(np.clip(np.dot(direction_vector, normal_vector), -1.0, 1.0))

    # Compute the quaternion
    #quaternion = Quaternion(axis=[1, 0, 0], angle=-np.pi/2) * Quaternion(axis=axis, angle=angle)
    quaternion = Quaternion(axis=axis, angle=angle)

    return quaternion

if __name__ == "__main__":
    my_listener = LeapListener()
    leap_connection = leap.Connection()
    leap_connection.add_listener(my_listener)
    with leap_connection.open():
        leap_connection.set_tracking_mode(leap.TrackingMode.Desktop)
        while True:
            print(len(leap_hands))