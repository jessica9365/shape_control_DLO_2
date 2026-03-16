#!/usr/bin/env python3
import time
import rtde_control
import dashboard_client
import numpy as np
import rtde_receive
import utils.print_utils as print_utils
import threading
import csv

class URrobot(): # create class for ur robot
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip

        # default homing parameters
        self.home_pose = [0.2, 0, 0.5, np.pi, 0, 0] # x,y,z,rx,ry,rz
        self.home_vel = 0.05
        self.home_acc = 0.25

        self.is_online = False
        self.logs = []
        

        try:
            """ # not available for polyscope < 5.6
            self.dashboard = dashboard_client.DashboardClient(self.robot_ip)
            self.dashboard.connect()
            if not self.dashboard.isInRemoteControl():
                print_utils.logwarn('Please enable remote control access on PolyScope to continue...')
                while not self.dashboard.isInRemoteControl():
                    pass 
            """ 
            print_utils.logwarn(f"Please manually enable remote control access on PolyScope {self.robot_ip} to continue...")
            input()
            self.control = rtde_control.RTDEControlInterface(self.robot_ip)
            self.receive = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            print_utils.loginfo(f"Connected to UR robot at {self.robot_ip}")
            self.is_online = True
        except Exception as e:
            print_utils.logerr(e)
    
    # Function to check closeness
    def isAtPose(self, target, actual=None, tol=0.001):  # 1 mm tolerance
        if actual == None:
            actual = self.receive.getActualTCPPose()
        return all(abs(t - a) < tol for t, a in zip(target[:3], list(actual)[:3]))
        
    def home(self, home_pos=None, home_vel=None, home_acc=None, auto_start=False):
        if home_pos == None:
            home_pos = self.home_pos
        if home_vel == None:
            home_vel = self.home_vel
        if home_acc == None:
            home_acc = self.home_acc
        print_utils.loginfo(f"Moving to home position: {home_pos}")
        if not auto_start:
            input("Press Enter to continue...")
        self.control.moveL(home_pos, home_vel, home_acc)
        self.control.speedStop(home_acc)    # ensure motion stops
        if self.isAtPose(home_pos, self.receive.getActualTCPPose()):
            print_utils.loginfo("Reached home position")
            return True
        else:
            print_utils.logwarn("Failed to reach home position!")
            return False
        
    # def velocityJogControl(self, velocity_vector, acc):
    #     #velocity_vector[0] *= -1
    #     self.control.jogStart(velocity_vector, acc=acc, feature=rtde_control.RTDEControlInterface.FEATURE_BASE)
    #

    def velocityJogControl(self, velocity_vector, acc, timeout=10):
        # velocity_vector[0] *= -1
        self.control.jogStart(velocity_vector, acc=acc, feature=rtde_control.RTDEControlInterface.FEATURE_BASE)

        def stopAfter(timeout):
            time_start = time.time()
            while time.time() - time_start < timeout:
                continue
            self.control.speedStop(acc)

        if timeout:
            t = threading.Thread(target=stopAfter, args=(timeout))
            t.start()

    def velocityJogControlStop(self):
        self.control.jogStop()

    def velocityControl(self, velocity_vector, acc, duration):
        self.control.speedL(velocity_vector, acc, duration)
        time.sleep(duration)
        self.control.speedStop(acc)

    def pointToPointMove(self, pose, vel, acc, wait=True):
        if wait:
            self.control.moveL(pose, vel, acc)
            self.control.speedStop(acc)
            if self.isAtPose(pose):
                return True
            else:
                print_utils.logwarn(f"Failed to reach pose! {pose}")
                return False
        else:
            t = threading.Thread(target=self.control.moveL, args=(pose, vel, acc))
            t.start()
            return t
        
    def waitStop(self, tol=0.001):
        while True:
            speed = self.receive.getActualTCPSpeed()
            if np.linalg.norm(speed) < tol:
                break
        self.control.speedStop(self.home_acc)

    def terminate(self):
        try:
            if self.is_online: # check if connected first
                if self.control:
                    self.control.disconnect()
                if self.receive:
                    self.receive.disconnect()
        except Exception as e:
            print_utils.logerr(e)
    
    def save_logs(self, filename):
        if not self.logs:
            print_utils.logwarn("No logs to save")
            return
        keys = list(self.logs[0].keys())
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.logs)
        print_utils.loginfo(f"Saved logs to {filename}")
