#! /usr/bin/env python

# usage: python3 ldr2urdf.py hoge.ldr

import os, sys, getopt
#import pathlib
#import math
import copy
import numpy
import transformations as TF
import UrdfTemplates as TP

# 1[LDU] = 0.4 [mm]
LDU2MM = 0.4
LDU2M  = 0.4 * 0.001

g_dae_path = "./new_parts_center/"
g_root = 'map'  # /base_link, map, ...

def inverseHomogeneous(mat):
    # mat is array
    # chech size of mat (whether homogeneous matrix or not)
    numColumn = mat.shape[0]
    numRow = mat.shape[0]
    if numColumn != 4 or numRow != 4:
        print("Size of matrix is wrong")
        return

    R = mat[0:3,0:3]
    T = mat[0:3,3:4]
    Rinv = numpy.linalg.inv(R)
    #Rinv = numpy.matrix.transpose(R)
    RinvT = numpy.dot(Rinv, T)
    Hinv = TF.identity_matrix()     # create 4x4 matrix
    Hinv[0:3,0:3] = Rinv
    Hinv[0:3,3:4] = -RinvT
    return Hinv


class Mimic():
    def __init__(self):
        self.is_mimic = False
        self.joint = ''         # ex: 'ref_205_joint'
        self.multiplier = 1.0
        self.offset = 0.0

class Xyz():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        
class Axis():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        
class Rpy():
    def __init__(self):
        self.r = 0.0
        self.p = 0.0
        self.y = 0.0

class Origin():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.r = 0.0
        self.p = 0.0
        self.y = 0.0
        
class DaePart():
    def __init__(self):
        self.name = ''   # ex: 99550c01 is EV3 Large Motor Case
        self.ref_id = -1        # ref_id
        self.ref_link = ''      # 'ref_xxx_link' (xxx is ref_id)
        self.color = 0
        self.rosHdae = numpy.identity(4) # H matrix in ROS coordinate system
        self.linkHdae = numpy.identity(4) # H matrix in link
        self.is_parent_joint = False
        self.is_child_joint = False
        self.is_root = False
        self.joint = Joint()    # joint to parent part
        self.parent_ref_link = ''   # ex: ref_xxx_link (root part's link name in each link)
        self.root_link = ''

    def print(self):
        if self.is_parent_joint == True:
            print("   parent joint")
        if self.is_child_joint == True:
            print("   child joint")
        # print("rHd = ", self.rHd)
        # print("rHd = ", self.lHd)
        # index = 0
        # for i in self.parts:
        #     print("part[%d]:" % index)
        #     i.print()
        #     index += 1
    

class Joint():
    def __init__(self):
        # 0 !LEOCAD GROUP BEGING Joint "id" "joint_type" "parent/child_id" continuous a_x a_y a_z
        self.joint_type = 'fixed'
        self.id = -1            # id defined in ldr by user (not ref_xxx_joint)
        self.parent_id = -1     # ex: parent "id" continuous 1 0 0
        self.child_id = -1      # ex: child "id" continuous 1 0 0
        self.mimic = Mimic()
        self.origin = Origin()
        self.axis = Axis()

class Link():
    def __init__(self):
        # revolute/continuous/prismatic/fixed/floating/planar
        self.parent_id = -1                  # id is defined in ldr
        self.self_id   = -1                  # id is defined in ldr
        self.child_id  = -1                  # id is defined in ldr
        self.rosHlink = numpy.identity(4)    # H matrix in ROS coorinate system
        self.parentHlink = numpy.identity(4) # H matrix from parent coordinate system
        self.parts = []                      # parts other than parents and child
        self.child_joint_part = DaePart()    # child part is always one
        self.root_part_ref_link = ''         # ref_xxx_link
        self.ref_joint = ''                  # ref_xxx_joint
        self.root_part = DaePart()           # root DaePart()
        self.parent_joints_ref_joint = []    # ref_xxx_joint (a link may have multiple joints.)
        self.child_joint = DaePart()
        self.parent_joints = []               # DaeParts() (a link may have multiple joints.)
            
class Urdf():
    def __init__(self):
        self.name = 'unknow'
        self.is_link  = False
        self.is_parent_joint = False
        self.is_parent_joint_part = False
        self.is_child_joint = False
        self.is_mimic_child_joint = False
        self.links = []
        
    def print(self):
        #print("name = %s" % self.name)
        index = 0
        print("num of links = %d" % len(self.links))
        for l in self.links:
            print("links[%d]:" % index)
            l.print()
            index += 1
            
    def load(self, obj_file, urdf_file):
        self.is_link = False
        print("Loading %s ..." % ("new_parts_center/" + obj_file), file=sys.stderr)
        if os.path.isfile("new_parts_center/" + obj_file) != True:
            print("%s not found" % "new_parts_center/" + obj_file, file=sys.stderr)
            return

        fp_urdf = open("urdf/" + urdf_file, "w")
        ### xml Header
        print("<?xml version=\"1.0\" ?>", file = fp_urdf)
        print("<!-- This file was generated from %s. -->" % ("new_parts_center/" + obj_file), file = fp_urdf)
        print("<robot name=\"%s\">" % urdf_file, file = fp_urdf)
        print("", file = fp_urdf)
        
        scale = 0.001
        d = {
            'refID' : "%s" % (obj_file),
            'mesh' : "%s" % ("../new_parts_center/" + obj_file),
            'm_scale' :'%s' % str(scale),
        }
        print(TP.link %d, file = fp_urdf)
        print("</robot>", file = fp_urdf)
        fp_urdf.close()
    


def main(argv, stdout, environ):
    all_obj_files = os.listdir("/home/jmji/DexterousHandEnvs/assets/urdf/objects/lego/new_parts_center")
    for obj_file in all_obj_files:
        ldr_file_body, ldr_file_ext = os.path.splitext(obj_file)
        urdf_file = ldr_file_body + ".urdf"
        urdf = Urdf()
        urdf.load(obj_file, urdf_file)
        
if __name__ == "__main__":
    main(sys.argv, sys.stdout, os.environ)
