#! /usr/bin/env python

# usage: python3 ldr2urdf.py hoge.ldr

import os, sys, getopt
#import pathlib
#import math
import copy
import numpy
import transformations as TF
import UrdfTemplates as TP
            
class Urdf():
    def __init__(self):
        self.name = 'unknow'
        self.is_link  = False
        self.is_parent_joint = False
        self.is_parent_joint_part = False
        self.is_child_joint = False
        self.is_mimic_child_joint = False
        self.links = []
        
    def load(self, ldr_file_body, obj_file, urdf_file):

        fp_urdf = open("urdf/" + urdf_file, "w")
        ### xml Header
        print("<?xml version=\"1.0\" ?>", file = fp_urdf)
        print("<!-- This file was generated from %s. -->" % ("obj_file/" + obj_file), file = fp_urdf)
        print("<robot name=\"%s\">" % urdf_file, file = fp_urdf)
        print("", file = fp_urdf)
        
        scale = 0.0015
        d = {
            'refID' : "%s" % (obj_file),
            'mesh' : "../origin_obj/%s" % (os.path.join(ldr_file_body, obj_file)),
            'm_scale' :'%s' % str(scale),
        }
        print(TP.link %d, file = fp_urdf)
        print("</robot>", file = fp_urdf)
        fp_urdf.close()

def main(argv, stdout, environ):
    all_dir = os.listdir("/home/jmji/DexterousHandEnvs/assets/urdf/leoCAD/origin_obj/")
    print(all_dir)
    for dir in all_dir:
        all_obj_files = os.listdir("/home/jmji/DexterousHandEnvs/assets/urdf/leoCAD/origin_obj/{}".format(dir))
        for obj_file in all_obj_files:
            ldr_file_body, ldr_file_ext = os.path.splitext(obj_file)
            if ldr_file_ext != ".obj":
                continue
            urdf_file = ldr_file_body + ".urdf"
            urdf = Urdf()
            urdf.load(ldr_file_body, obj_file, urdf_file)
            
if __name__ == "__main__":
    main(sys.argv, sys.stdout, os.environ)
