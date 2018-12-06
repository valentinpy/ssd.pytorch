import sys
import os
from glob import glob
import argparse

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def main():
    args = arg_parser()
    argcheck(args)
    parse_annotations(args)

def arg_parser():
    parser = argparse.ArgumentParser(description='Script to generate imageSet files for kaist')

    parser.add_argument('--dataset_root', default="/home/valentinpy/data/kaist",
                        help='Dataset root directory path')

    parser.add_argument('--output_file', default=None,
                        help='Output file name')
    parser.add_argument('--output_folder', default="/home/valentinpy/data/kaist/rgbt-ped-detection/data/kaist-rgbt/imageSets/",
                        help='Output folder')

    parser.add_argument('--day_only', default=True, type=str2bool,
                        help='Only day')

    parser.add_argument('--use_set00_01_02', default=False, type=str2bool,
                        help='Use set00_01_02')

    parser.add_argument('--use_set03_04_05', default=False, type=str2bool,
                        help='Use set03_04_05')

    parser.add_argument('--use_set06_07_08', default=False, type=str2bool,
                        help='Use set06_07_08')

    parser.add_argument('--use_set09_10_11', default=False, type=str2bool,
                        help='Use set09_10_11')

    parser.add_argument('--only_person', default=False, type=str2bool,
                        help="Filter out images which do not contain exclusively 'person'")

    parser.add_argument('--min_annotation_height', default=-1, type=int,
                        help='Filter out images with annotation smaller as specified [px]')

    parser.add_argument('--remove_occlusions', default=False, type=str2bool,
                        help='Remove images with occlusions')

    parser.add_argument('--remove_no_person_annotations', default=True, type=str2bool,
                        help='Remove images without person annoations')

    parser.add_argument('--max_images', default=-1, type=int,
                        help='Max number of images [-1: unlimited]')


    args = parser.parse_args()

    return args


def argcheck(args):
        if not os.path.exists(args.dataset_root):
            print("dataset not found!")
            sys.exit(-1)

        if not os.path.exists(args.output_folder):
            print("output folder not found!")
            sys.exit(-1)

        if args.day_only is not True:
            print("Only day supported yet")
            sys.exit(-1)

        set_used = 0
        if args.use_set00_01_02:
            set_used+=1
        if args.use_set03_04_05:
            set_used+=1
        if args.use_set06_07_08:
            set_used+=1
        if args.use_set09_10_11:
            set_used+=1

        if set_used is not 1:
            print("only one set can be chosen at a time")
            sys.exit(-1)

        if args.max_images == -1:
            args.max_images = sys.maxsize

def filter(annofilename, args):

    # flags
    keep_file = False
    person_detected = False
    people_detected = False
    person_not_sure_detected = False
    cyclist_detected = False
    only_person_detected = False
    occlusion_detected = False
    too_small = False

    with open(annofilename) as annofile:
        for annoline in annofile: # loop for each line of each annofile

            if not annoline.startswith("%"): # if not a comment
                # split annotation lines
                annosplit = annoline.split(" ")
                if len(annosplit) > 5:
                    # for each file, exact flags
                    if annosplit[0] == 'person':  # only keep images which contains a "person"
                        person_detected = True
                    elif annosplit[0] == 'people':
                        people_detected = True
                    elif annosplit[0] == 'person?':
                        person_not_sure_detected = True
                    elif annosplit[0] == 'cyclist':
                        cyclist_detected = True
                    else:
                        print("Annotation not recognized!")
                        sys.exit(-1)

                    #print(annosplit[5])
                    if int(annosplit[5]) != 0:
                        occlusion_detected = True

                    if int(annosplit[4]) < args.min_annotation_height:
                        too_small = True

    if (person_detected) and (not person_not_sure_detected) and (not people_detected) and (not cyclist_detected):
        only_person_detected = True

    #use args filter to cancel useless excluding flags
    if not args.only_person:
        only_person_detected = True
    if not args.remove_occlusions:
        occlusion_detected = False
    if args.min_annotation_height == -1:
        too_small = False
    if not args.remove_no_person_annotations:
        person_detected = True

    # according to flags, do we keep this entry ?
    keep_file = only_person_detected and (not occlusion_detected) and (not too_small) and person_detected

    return keep_file

def parse_annotations(args):
    i=0

    # get all annotation files in the sets (set 0-2 or set 6-8)
    annotations_folder = os.path.join(args.dataset_root, 'rgbt-ped-detection/data/kaist-rgbt/annotations')
    annotation_files = [y for x in os.walk(annotations_folder) for y in glob(os.path.join(x[0], '*.txt'))]

    if args.use_set00_01_02:
        annotation_files = [x for x in annotation_files if (("set00" in x) or ("set01" in x) or ("set02" in x))] #train day
    elif args.use_set06_07_08:
        annotation_files = [x for x in annotation_files if (("set06" in x) or ("set07" in x) or ("set08" in x))] #test day
    elif args.use_set03_04_05:
        annotation_files = [x for x in annotation_files if (("set03" in x) or ("set04" in x) or ("set05" in x))] #train night
    elif args.use_set09_10_11:
        annotation_files = [x for x in annotation_files if (("set09" in x) or ("set10" in x) or ("set11" in x))] #test night
    else:
        sys.exit(-1)

    #open output imageSet file
    with open(os.path.join(args.output_folder, args.output_file), "w") as imageset_file:
        imageset_file.write("# ImageSet automatically generated by 'kaist_create_imageset.py' " + repr(args) + "\n")
        for annofilename in annotation_files:
            if filter(annofilename, args) and (i < args.max_images):
                newline = "{}/{}/{}\n".format((annofilename.split('/'))[-3], (annofilename.split('/'))[-2], (annofilename.split('/'))[-1].split('.')[0])
                imageset_file.write(newline) # todo output syntax
                i+= 1
    print("Finished\nnumber of files kept: {}/{}\nOutput file is {}".format(i, len(annotation_files),  os.path.join(args.output_folder, args.output_file)))


if __name__ == '__main__':
    main()
