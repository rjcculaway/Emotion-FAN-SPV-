import os
import shutil

def main(frame_dir, eval_dir):
  file_handle = open("out.txt", "w")
  for identity in [ f.path for f in os.scandir(frame_dir) if f.is_dir() ]:
    frame_folders = [f.path for f in os.scandir(identity) if f.is_dir()]
    # Move all videos frames greater than 020 to the eval folder
    for frames_folder in filter(lambda path: int(path.split('-')[-1]) > 20, frame_folders):
      file_handle.write(frames_folder + ' to ' + os.path.join(eval_dir, os.path.basename(identity), os.path.basename(frames_folder)) + '\n')
      shutil.move(frames_folder, os.path.join(eval_dir, os.path.basename(identity), os.path.basename(frames_folder)))
  file_handle.close()    

if __name__ == "__main__":
  frame_dir = '../frame/train_mead'
  eval_dir = '../frame/eval_mead'
  main(frame_dir, eval_dir)