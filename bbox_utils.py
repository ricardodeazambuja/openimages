import os.path
from collections.abc import Mapping
import json
import sys

import requests
import pandas as pd

import downloader as dl


class bboxUtils():
  def __init__(self, download_folder=""):
    self.download_folder = download_folder

    print("Downloading https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv")
    self.download_file("https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv", download_folder, verbose=True)
    self.bbox_df = pd.read_csv(os.path.join(download_folder,"class-descriptions-boxable.csv"), header=None)
    self.bbox_df.columns = ['Code', 'Label']

    print("Downloading https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json")
    self.download_file("https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json", download_folder, verbose=True)
    with open("bbox_labels_600_hierarchy.json","r") as f:
      self.bbox_h_df = json.load(f)

    print("Downloading https://storage.googleapis.com/openimages/v6/oidv6-relationship-triplets.csv")
    self.download_file("https://storage.googleapis.com/openimages/v6/oidv6-relationship-triplets.csv", download_folder, verbose=True)
    self.triplets_df = pd.read_csv(os.path.join(download_folder,"oidv6-relationship-triplets.csv"))
    self.triplets_df.columns = ['Code1', 'Code2', 'Relationship']

    self._code2label = dict(zip(list(self.bbox_df.iloc[:,0]),list(self.bbox_df.iloc[:,1])))
    self._code2label['/m/0bl9f'] = "Entity"
    
    self._name2code = dict(zip(list(self.bbox_df.iloc[:,1]),list(self.bbox_df.iloc[:,0])))
    self._name2code['Entity'] = "/m/0bl9f"

  def _get_bbox_files(self, split="train"):
    if split == "train":
      print("Downloading https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv")
      self.download_file("https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv", self.download_folder, verbose=True)
    elif split == "test":
      print("Downloading https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv")
      self.download_file("https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv", self.download_folder, verbose=True)
    elif split == "validation":
      print("Downloading https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv")
      self.download_file("https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv", self.download_folder, verbose=True)
    else:
      assert any([split==i for i in ["train", "test", "validation"]]), f"Wrong split ({split})! Try train, test or validation"

  def _get_vrd_files(self, split="train"):
    if split == "train":
      print("Downloading https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-vrd.csv")
      self.download_file("https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-vrd.csv", self.download_folder, verbose=True)
    elif split == "test":
      print("Downloading https://storage.googleapis.com/openimages/v6/oidv6-test-annotations-vrd.csv")
      self.download_file("https://storage.googleapis.com/openimages/v6/oidv6-test-annotations-vrd.csv", self.download_folder, verbose=True)
    elif split == "validation":
      print("Downloading https://storage.googleapis.com/openimages/v6/oidv6-validation-annotations-vrd.csv")
      self.download_file("https://storage.googleapis.com/openimages/v6/oidv6-validation-annotations-vrd.csv", self.download_folder, verbose=True)
    else:
      assert any([split==i for i in ["train", "test", "validation"]]), f"Wrong split ({split})! Try train, test or validation"

  def code2label(self, code):
    tmp = self.bbox_df.loc[self.bbox_df['Code']==code]['Label']
    return tmp.item() if len(tmp) else None

  def label2code(self, label):
    tmp = self.bbox_df.loc[self.bbox_df['Label']==label]['Code']
    return tmp.item() if len(tmp) else None

  def download_file(self, url, download_folder="", verbose=False, overwrite=False):
    # Test if file already exists
    filename = url.split('/')[-1]
    if (not os.path.isfile(os.path.join(download_folder, filename))) or overwrite:
      r = requests.get(url, allow_redirects=True)
      if r.status_code == 200:
        # Save file
        open(os.path.join(download_folder, filename), 'wb').write(r.content)
        if verbose: print(f"{filename} saved!")
      else:
        assert r.status_code == 200, f"Something went wrong with {url} [{r.status_code}]"
    else:
      if verbose: print(f"{filename} already exists")

  def search_by_name(self, search_term, case=False):
    return self.bbox_df.loc[self.bbox_df.iloc[:,1].str.contains(search_term, case)]

  def find_relationships(self, code):
    return self.triplets_df[(self.triplets_df['Code1']==code) | (self.triplets_df['Code2']==code)]

  def find_parents(self, code, level=[]):
    return list(self._find_parents(self.bbox_h_df, code, level))

  # based on https://stackoverflow.com/a/5071489/7658422
  def _find_parents(self, bbox_h_df, code, level=[]):
    if isinstance(bbox_h_df, Mapping):
      if code in bbox_h_df.values(): # test against LabelName
        if isinstance(level, list):
          yield [self.code2label(li) for li in level]
        else:
          yield self.code2label(level)
      if 'Subcategory' in bbox_h_df.keys():
        for found in self._find_parents(bbox_h_df['Subcategory'], code, level=level + [bbox_h_df['LabelName']]):
          yield found
      if 'Part' in bbox_h_df.keys():
        for found in self._find_parents(bbox_h_df['Part'], code, level=level + [bbox_h_df['LabelName']]):
          yield found
    elif isinstance(bbox_h_df, list):
      for l in bbox_h_df:
        for found in self._find_parents(l, code, level=level):
          yield found

  def get_bboxes(self, split, codes):
    assert isinstance(codes, list), "codes should be a list of codes..."
    if split == "train":
      filename = os.path.join(self.download_folder, "oidv6-train-annotations-bbox.csv")
    elif split == "test":
      filename = os.path.join(self.download_folder, "test-annotations-bbox.csv")
    elif split == "validation":
      filename = os.path.join(self.download_folder, "validation-annotations-bbox.csv")
    else:
      assert any([split==i for i in ["train", "test", "validation"]]), f"Wrong split ({split})! Try train, test or validation"

    self._get_bbox_files(split)

    return self._get_rows(codes, filename)

  def get_relationships(self, split, codes):
    assert isinstance(codes, list), "codes should be a list of codes..."
    if split == "train":
      filename = os.path.join(self.download_folder, "oidv6-train-annotations-vrd.csv")
    elif split == "test":
      filename = os.path.join(self.download_folder, "oidv6-test-annotations-vrd.csv")
    elif split == "validation":
      filename = os.path.join(self.download_folder, "oidv6-validation-annotations-vrd.csv")
    else:
      assert any([split==i for i in ["train", "test", "validation"]]), f"Wrong split ({split})! Try train, test or validation"

    self._get_vrd_files(split)

    return self._get_rows(codes, filename)

  def _get_rows(self, codes, dataset):
    rows = []
    with open(dataset, "r") as f:
      print(f"Searching for {codes}...")
      header = next(f).rstrip('\n')
      for l in f:
        if all([code in l for code in codes]):
          rows.append(l.rstrip('\n').split(','))
    return pd.DataFrame(rows, columns=header.split(','))

  def get_unique_img_ids(self, dataframe):
    return dataframe['ImageID'].unique()

  # modified from downloader.py (download_all_images)
  def get_images(self, img_ids, split, img_folder, num_processes=5):
    bucket = dl.boto3.resource(
          's3', config=dl.botocore.config.Config(
              signature_version=dl.botocore.UNSIGNED)).Bucket(dl.BUCKET_NAME)

    if not os.path.exists(img_folder):
      os.makedirs(img_folder)

    try:
      image_list = list(
          dl.check_and_homogenize_image_list([f"{split}/{img_id}" for img_id in img_ids])
          )
    except ValueError as exception:
      sys.exit(exception)

    progress_bar = dl.tqdm.tqdm(
        total=len(image_list), desc='Downloading images', leave=True)
    
    with dl.futures.ThreadPoolExecutor(
        max_workers=num_processes) as executor:
      all_futures = [
          executor.submit(dl.download_one_image, bucket, split_i, image_id,
                          img_folder) for (split_i, image_id) in image_list
      ]
      for future in dl.futures.as_completed(all_futures):
        future.result()
        progress_bar.update(1)
    progress_bar.close()
