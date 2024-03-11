import fitz
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import sys
from typing import Tuple

def OCR(filename1: str, filename2: str, page: int) -> Tuple[pd.DataFrame, np.ndarray]:    
  def ocr(filename: str, PAGE: int) -> Tuple[pd.DataFrame, np.ndarray]:
    pdf = fitz.Document(filename)
    MAG = 1
    CROP = (0, None, 0, None) #(top, bottom, left, right)
    pm = pdf[PAGE].get_pixmap(matrix=fitz.Matrix(MAG, MAG), alpha=False)
    img_pil = Image.frombytes('RGB', [pm.width, pm.height], pm.samples)
    crop_np = np.array(img_pil)[CROP[0]:CROP[1],CROP[2]:CROP[3]] # (2, 2)

    blocks = pdf[PAGE].get_text('blocks')
    blocks_s = sorted(blocks, key=lambda k : (k[1], k[0]))
    data = [
      {
        'x0' : block[0],
        'y0' : block[1],
        'x1' : block[2],
        'y1' : block[3],
        'text' : block[4],
      } for block in blocks_s
    ]

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    df = pd.DataFrame(data)
    
    return df, crop_np
  
  def compare(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame]:
    # Merge the two dataframes on the columns with numbers
    df = pd.merge(df1, df2, on=['x0', 'y0', 'x1', 'y1'], how='outer', suffixes=('_df1', '_df2'))

    # Create a new column that indicates whether the text is equal
    df['text_equal']= df['text_df1']== df['text_df2']
    
    df = df[df['text_equal'] == False]
    
    list1 = []
    list2 = []
    for i in range(len(df)):
      for j in range(i+1, len(df)):
        if not df.iloc[i]['text_equal']:
          # threshold value of 100 to check if the text box is the same one
          threshold = 10
          if abs(df.iloc[i]['x0'] - df.iloc[j]['x0']) <= threshold and abs(df.iloc[i]['y0'] - df.iloc[j]['y0']) <= threshold:
            # if the text is same
            if df.iloc[i]['text_df1'] == df.iloc[j]['text_df2']:
              list1.append([df.index[i], df.index[j]])
            # if the text is different
            else:
              list2.append([df.index[i], df.index[j]])
    list1.sort()
    list2.sort()
    
    list1_flatten = [item for sub_list in list1 for item in sub_list]
    for item in list1_flatten:
      df = df.drop(item)
    
    for pair in list2:
      if pair[0] in list1_flatten or pair[1] in list1_flatten:
        continue
      row1 = df.loc[pair[0]]
      row2 = df.loc[pair[1]]
      
      combined_row = row2.combine_first(row1)
      df.loc[pair[1]] = combined_row
      df = df.drop(pair[0])

    return df
  
  df1, img1 = ocr(filename1, page)
  df2, img2 = ocr(filename2, page)
  df = compare(df1, df2)

  return df, img1, img2

def plot(df: pd.DataFrame, page: int, image1: np.ndarray, image2: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
  # Modified text box: blue color
  for i, _ in df.iterrows():
    start_point = (int(df.loc[i]['x0']), int(df.loc[i]['y0']))
    end_point = (int(df.loc[i]['x1']), int(df.loc[i]['y1']))
    image1 = cv2.rectangle(image1, start_point, end_point, color=(255, 0, 0), thickness=1)
    image2 = cv2.rectangle(image2, start_point, end_point, color=(255, 0, 0), thickness=1)
    
  df.index = str(page+1) + '_' + df.index.astype(str)
    
  # # Define the output file name
  output_file1 = "output1" + "_" + str(page+1) + ".png"
  output_file2 = "output2" + "_" + str(page+1) + ".png"

  # Save the image
  cv2.imwrite(output_file1, image1)
  cv2.imwrite(output_file2, image2)
  
  return df, image1, image2

def main():
  f1 = sys.argv[1]
  f2 = sys.argv[2]

  
  pdf1 = fitz.Document(f1)
  pdf2 = fitz.Document(f2)
  page_num1 = pdf1.page_count
  page_num2 = pdf2.page_count
  page_num = min(page_num1, page_num2)
  df_list = []
  img_list = []
  
  for i in range(page_num):
    df, image1, image2= OCR(f1, f2, i)
    df, image1, image2 = plot(df, i, image1, image2)
    df_list.append(df)
    img_list.append(image1)
    img_list.append(image2)
  df_final = pd.concat(df_list)
  
  for j in range(page_num, page_num1):
    pdf = fitz.Document(f1)
    MAG = 1
    CROP = (0, None, 0, None) #(top, bottom, left, right)
    pm = pdf[j].get_pixmap(matrix=fitz.Matrix(MAG, MAG), alpha=False)
    img_pil = Image.frombytes('RGB', [pm.width, pm.height], pm.samples)
    crop_np = np.array(img_pil)[CROP[0]:CROP[1],CROP[2]:CROP[3]] # (2, 2)
    img_list.append(crop_np)
    df_final.loc[str(j+1) + '_f1'] = None
  
  for k in range(page_num, page_num2):
    pdf = fitz.Document(f2)
    MAG = 1
    CROP = (0, None, 0, None) #(top, bottom, left, right)
    pm = pdf[k].get_pixmap(matrix=fitz.Matrix(MAG, MAG), alpha=False)
    img_pil = Image.frombytes('RGB', [pm.width, pm.height], pm.samples)
    crop_np = np.array(img_pil)[CROP[0]:CROP[1],CROP[2]:CROP[3]] # (2, 2)
    img_list.append(crop_np)
    df_final.loc[str(k+1) + '_f2'] = None
  
  return df_final, img_list
  
if __name__ == '__main__':
  main()
